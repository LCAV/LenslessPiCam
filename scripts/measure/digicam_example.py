"""
DigiCam example to remotely:
1. Set mask pattern.
2. Capture image.
3. Reconstruct image with simulated PSF.

TODO: display image. At the moment should be done with `scripts/measure/remote_display.py`

"""


import hydra
from hydra.utils import to_absolute_path
import numpy as np
from lensless.hardware.slm import set_programmable_mask, adafruit_sub2full
from lensless.hardware.utils import capture
import torch
from lensless import ADMM
from lensless.utils.io import save_image
from lensless.hardware.trainable_mask import AdafruitLCD
from lensless.utils.io import load_image, load_psf
from lensless.utils.image import gamma_correction


@hydra.main(version_base=None, config_path="../../configs", config_name="digicam_example")
def digicam(config):
    measurement_fp = config.capture.fp
    psf_fp = config.psf
    mask_fp = config.mask.fp
    seed = config.mask.seed
    rpi_username = config.rpi.username
    rpi_hostname = config.rpi.hostname
    mask_shape = config.mask.shape
    mask_center = config.mask.center
    torch_device = config.recon.torch_device
    capture_config = config.capture
    simulation_config = config.simulation

    # load mask
    if mask_fp is not None:
        mask_vals = np.load(to_absolute_path(mask_fp))
    else:
        # create random mask within [0, 1]
        np.random.seed(seed)
        mask_vals = np.random.uniform(0, 1, mask_shape)

    # create mask
    mask = AdafruitLCD(
        initial_vals=torch.from_numpy(mask_vals.astype(np.float32)),
        sensor=capture_config["sensor"],
        slm="adafruit",
        downsample=capture_config["down"],
        flipud=capture_config["flip"],
        use_waveprop=simulation_config.get("use_waveprop", False),
        scene2mask=simulation_config.get("scene2mask", None),
        mask2sensor=simulation_config.get("mask2sensor", None),
        deadspace=simulation_config.get("deadspace", True),
        # color_filter=color_filter,
    )

    # use measured PSF or simulate
    if psf_fp is not None:
        psf = load_psf(
            fp=to_absolute_path(psf_fp),
            downsample=capture_config["down"],
            flip=capture_config["flip"],
        )
        psf = torch.from_numpy(psf).type(torch.float32).to(torch_device)
    else:
        psf = mask.get_psf().to(torch_device).detach()
    psf_np = psf[0].cpu().numpy()
    psf_fp = "digicam_psf.png"

    gamma = simulation_config.get("gamma", None)
    if gamma is not None:
        psf_np = psf_np / psf_np.max()
        psf_np = gamma_correction(psf_np, gamma=gamma)

    save_image(psf_np, psf_fp)
    print(f"PSF shape: {psf.shape}")
    print(f"PSF saved to {psf_fp}")

    if measurement_fp is not None:
        # load image
        img = load_image(
            to_absolute_path(measurement_fp),
            verbose=True,
        )

    else:
        ## measure data
        # -- prepare full mask
        pattern = adafruit_sub2full(
            mask_vals,
            center=mask_center,
        )

        # -- set mask
        print("Setting mask...")
        set_programmable_mask(
            pattern,
            "adafruit",
            rpi_username=rpi_username,
            rpi_hostname=rpi_hostname,
        )

        # -- capture
        print("Capturing...")
        localfile, img = capture(
            rpi_username=rpi_username,
            rpi_hostname=rpi_hostname,
            verbose=False,
            **capture_config,
        )
        print(f"Captured to {localfile}")

    """ analyze image """
    print("image range: ", img.min(), img.max())

    """ reconstruction """
    # -- normalize
    img = img.astype(np.float32) / img.max()
    # prep
    img = torch.from_numpy(img)
    # -- if [H, W, C] -> [D, H, W, C]
    if len(img.shape) == 3:
        img = img.unsqueeze(0)
    if capture_config["flip"]:
        img = torch.rot90(img, dims=(-3, -2), k=2)

    # reconstruct
    print("Reconstructing")
    recon = ADMM(psf)
    recon.set_data(img.to(psf.device))
    res = recon.apply(disp_iter=None, plot=False, n_iter=config.recon.n_iter)
    res_np = res[0].cpu().numpy()
    res_np = res_np / res_np.max()
    lensless_np = img[0].cpu().numpy()
    save_image(lensless_np, "digicam_raw.png")
    save_image(res_np, "digicam_recon.png")

    print("Done")


if __name__ == "__main__":
    digicam()
