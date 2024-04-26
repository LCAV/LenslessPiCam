import hydra
import numpy as np
from lensless.hardware.slm import set_programmable_mask, adafruit_sub2full, full2subpattern
from lensless.hardware.utils import capture
import torch
from lensless import ADMM
from lensless.utils.io import save_image
from lensless.hardware.trainable_mask import AdafruitLCD
from lensless.utils.io import load_image


@hydra.main(version_base=None, config_path="../../configs", config_name="defaults_recon")
def digicam(config):
    measurement_fp = "1.png"
    # mask_fp = "mask_0.npy"
    mask_fp = None

    seed = 1
    rpi_username = None
    rpi_hostname = None
    mask_shape = [54, 26]
    mask_center = [57, 77]
    torch_device = "cuda:0"
    capture_config = {
        "exp": 0.8,
        "sensor": "rpi_hq",
        "script": "~/LenslessPiCam/scripts/measure/on_device_capture.py",
        "iso": 100,
        "config_pause": 1,
        "sensor_mode": "0",
        "nbits_out": 8,
        "nbits_capture": 12,
        "legacy": True,
        "gray": False,
        "fn": "raw_data",
        "bayer": True,
        "awb_gains": [1.6, 1.2],
        "rgb": True,
        "down": 8,
        "flip": True,
    }

    # load mask
    if mask_fp is not None:
        mask_vals = np.load(mask_fp)
    else:
        # create random mask within [0, 1]
        np.random.seed(seed)
        # mask_vals = np.random.rand(mask_shape[0], mask_shape[1])
        mask_vals = np.random.uniform(0, 1, mask_shape)

    # pattern = np.load("data/psf/adafruit_random_pattern_20231107_150902.npy")
    # mask_vals = full2subpattern(
    #     pattern=pattern,
    #     shape=[18, 26],
    #     center=[57, 77],
    #     slm="adafruit",
    # )

    # simulate PSF
    # # - random color filter
    # from waveprop.devices import slm_dict
    # from waveprop.devices import SLMParam as SLMParam_wp
    # slm_param = slm_dict["adafruit"]
    # color_filter = slm_param[SLMParam_wp.COLOR_FILTER]
    # color_filter = torch.from_numpy(color_filter.copy()).to(dtype=torch.float32)
    # color_filter = color_filter + 0.1 * torch.rand_like(color_filter)
    # - simulate
    mask = AdafruitLCD(
        initial_vals=torch.from_numpy(mask_vals.astype(np.float32)),
        sensor="rpi_hq",
        slm="adafruit",
        downsample=capture_config["down"],
        flipud=capture_config["flip"],
        # color_filter=color_filter,
    )
    psf = mask.get_psf().to(torch_device).detach()
    save_image(psf[0].cpu().numpy(), "digicam_psf.png")
    print(f"PSF shape: {psf.shape}")
    print("PSF saved to digicam_psf.png")

    if measurement_fp is not None:
        # load image
        img = load_image(
            measurement_fp,
            verbose=True,
        )

    else:
        # measure data
        # prepare full mask
        pattern = adafruit_sub2full(
            mask_vals,
            center=mask_center,
        )

        # set mask
        print("Setting mask")
        set_programmable_mask(
            pattern,
            "adafruit",
            rpi_username=rpi_username,
            rpi_hostname=rpi_hostname,
        )

        # capture
        print("Capturing")
        localfile, img = capture(
            rpi_username=rpi_username,
            rpi_hostname=rpi_hostname,
            verbose=False,
            **capture_config,
        )
        print(f"Captured to {localfile}")

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
    res = recon.apply(disp_iter=None, plot=False, n_iter=100)
    res_np = res[0].cpu().numpy()
    res_np = res_np / res_np.max()
    lensless_np = img[0].cpu().numpy()
    save_image(lensless_np, "digicam_raw.png")
    save_image(res_np, "digicam_recon.png")

    print("Done")


if __name__ == "__main__":
    digicam()
