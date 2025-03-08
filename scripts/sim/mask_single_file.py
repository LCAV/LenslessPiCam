"""

Simulate a mask, simulate a measurement with it, and reconstruct the image.

Procedure is as follows:

1) Simulate the mask.
2) Simulate a measurement with the mask and specified physical parameters.
3) Reconstruct the image from the measurement.

Example usage:

Simulate FlatCam with separable simulation and Tikhonov reconstuction (https://arxiv.org/abs/1509.00116, Eq 7):
```
# MLS mask
python scripts/sim/mask_single_file.py mask.type=MLS simulation.flatcam=True recon.algo=tikhonov

# MURA mask
python scripts/sim/mask_single_file.py mask.type=MURA mask.n_bits=99 simulation.flatcam=True recon.algo=tikhonov
```

Using Torch
```
python scripts/sim/mask_single_file.py mask.type=MLS simulation.flatcam=True recon.algo=tikhonov use_torch=True
```

Simulate FlatCam with PSF simulation and Tikhonov reconstuction:
```
python scripts/sim/mask_single_file.py mask.type=MLS simulation.flatcam=False recon.algo=tikhonov
```

Simulate FlatCam with PSF simulation and ADMM reconstruction. Doesn't work well.
```
python scripts/sim/mask_single_file.py mask.type=MLS simulation.flatcam=False recon.algo=admm
```

Simulate Fresnel Zone Aperture camera with PSF simulation and ADMM reconstuction (https://www.nature.com/articles/s41377-020-0289-9):
Doesn't work well, maybe need to remove DC offset which hurts reconstructions?
```
python scripts/sim/mask_single_file.py mask.type=FZA recon.algo=admm recon.admm.n_iter=18
```

Simulate PhaseContour camera with PSF simulation and ADMM reconstuction (https://ieeexplore.ieee.org/document/9076617):
```
python scripts/sim/mask_single_file.py mask.type=PhaseContour recon.algo=admm recon.admm.n_iter=10
```

"""

import hydra
import warnings
from hydra.utils import to_absolute_path
from lensless.utils.io import load_image, save_image
from lensless.utils.image import rgb2gray, rgb2bayer, bayer2rgb
import numpy as np
import matplotlib.pyplot as plt
from lensless import ADMM
from lensless.utils.plot import plot_image
from lensless.eval.metric import mse, psnr, ssim, lpips
from waveprop.simulation import FarFieldSimulator
import os
from lensless.hardware.mask import CodedAperture, PhaseContour, FresnelZoneAperture
from lensless.recon.tikhonov import CodedApertureReconstruction
import torch


@hydra.main(version_base=None, config_path="../../configs/simulate", config_name="mask_sim_single")
def simulate(config):

    fp = to_absolute_path(config.files.original)
    assert os.path.exists(fp), f"File {fp} does not exist."

    # simulation parameters
    object_height = config.simulation.object_height
    scene2mask = config.simulation.scene2mask
    mask2sensor = config.simulation.mask2sensor
    sensor = config.simulation.sensor
    snr_db = config.simulation.snr_db
    downsample = config.simulation.downsample
    max_val = config.simulation.max_val

    image_format = config.simulation.image_format.lower()
    if image_format not in ["grayscale", "rgb"]:
        bayer = True
    else:
        bayer = False

    # 1) simulate mask
    mask_type = config.mask.type
    if mask_type.upper() in ["MURA", "MLS"]:
        mask = CodedAperture.from_sensor(
            sensor_name=sensor,
            downsample=downsample,
            method=mask_type,
            distance_sensor=mask2sensor,
            **config.mask,
        )
    elif mask_type.upper() == "FZA":
        mask = FresnelZoneAperture.from_sensor(
            sensor_name=sensor,
            downsample=downsample,
            distance_sensor=mask2sensor,
            **config.mask,
        )
    elif mask_type.lower() == "PhaseContour".lower():
        mask = PhaseContour.from_sensor(
            sensor_name=sensor,
            downsample=downsample,
            distance_sensor=mask2sensor,
            n_iter=config.mask.phase_mask_iter,
            **config.mask,
        )

    # 2) simulate measurement
    image = load_image(fp, verbose=True) / 255
    if config.use_torch:
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image).float()

    flatcam_sim = config.simulation.flatcam
    if flatcam_sim and mask_type.upper() not in ["MURA", "MLS"]:
        warnings.warn(
            "Flatcam simulation only supported for MURA and MLS masks. Using far field simulation with PSF."
        )
        flatcam_sim = False

    # use far field simulator to get correct object plane sizing
    psf = mask.psf
    if config.use_torch:
        psf = psf.transpose(2, 0, 1)
        psf = torch.from_numpy(psf).float()

    simulator = FarFieldSimulator(
        psf=psf,
        object_height=object_height,
        scene2mask=scene2mask,
        mask2sensor=mask2sensor,
        sensor=sensor,
        snr_db=snr_db,
        max_val=max_val,
        is_torch=config.use_torch,
    )
    image_plane, object_plane = simulator.propagate(image, return_object_plane=True)

    # channels as last dimension
    if config.use_torch:
        image_plane = image_plane.permute(1, 2, 0)
        object_plane = object_plane.permute(1, 2, 0)
        image = image.permute(1, 2, 0)

    if image_format == "grayscale":
        image_plane = rgb2gray(image_plane)
        object_plane = rgb2gray(object_plane)
    elif "bayer" in image_format:
        image_plane = rgb2bayer(image_plane, pattern=image_format[-4:])
        object_plane = rgb2bayer(object_plane, pattern=image_format[-4:])
    else:
        # make sure image is RGB
        assert image_plane.shape[-1] == 3, "Image plane must be RGB"
        assert object_plane.shape[-1] == 3, "Object plane must be RGB"

    if flatcam_sim:
        image_plane = mask.simulate(object_plane, snr_db=snr_db)

    # 3) reconstruct image
    save = config["save"]
    if save:
        save = os.getcwd()

    if config.recon.algo.lower() == "tikhonov":
        recon = CodedApertureReconstruction(
            mask, object_plane.shape, lmbd=config.recon.tikhonov.reg
        )
        recovered = recon.apply(image_plane)

    elif config.recon.algo.lower() == "admm":

        if bayer:
            raise ValueError("ADMM reconstruction not supported for Bayer images.")

        # prep PSF
        if image_format == "grayscale":
            psf = rgb2gray(mask.psf)
        else:
            psf = mask.psf
        psf = psf[np.newaxis, :, :, :] / np.linalg.norm(mask.psf.ravel())

        # prep recon
        recon = ADMM(psf, **config.recon.admm)

        # add depth channel
        recon.set_data(image_plane[None, :, :, :])
        res = recon.apply(
            n_iter=config.recon.admm.n_iter, disp_iter=config.recon.admm.disp_iter, save=save
        )[0]

        # remove depth channel
        recovered = res[0]
    else:
        raise ValueError(f"Reconstruction algorithm {config.recon.algo} not recognized.")

    # back to numpy for evaluation and plotting
    if config.use_torch:
        recovered = recovered.numpy()
        object_plane = object_plane.numpy()
        image_plane = image_plane.numpy()

    # 4) evaluate
    if image_format == "grayscale":
        object_plane = object_plane[:, :, 0]
        recovered = recovered[:, :, 0]

    print("\nEvaluation:")
    print("MSE", mse(object_plane, recovered))
    print("PSNR", psnr(object_plane, recovered))
    if image_format == "grayscale":
        print("SSIM", ssim(object_plane, recovered, channel_axis=None))
    else:
        print("SSIM", ssim(object_plane, recovered))
    if image_format == "rgb":
        print("LPIPS", lpips(object_plane, recovered))

    # -- plot
    if bayer:
        print("Converting to RGB for plotting and saving...")
        image_plane = bayer2rgb(image_plane, pattern=image_format[-4:])
        object_plane = bayer2rgb(object_plane, pattern=image_format[-4:])
        recovered = bayer2rgb(recovered, pattern=image_format[-4:])

    _, ax = plt.subplots(ncols=5, nrows=1, figsize=(24, 5))
    plot_image(object_plane, ax=ax[0])
    ax[0].set_title("Object plane")
    if np.iscomplexobj(mask.mask):
        # plot phase
        plot_image(np.angle(mask.mask), ax=ax[1])
        ax[1].set_title("Phase mask")
    else:
        plot_image(mask.mask, ax=ax[1])
        ax[1].set_title("Amplitude mask")
    plot_image(mask.psf, ax=ax[2], gamma=2.2)
    ax[2].set_title("PSF")
    plot_image(image_plane, ax=ax[3])
    ax[3].set_title("Raw data")
    plot_image(recovered, ax=ax[4])
    ax[4].set_title("Reconstruction")

    for a in ax:
        a.set_axis_off()

    plt.tight_layout()
    plt.savefig("result.png")

    if config.save:
        save_image(recovered, "reconstruction.png")

    plt.show()


if __name__ == "__main__":
    simulate()
