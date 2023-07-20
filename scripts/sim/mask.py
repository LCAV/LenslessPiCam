"""

Simulate a mask, simulate a measurement with it, and reconstruct the image.

Procedure is as follows:

1) Simulate the mask.
2) Simulate a measurement with the mask and specified physical parameters.
3) Reconstruct the image from the measurement.

Example usage:

Simulate FlatCam with separable simulation and Tikhonov reconstuction (https://arxiv.org/abs/1509.00116, Eq 7):
```
python scripts/sim/mask.py mask.type=MURA simulation.flatcam=True recon.algo=tikhonov
```

Simulate FlatCam with PSF simulation and Tikhonov reconstuction:
 (TODO doesn't work)
```
python scripts/sim/mask.py mask.type=MURA simulation.flatcam=False recon.algo=tikhonov
```

Simulate a Fresnel Zone Aperture mask with PSF simulation and ADMM reconstuction (https://www.nature.com/articles/s41377-020-0289-9):
 (TODO doesn't work)
```
python scripts/sim/mask.py mask.type=FZA recon.algo=admm

```

Simulate a PhaseContour mask with PSF simulation and ADMM reconstuction (https://www.nature.com/articles/s41377-020-0289-9):
```
python scripts/sim/mask.py mask.type=PhaseContour recon.algo=admm
```

"""

import hydra
import warnings
from hydra.utils import to_absolute_path
from lensless.utils.io import load_image, save_image
from lensless.utils.image import rgb2gray, rgb_to_bayer4d, bayer4d_to_rgb
import numpy as np
import matplotlib.pyplot as plt
from lensless import ADMM
from lensless.utils.plot import plot_image
from lensless.eval.metric import mse, psnr, ssim, lpips
from waveprop.simulation import FarFieldSimulator
import os
from numpy.linalg import multi_dot
from scipy.linalg import circulant
from waveprop.noise import add_shot_noise
from lensless.hardware.mask import CodedAperture, PhaseContour, FresnelZoneAperture
from lensless.recon.tikhonov import CodedApertureReconstruction


def fc_simulation(img, mask, format="RGB", SNR=40):
    """
    Simulation function
    """
    format = format.lower()
    assert format in [
        "grayscale",
        "rgb",
        "bayer_rggb",
        "bayer_bggr",
        "bayer_grbg",
        "bayer_gbrg",
    ], "color_profile must be in ['grayscale', 'rgb', 'bayer_rggb', 'bayer_bggr', 'bayer_grbg', 'bayer_gbrg']"
    # print(f"######### SQUEEZE SHAPE:", img.squeeze().shape, "#########")
    if len(img.squeeze().shape) == 2:
        # print('here 1')
        n_channels = 1
        img_ = img.copy()
    elif format == "grayscale":
        # print('here 2')
        n_channels = 1
        img_ = rgb2gray(img)
    elif format == "rgb":
        n_channels = 3
        img_ = img.copy()
    else:
        n_channels = 4
        img_ = rgb_to_bayer4d(img, pattern=format[-4:])
    P = circulant(np.resize(mask.col, mask.mask.shape[0]))[:, : img.shape[0]]
    Q = circulant(np.resize(mask.row, mask.mask.shape[1]))[:, : img.shape[1]]
    Y = np.dstack([multi_dot([P, img_[:, :, c], Q.T]) for c in range(n_channels)])
    Y = (Y - Y.min()) / (Y.max() - Y.min())
    Y = add_shot_noise(Y, snr_db=SNR)
    Y = (Y - Y.min()) / (Y.max() - Y.min())

    return Y, P, Q


@hydra.main(
    version_base=None, config_path="../../configs", config_name="mask_sim"
)
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
    grayscale = config.simulation.grayscale
    max_val = config.simulation.max_val

    if grayscale:
        image_format = "grayscale"
    else:
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
            sensor_name=sensor, downsample=downsample, distance_sensor=mask2sensor, **config.mask,
        )
    elif mask_type == "PhaseContour":
        mask = PhaseContour.from_sensor(
            sensor_name=sensor, downsample=downsample, distance_sensor=mask2sensor, **config.mask,
        )

    # 2) simulate measurement
    image = load_image(fp, verbose=True)
    if grayscale and len(image.shape) == 3:
        image = rgb2gray(image)

    flatcam_sim = config.simulation.flatcam
    if flatcam_sim and mask_type.upper() not in ["MURA", "MLS"]:
        warnings.warn("Flatcam simulation only supported for MURA and MLS masks. Using far field simulation with PSF.")
        flatcam_sim = False

    # use far field simulator to get correct object plane sizing
    simulator = FarFieldSimulator(
        psf=mask.psf.squeeze(),  # only support one depth plane
        object_height=object_height,
        scene2mask=scene2mask,
        mask2sensor=mask2sensor,
        sensor=sensor,
        snr_db=snr_db,
        max_val=max_val,
    )
    image_plane, object_plane = simulator.propagate(image, return_object_plane=True)

    if flatcam_sim:
        # apply flatcam simulation to object plane
        image_plane, P, Q = fc_simulation(object_plane, mask, format=image_format, SNR=snr_db)

    # 3) reconstruct image
    if config.recon.algo.lower() == "tikhonov":
        _, P, Q = fc_simulation(object_plane, mask, format=image_format, SNR=snr_db)
        recon = CodedApertureReconstruction(mask, object_plane.shape, P=P, Q=Q, lmbd=config.recon.tikhonov.reg)
        res = recon.apply(image_plane, color_profile=image_format)
        if bayer:
            recovered = bayer4d_to_rgb(res)
        else:
            recovered = res
    elif config.recon.algo.lower() == "admm":
        psf = mask.psf[np.newaxis, :, :, :] / np.linalg.norm(mask.psf.ravel())
        recon = ADMM(psf, **config.recon.admm)
        if grayscale:
            recon.set_data(image_plane[None, :, :, None])
        else:
            recon.set_data(image_plane[None, :, :, :])
        res = recon.apply(n_iter=config.recon.admm.n_iter, disp_iter=config.recon.admm.disp_iter)
        recovered = res[0]
    else:
        raise ValueError(f"Reconstruction algorithm {config.recon.algo} not recognized.")

    # 4) evaluate
    object_plane = object_plane.astype(np.float32)
    recovered = recovered.astype(np.float32).squeeze()

    print("\nEvaluation:")
    print("MSE", mse(object_plane, recovered))
    print("PSNR", psnr(object_plane, recovered))
    if grayscale:
        try:
            print("SSIM", ssim(object_plane, recovered, channel_axis=None))
        except Exception:
            print("SSIM error")
    else:
        try:
            print("SSIM", ssim(object_plane, recovered))
        except Exception:
            print("SSIM error")
        try:
            print("LPIPS", lpips(object_plane, recovered))
        except Exception:
            print("LPIPS error")

    # -- plot
    _, ax = plt.subplots(ncols=4, nrows=1, figsize=(15, 5))
    plot_image(object_plane, ax=ax[0])
    ax[0].set_title("Object plane")
    plot_image(mask.psf, ax=ax[1], gamma=2.2)
    ax[1].set_title("PSF")
    plot_image(image_plane, ax=ax[2])
    ax[2].set_title("Raw data")
    plot_image(recovered, ax=ax[3])
    ax[3].set_title("Reconstruction")
    if config.save:
        save_image(recovered, "reconstruction.png")

    plt.show()


    # fc_regul = config.simulation.regul

    # n_bits = config.simulation.n_bits
    # if mask_type.upper() not in ["MURA", "MLS"]:
    #     flatcam_simulation = False
    # else:
    #     flatcam_simulation = config.simulation.flatcam_simulation

    # # load image as numpy array
    # print("\nImage:")
    # image = load_image(fp, verbose=True)
    # # print(f"######### IMAGE SHAPE:", image.shape, "#########")
    # if grayscale and len(image.shape) == 3:
    #     image = rgb2gray(image)

    # # load psf as numpy array
    # print("\nPSF:")
    # if config.simulation.mask_type.upper() in ["MURA", "MLS"]:
    #     mask = CodedAperture.from_sensor(
    #         sensor_name=sensor,
    #         downsample=downsample,
    #         method=mask_type,
    #         n_bits=n_bits,
    #         distance_sensor=mask2sensor,
    #     )
    #     psf = mask.psf[np.newaxis, :, :, :] / np.linalg.norm(mask.psf.ravel())
    # elif mask_type.upper() == "FZA":
    #     mask = FresnelZoneAperture.from_sensor(
    #         sensor_name=sensor, downsample=downsample, distance_sensor=mask2sensor
    #     )
    #     psf = mask.psf[np.newaxis, :, :, :] / np.linalg.norm(mask.psf.ravel())
    # elif mask_type.lower() == "phase":
    #     mask = PhaseContour.from_sensor(
    #         sensor_name=sensor, downsample=downsample, distance_sensor=mask2sensor
    #     )
    #     psf = mask.psf[np.newaxis, :, :, :] / np.linalg.norm(mask.psf.ravel())
    # else:
    #     psf_fp = to_absolute_path(config.files.psf)
    #     assert os.path.exists(psf_fp), f"PSF {psf_fp} does not exist."
    #     psf = load_psf(psf_fp, verbose=True, downsample=downsample, bg_pix=None)
    # print(f"######### {mask_type.upper()} SHAPE:", psf.shape, "#########")

    # plt.figure(figsize=(10, 10))
    # if flatcam_simulation:
    #     plt.imshow(mask.mask, cmap="gray")
    # else:
    #     plt.imshow(psf.squeeze() * 255, cmap="gray")
    # plt.colorbar()

    # if grayscale and psf.shape[-1] == 3:
    #     psf = rgb2gray(psf)
    # if downsample > 1:
    #     print(f"Downsampled to {psf.shape}.")

    # # check PSF is not Nan
    # assert not np.isnan(psf).any()

    # """ Simulation"""
    # simulator = FarFieldSimulator(
    #     psf=psf.squeeze(),  # only support one depth plane
    #     object_height=object_height,
    #     scene2mask=scene2mask,
    #     mask2sensor=mask2sensor,
    #     sensor=sensor,
    #     snr_db=snr_db,
    #     max_val=max_val,
    # )

    # if flatcam_simulation:
    #     object_plane = image
    #     image_plane, P, Q = fc_simulation(image, mask, format=image_format, SNR=snr_db)
    # else:
    #     image_plane, object_plane = simulator.propagate(image, return_object_plane=True)
    # if config.save:
    #     print("######### OBJECT PLANE SHAPE:", object_plane.shape, "#########")
    #     save_image(object_plane, "object_plane.png")
    #     save_image(image_plane, "image_plane.png", max_val=max_val)
    # ax = plot_image(image_plane)
    # ax.set_title("Simulated lensless image")

    # """ Reconstruction """
    # if flatcam_simulation:
    #     recon = CodedApertureReconstruction(mask, image.shape, P=P, Q=Q, lmbd=fc_regul)
    #     res = recon.apply(image_plane, color_profile=image_format)
    #     if bayer:
    #         recovered = bayer4d_to_rgb(res)
    #     else:
    #         recovered = res
    # else:
    #     recon = ADMM(psf, **config.admm)
    #     if grayscale:
    #         recon.set_data(image_plane[None, :, :, None])
    #     else:
    #         recon.set_data(image_plane[None, :, :, :])
    #     res = recon.apply(n_iter=config.admm.n_iter, disp_iter=config.admm.disp_iter)
    #     recovered = res[0]

    # """ Evaluation """
    # object_plane = object_plane.astype(np.float32)
    # recovered = recovered.astype(np.float32).squeeze()

    # _, ax = plt.subplots(ncols=2, nrows=1, figsize=(10, 5))
    # plot_image(recovered, ax=ax[0])
    # ax[0].set_title("Reconstruction")
    # plot_image(object_plane, ax=ax[1])
    # ax[1].set_title("Original")
    # if config.save:
    #     save_image(recovered, "reconstruction.png")

    # print("\nEvaluation:")
    # print("MSE", mse(object_plane, recovered))
    # print("PSNR", psnr(object_plane, recovered))
    # if grayscale:
    #     try:
    #         print("SSIM", ssim(object_plane, recovered, channel_axis=None))
    #     except Exception:
    #         print("SSIM error")
    # else:
    #     try:
    #         print("SSIM", ssim(object_plane, recovered))
    #     except Exception:
    #         print("SSIM error")
    #     try:
    #         print("LPIPS", lpips(object_plane, recovered))
    #     except Exception:
    #         print("LPIPS error")

    # plt.show()


if __name__ == "__main__":
    simulate()
