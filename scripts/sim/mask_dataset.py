"""

Simulate a mask, simulate a few measurements with it, and reconstruct the images.

Procedure is as follows:

1) Simulate the mask.
2) Simulate measurements with the mask and specified physical parameters.
3) Reconstruct the images from the measurements.

Example usage:

Simulate FlatCam with separable simulation and Tikhonov reconstuction (https://arxiv.org/abs/1509.00116, Eq 7):
```
python scripts/sim/mask_dataset.py mask.type=MLS simulation.flatcam=True recon.algo=tikhonov
```

Simulate FlatCam with PSF simulation and Tikhonov reconstuction:
```
python scripts/sim/mask_dataset.py mask.type=MLS simulation.flatcam=False recon.algo=tikhonov
```

Simulate FlatCam with PSF simulation and ADMM reconstruction:
```
python scripts/sim/mask_dataset.py mask.type=MLS simulation.flatcam=False recon.algo=admm
```

Simulate Fresnel Zone Aperture camera with PSF simulation and ADMM reconstuction (https://www.nature.com/articles/s41377-020-0289-9):
```
python scripts/sim/mask_dataset.py mask.type=FZA recon.algo=admm
```

Simulate PhaseContour camera with PSF simulation and ADMM reconstuction (https://ieeexplore.ieee.org/document/9076617):
```
python scripts/sim/mask_dataset.py mask.type=PhaseContour recon.algo=admm
```

If Pytorch is installed, you can use the `torch` flag to use Pytorch for the reconstruction (ADMM only):
```
python scripts/sim/mask_dataset.py mask.type=PhaseContour recon.algo=admm
```

"""

import hydra
import warnings
from hydra.utils import to_absolute_path
from lensless.utils.io import load_image, save_image
from lensless.utils.image import rgb2gray
import numpy as np
from lensless import ADMM
from lensless.eval.metric import mse, psnr, ssim, lpips
from waveprop.simulation import FarFieldSimulator
import glob
import os
from tqdm import tqdm
from lensless.hardware.mask import CodedAperture, PhaseContour, FresnelZoneAperture
from lensless.recon.tikhonov import CodedApertureReconstruction


@hydra.main(version_base=None, config_path="../../configs/simulate", config_name="mask_sim_dataset")
def simulate(config):

    if config.torch:
        try:
            import torch
        except ImportError:
            raise ImportError("Pytorch not found. Please install pytorch to use torch mode.")

    # set seed
    np.random.seed(config.seed)

    mask2sensor = config.simulation.mask2sensor
    sensor = config.simulation.sensor
    snr_db = config.simulation.snr_db
    downsample = config.simulation.downsample

    dataset = to_absolute_path(config.files.dataset)
    if not os.path.isdir(dataset):
        print(f"No dataset found at {dataset}")
        try:
            from torchvision.datasets.utils import download_and_extract_archive, download_url
        except ImportError:
            exit()
        msg = "Do you want to download the sample CelebA dataset (764KB)?"

        # default to yes if no input is given
        valid = input("%s (Y/n) " % msg).lower() != "n"
        if valid:
            url = "https://drive.switch.ch/index.php/s/Q5OdDQMwhucIlt8/download"
            filename = "celeb_mini.zip"
            download_and_extract_archive(
                url, os.path.dirname(dataset), filename=filename, remove_finished=True
            )

    mask_type = config.mask.type

    # check for flatcam simulation
    flatcam_sim = config.simulation.flatcam
    if flatcam_sim and mask_type.upper() not in ["MURA", "MLS"]:
        warnings.warn(
            "Flatcam simulation only supported for MURA and MLS masks. Using far field simulation with PSF."
        )
        flatcam_sim = False

    if config.save:
        if flatcam_sim:
            save_dir = to_absolute_path(config.files.dataset + "_" + mask_type + "_flatcam_sim")
        else:
            save_dir = to_absolute_path(config.files.dataset + "_" + mask_type)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            os.makedirs(os.path.join(save_dir, "sensor_plane"))
            os.makedirs(os.path.join(save_dir, "object_plane"))
            os.makedirs(os.path.join(save_dir, "reconstruction"))

    # simulate mask
    mask = None
    if mask_type.upper() in ["MURA", "MLS"]:
        mask = CodedAperture.from_sensor(
            sensor_name=sensor,
            downsample=downsample,
            method=mask_type,
            distance_sensor=mask2sensor,
            **config.mask,
        )
        psf_sim = mask.psf / np.linalg.norm(mask.psf.ravel())
    elif mask_type.upper() == "FZA":
        mask = FresnelZoneAperture.from_sensor(
            sensor_name=sensor,
            downsample=downsample,
            distance_sensor=mask2sensor,
            **config.mask,
        )
        psf_sim = mask.psf / np.linalg.norm(mask.psf.ravel())
    elif mask_type == "PhaseContour":
        mask = PhaseContour.from_sensor(
            sensor_name=sensor,
            downsample=downsample,
            distance_sensor=mask2sensor,
            n_iter=config.mask.phase_mask_iter,
            **config.mask,
        )
        psf_sim = mask.psf / np.linalg.norm(mask.psf.ravel())
    assert mask is not None, f"Mask type {mask_type} not implemented."

    if config.simulation.grayscale and len(psf_sim.shape) == 3:
        psf_sim = rgb2gray(psf_sim)

    if config.simulation.downsample > 1:
        print(f"Downsampled to {psf_sim.shape}.")

    # prepare simulator object
    simulator = FarFieldSimulator(psf=psf_sim, **config.simulation)

    # loop over files in dataset
    print("\nSimulating dataset...")
    files = glob.glob(os.path.join(dataset, f"*.{config.files.image_ext}"))
    if config.files.n_files is not None:
        files = files[: config.files.n_files]

    for fp in tqdm(files):

        # load image as numpy array
        image = load_image(fp) / 255
        if config.simulation.grayscale and len(image.shape) == 3:
            image = rgb2gray(image)

        # simulate image
        image_plane, object_plane = simulator.propagate(image, return_object_plane=True)
        if flatcam_sim:
            image_plane = mask.simulate(object_plane, snr_db=snr_db)

        if config.save:

            bn = os.path.basename(fp).split(".")[0] + ".png"

            # can serve as ground truth
            object_plane_fp = os.path.join(save_dir, "object_plane", bn)
            save_image(object_plane, object_plane_fp)  # use max range of 255

            # lensless image
            lensless_fp = os.path.join(save_dir, "sensor_plane", bn)
            save_image(image_plane, lensless_fp, max_val=config.simulation.max_val)

    # reconstruction
    recon_algo = config.recon.algo.lower()

    if config.recon.algo is not None:

        print("\nReconstructing lensless measurements...")
        # -- initialize reconstruction object
        if recon_algo == "tikhonov":
            if config.torch:
                raise NotImplementedError("Tikhonov reconstruction not implemented for torch.")
            recon = CodedApertureReconstruction(
                mask, object_plane.shape, lmbd=config.recon.tikhonov.reg
            )
        elif recon_algo == "admm":
            psf = psf_sim[np.newaxis, :, :, :]
            if config.torch:
                psf = torch.from_numpy(psf).to(config.torch_device)
            recon = ADMM(psf, **config.recon.admm)

        # -- metrics
        mse_vals = []
        psnr_vals = []
        ssim_vals = []
        if not config.simulation.grayscale:
            lpips_vals = []
        else:
            lpips_vals = None

        # -- loop over files in dataset
        files = glob.glob(os.path.join(save_dir, "sensor_plane", "*.png"))
        if config.files.n_files is not None:
            files = files[: config.files.n_files]

        for fp in tqdm(files):

            lensless = load_image(fp, as_4d=True)
            lensless = lensless / np.max(lensless)
            if recon_algo == "tikhonov":
                recovered = recon.apply(lensless[0])
            elif recon_algo == "admm":
                if config.torch:
                    lensless = torch.from_numpy(lensless).to(config.torch_device)
                recon.set_data(lensless)
                res, _ = recon.apply(n_iter=config.recon.admm.n_iter)

                # get first depth
                if config.torch:
                    recovered = res[0].cpu().numpy()
                else:
                    recovered = res[0]

            if config.save:
                bn = os.path.basename(fp).split(".")[0] + ".png"
                lensless_fp = os.path.join(save_dir, "reconstruction", bn)

                save_image(recovered, lensless_fp, max_val=config.simulation.max_val)

            # compute metrics
            object_plane_fp = os.path.join(save_dir, "object_plane", os.path.basename(fp))
            object_plane = load_image(object_plane_fp)

            mse_vals.append(mse(object_plane, recovered))
            psnr_vals.append(psnr(object_plane, recovered))
            if config.simulation.grayscale:
                ssim_vals.append(ssim(object_plane, recovered, channel_axis=None))
            else:
                ssim_vals.append(ssim(object_plane, recovered))
            if lpips_vals is not None:
                lpips_vals.append(lpips(object_plane, recovered))

    print("\nMSE (avg)", np.mean(mse_vals))
    print("PSNR (avg)", np.mean(psnr_vals))
    print("SSIM (avg)", np.mean(ssim_vals))
    if lpips_vals is not None:
        print("LPIPS (avg)", np.mean(lpips_vals))

    print("Results saved to", save_dir)


if __name__ == "__main__":
    simulate()
