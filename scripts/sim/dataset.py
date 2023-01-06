"""

This script shows:
1) how to simulate a dataset
2) apply a reconstruction algorithm
3) compute metrics

"""

import hydra
from hydra.utils import to_absolute_path
from lensless.io import load_image, load_psf, save_image
from lensless.util import rgb2gray, resize
import numpy as np
from lensless import ADMM
from lensless.metric import mse, psnr, ssim, lpips, LPIPS_MIN_DIM
from waveprop.simulation import FarFieldSimulator
import glob
import os
from tqdm import tqdm


@hydra.main(version_base=None, config_path="../../configs", config_name="simulate_dataset")
def simulate(config):

    dataset = to_absolute_path(config.files.dataset)
    assert os.path.exists(
        dataset
    ), f"Dataset {dataset} does not exist. Download from `celeb_mini` from: https://drive.switch.ch/index.php/s/Q5OdDQMwhucIlt8"
    psf_fp = to_absolute_path(config.files.psf)
    assert os.path.exists(psf_fp), f"PSF {psf_fp} does not exist."

    if config.save.bool:
        save_dir = to_absolute_path(config.save.dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            os.makedirs(os.path.join(save_dir, "sensor_plane"))
            os.makedirs(os.path.join(save_dir, "object_plane"))
            os.makedirs(os.path.join(save_dir, "reconstruction"))

    # load psf as numpy array
    print("\nPSF:")
    psf = load_psf(psf_fp, verbose=True, downsample=config.simulation.downsample)
    if config.simulation.grayscale and len(psf.shape) == 3:
        psf = rgb2gray(psf)
    if config.simulation.downsample > 1:
        print(f"Downsampled to {psf.shape}.")

    # prepare simulator object
    simulator = FarFieldSimulator(psf=psf, **config.simulation)

    # loop over files in dataset
    print("\nSimulating dataset...")
    files = glob.glob(os.path.join(dataset, f"*.{config.files.image_ext}"))
    if config.files.n_files is not None:
        files = files[: config.files.n_files]
    for fp in tqdm(files):

        # load image as numpy array
        image = load_image(fp)
        if config.simulation.grayscale and len(image.shape) == 3:
            image = rgb2gray(image)

        # simulate image
        image_plane, object_plane = simulator.propagate(image, return_object_plane=True)
        if config.save.bool:

            bn = os.path.basename(fp).split(".")[0] + ".png"

            # can serve as ground truth
            object_plane_fp = os.path.join(save_dir, "object_plane", bn)
            save_image(object_plane, object_plane_fp)  # use max range of 255

            # lensless image
            lensless_fp = os.path.join(save_dir, "sensor_plane", bn)
            save_image(image_plane, lensless_fp, max_val=config.simulation.max_val)

    # reconstruction
    if config.admm.bool:

        print("\nReconstructing lensless measurements...")

        output_dim = image_plane.shape
        if config.simulation.output_dim is not None:
            # best would be to incorporate downsampling in the reconstruction
            # for now downsample the PSF
            print("-- Resizing PSF to", config.simulation.output_dim, "for reconstruction.")
            psf = resize(psf, shape=config.simulation.output_dim)

        # -- initialize reconstruction object
        recon = ADMM(psf, **config.admm)

        # -- metrics
        mse_vals = []
        psnr_vals = []
        ssim_vals = []
        if not config.simulation.grayscale and np.min(output_dim[:2]) >= LPIPS_MIN_DIM:
            lpips_vals = []
        else:
            lpips_vals = None

        # -- loop over files in dataset
        files = glob.glob(os.path.join(save_dir, "sensor_plane", "*.png"))
        if config.files.n_files is not None:
            files = files[: config.files.n_files]
        for fp in tqdm(files):

            lensless = load_image(fp)
            lensless = lensless / np.max(lensless)
            recon.set_data(lensless)
            res = recon.apply(n_iter=config.admm.n_iter, disp_iter=config.admm.disp_iter)
            recovered = res[0]

            if config.save.bool:
                bn = os.path.basename(fp).split(".")[0] + ".png"
                lensless_fp = os.path.join(save_dir, "reconstruction", bn)
                save_image(recovered, lensless_fp, max_val=config.simulation.max_val)

            # compute metrics
            object_plane_fp = os.path.join(save_dir, "object_plane", os.path.basename(fp))
            object_plane = load_image(object_plane_fp)

            if config.simulation.output_dim is not None:
                # best would be to incorporate downsampling in the reconstruction
                # for now downsample the PSF
                # print("-- Resizing object plane to", config.simulation.output_dim, "for metric.")
                object_plane = resize(object_plane, shape=config.simulation.output_dim)

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


if __name__ == "__main__":
    simulate()
