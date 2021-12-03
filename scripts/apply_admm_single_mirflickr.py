"""
Apply ADMM on a single image from DiffuserCam dataset: https://github.com/Waller-Lab/LenslessLearning

Or download a subset here: https://drive.switch.ch/index.php/s/vmAZzryGI8U8rcE

```
python scripts/apply_admm_single_mirflickr.py \
--data DiffuserCam_Mirflickr_200_3011302021_11h43_seed11 \
--fid 172
```

"""
import glob
import numpy as np
from diffcam.util import print_image_info
from diffcam.io import load_image, load_psf
from diffcam.plot import plot_image
import matplotlib.pyplot as plt
from datetime import datetime
import pathlib as plib
import os
import random
import click
import time
from diffcam.mirflickr import ADMM_MIRFLICKR, postprocess
from diffcam.metric import mse, psnr, ssim, lpips


@click.command()
@click.option(
    "--data",
    type=str,
    help="Path to dataset.",
)
@click.option(
    "--fid",
    type=str,
    help="ID of data to reconstruct. If not provided select at random.",
)
@click.option(
    "--gamma",
    default=None,
    type=float,
    help="Gamma factor for plotting.",
)
@click.option(
    "--single_psf",
    is_flag=True,
    help="Same PSF for all channels (sum) or unique PSF for RGB.",
)
@click.option(
    "--n_iter",
    type=int,
    default=100,
    help="Number of iterations.",
)
@click.option(
    "--disp",
    default=10,
    type=int,
    help="How many iterations to wait for intermediate plot. Set to negative value for no intermediate plots.",
)
@click.option(
    "--save",
    is_flag=True,
    help="Whether to save reconstructions.",
)
def apply_admm(data, fid, gamma, single_psf, n_iter, disp, save):
    if fid is None:
        fn = glob.glob(os.path.join(os.path.join(data, "diffuser"), "*.npy"))
        fn = [_fn.split("im")[1] for _fn in fn]
        fn = [int(_fn.split(".")[0]) for _fn in fn]
        fid = random.choice(fn)
        print(f"Reconstruction : {fid}")
    downsample = 4  # has to be this for collected data!

    # form file path
    psf_fp = os.path.join(data, "psf.tiff")
    dataset_dir = os.path.join(data, "dataset")
    if os.path.isdir(dataset_dir):
        lensless_fp = os.path.join(dataset_dir, f"diffuser_images/im{fid}.npy")
        lensed_fp = os.path.join(dataset_dir, f"ground_truth_lensed/im{fid}.npy")
    else:
        lensless_fp = os.path.join(data, f"diffuser/im{fid}.npy")
        lensed_fp = os.path.join(data, f"lensed/im{fid}.npy")

    # initialize plot
    _, ax = plt.subplots(ncols=2, nrows=2, figsize=(15, 10))

    # diffuser data
    print("Diffuser data")
    diffuser = np.load(lensless_fp)
    print_image_info(diffuser)
    _ax = plot_image(postprocess(diffuser), gamma=gamma, ax=ax[0, 0])
    _ax.set_title("Diffuser")

    # Lensed data
    print("\nLensed data")
    lensed = np.load(lensed_fp)
    print_image_info(lensed)
    lensed = postprocess(lensed)
    _ax = plot_image(lensed, gamma=gamma, ax=ax[0, 1])
    _ax.set_title("Lensed")

    # PSF
    print("\nPSF data")
    psf = load_image(fp=psf_fp, verbose=True)
    _ax = plot_image(psf, gamma=gamma, ax=ax[1, 0])
    _ax.set_title("PSF")

    # RECONSTRUCTION
    # -- prepare data
    print("\nPrepared PSF data")
    psf_float, background = load_psf(
        psf_fp,
        downsample=downsample,
        return_float=True,
        return_bg=True,
        bg_pix=(0, 15),
        single_psf=single_psf,
    )
    print_image_info(psf_float)

    print("\nPrepared diffuser data")
    diffuser_prep = diffuser - background
    diffuser_prep = np.clip(diffuser_prep, a_min=0, a_max=1)
    diffuser_prep /= np.linalg.norm(diffuser_prep.ravel())
    print_image_info(diffuser_prep)

    # apply ADMM
    if save:
        save = os.path.basename(lensless_fp).split(".")[0]
        timestamp = datetime.now().strftime("_%d%m%d%Y_%Hh%M")
        save = "admm_" + save + timestamp
        save = plib.Path(__file__).parent / save
        save.mkdir(exist_ok=False)
    recon = ADMM_MIRFLICKR(psf_float)
    recon.set_data(diffuser_prep)
    start_time = time.time()
    recon.apply(n_iter=n_iter, disp_iter=disp, save=save, gamma=gamma, ax=ax[1, 1])
    proc_time = time.time() - start_time
    print(f"Processing time : {proc_time} seconds")
    _ax.set_title("ADMM reconstruction")

    # -- compute metrics
    print("\nReconstruction")
    est = recon.get_image_est()
    print_image_info(est)

    print("\nMSE", mse(lensed, est))
    print("PSNR", psnr(lensed, est))
    print("SSIM", ssim(lensed, est))
    print("LPIPS", lpips(lensed, est))

    plt.show()


if __name__ == "__main__":
    apply_admm()
