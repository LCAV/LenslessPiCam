"""

Download subset from here: https://drive.switch.ch/index.php/s/vmAZzryGI8U8rcE
Or full dataset here: https://github.com/Waller-Lab/LenslessLearning

```
python scripts/eval/mirflickr_admm.py
```

"""

import hydra
from hydra.utils import to_absolute_path
import glob
import os
import pathlib as plib
from datetime import datetime
from lensless.utils.io import load_psf
import numpy as np
from lensless.utils.image import print_image_info
from PIL import Image
from lensless.recon.mirflickr import ADMM_MIRFLICKR, postprocess
from lensless.eval.metric import mse, psnr, ssim, lpips


@hydra.main(version_base=None, config_path="../../configs", config_name="evaluate_mirflickr_admm")
def mirflickr_dataset(config):
    data = to_absolute_path(config.dataset)
    n_files = config.n_files
    single_psf = config.preprocess.single_psf
    save = config.save
    n_iter = config.admm.n_iter

    dataset_dir = os.path.join(data, "dataset")
    if os.path.isdir(dataset_dir):
        diffuser_dir = os.path.join(dataset_dir, "diffuser_images")
        lensed_dir = os.path.join(dataset_dir, "ground_truth_lensed")
    else:
        diffuser_dir = os.path.join(data, "diffuser")
        lensed_dir = os.path.join(data, "lensed")
    psf_fp = os.path.join(data, "psf.tiff")
    downsample = 4  # has to be this for collected data!

    # determine files
    files = glob.glob(diffuser_dir + "/*.npy")
    if n_files:
        files = files[:n_files]
    files = [os.path.basename(fn) for fn in files]
    print("Number of files : ", len(files))

    # -- prepare PSF
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

    if save:
        timestamp = datetime.now().strftime("_%d%m%Y_%Hh%M")
        save = "admm_mirflickr" + timestamp
        save = plib.Path(__file__).parent / save
        save.mkdir(exist_ok=False)

    # -- create ADMM object
    recon = ADMM_MIRFLICKR(psf_float, **config.admm)
    print("\nLooping through files...")
    mse_scores = []
    psnr_scores = []
    ssim_scores = []
    lpips_scores = []
    for fn in files:
        bn = os.path.basename(fn).split(".")[0]
        print(f"\n{bn}")

        # load diffuser data
        lensless_fp = os.path.join(diffuser_dir, fn)
        diffuser = np.load(lensless_fp)
        diffuser_prep = diffuser - background
        diffuser_prep = np.clip(diffuser_prep, a_min=0, a_max=1)
        diffuser_prep /= np.linalg.norm(diffuser_prep.ravel())
        recon.set_data(diffuser_prep)
        est = recon.apply(n_iter=n_iter, plot=False)

        if save:
            np.save(os.path.join(save, f"{bn}.npy"), est)
            # viewable data
            output_fn = os.path.join(save, f"{bn}.tif")
            est_norm = est / est.max()
            image_data = (est_norm * 255).astype(np.uint8)
            im = Image.fromarray(image_data)
            im.save(output_fn)

        # compute scores
        lensed_fp = os.path.join(lensed_dir, fn)
        lensed = np.load(lensed_fp)
        lensed = postprocess(lensed)
        mse_scores.append(mse(lensed, est))
        psnr_scores.append(psnr(lensed, est))
        ssim_scores.append(ssim(lensed, est))
        lpips_scores.append(lpips(lensed, est))
        print(mse_scores[-1])
        print(psnr_scores[-1])
        print(ssim_scores[-1])
        print(lpips_scores[-1])

    if save:
        print(f"\nReconstructions saved to : {save}")

    print("\nMSE (avg)", np.mean(mse_scores))
    print("PSNR (avg)", np.mean(psnr_scores))
    print("SSIM (avg)", np.mean(ssim_scores))
    print("LPIPS (avg)", np.mean(lpips_scores))


if __name__ == "__main__":
    mirflickr_dataset()
