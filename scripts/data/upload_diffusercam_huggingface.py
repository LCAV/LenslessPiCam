"""
Push DiffuserCam dataset to HuggingFace.

```bash
# install
pip install datasets
pip install huggingface_hub
pip install joblib

# make a write token on HuggingFace

# run
python scripts/data/upload_diffusercam_huggingface.py \
hf_token=... \
```
"""

import hydra
import time
import numpy as np
import os
import glob
from lensless.utils.io import save_image
import cv2
import PIL
from datasets import Dataset, DatasetDict, Image
from huggingface_hub import upload_file
from lensless.utils.dataset import natural_sort
from tqdm import tqdm
from joblib import Parallel, delayed


@hydra.main(
    version_base=None, config_path="../../configs", config_name="upload_diffusercam_huggingface"
)
def upload_dataset(config):

    # parameters
    repo_id = config.repo_id
    dir_diffuser = config.dir_diffuser
    dir_lensed = config.dir_lensed
    psf_fp = config.psf_fp
    hf_token = config.hf_token
    file_ext = config.file_ext
    n_files = config.n_files
    n_jobs = config.n_jobs

    assert hf_token is not None, "Please provide a HuggingFace token."

    start_time = time.time()

    # load PSF, convert to RGB, save as PNG
    psf_img = np.array(PIL.Image.open(psf_fp))
    psf_img = cv2.cvtColor(psf_img, cv2.COLOR_BGR2RGB)  # convert to RGB
    psf_fp_png = psf_fp.replace(".tiff", ".png")
    save_image(psf_img, psf_fp_png)

    # get all lensless-lensed pairs
    files_diffuser = glob.glob(os.path.join(dir_diffuser, "*" + file_ext))
    files_lensed = glob.glob(os.path.join(dir_lensed, "*" + file_ext))

    # only keep if in both
    bn_diffuser = [os.path.basename(f) for f in files_diffuser]
    bn_lensed = [os.path.basename(f) for f in files_lensed]
    common_files = list(set(bn_diffuser).intersection(bn_lensed))
    common_files = natural_sort(common_files)
    print(f"Number of common files: {len(common_files)}")
    if n_files is not None:
        print(f"Only keeping {n_files} files...")
        common_files = common_files[:n_files]

    # save as PNG
    dir_diffuser_png = dir_diffuser.replace("diffuser_images", "diffuser_png")
    os.makedirs(dir_diffuser_png, exist_ok=True)
    dir_lensed_png = dir_lensed.replace("ground_truth_lensed", "lensed_png")
    os.makedirs(dir_lensed_png, exist_ok=True)
    diffuser_png_files = []
    lensed_png_files = []

    # -- parallelize with joblib
    def save_png(f, dir_diffuser, dir_diffuser_png, dir_lensed, dir_lensed_png):
        diffuser_img = np.load(os.path.join(dir_diffuser, f))
        diffuser_img = cv2.cvtColor(diffuser_img, cv2.COLOR_BGR2RGB)  # convert to RGB
        diffuser_fn = os.path.join(dir_diffuser_png, f.replace(file_ext, ".png"))
        diffuser_png_files.append(diffuser_fn)
        save_image(diffuser_img, diffuser_fn)

        lensed_img = np.load(os.path.join(dir_lensed, f))
        lensed_img = cv2.cvtColor(lensed_img, cv2.COLOR_BGR2RGB)  # convert to RGB
        lensed_fn = os.path.join(dir_lensed_png, f.replace(file_ext, ".png"))
        lensed_png_files.append(lensed_fn)
        save_image(lensed_img, lensed_fn)

    Parallel(n_jobs=n_jobs)(
        delayed(save_png)(f, dir_diffuser, dir_diffuser_png, dir_lensed, dir_lensed_png)
        for f in tqdm(common_files)
    )

    # step 1: create Dataset objects
    dataset = Dataset.from_dict(
        {
            "lensless": diffuser_png_files,
            "lensed": lensed_png_files,
        }
    )
    dataset = dataset.cast_column("lensless", Image())
    dataset = dataset.cast_column("lensed", Image())

    # step 2: create DatasetDict
    dataset_dict = DatasetDict(
        {
            "all": dataset,
        }
    )
    # step 3: push to hub
    # -- dataset
    dataset_dict.push_to_hub(
        repo_id,
        token=hf_token,
    )
    upload_file(
        path_or_fileobj=psf_fp_png,
        path_in_repo="psf.png",
        repo_id=repo_id,
        repo_type="dataset",
        token=hf_token,
    )
    upload_file(
        path_or_fileobj=diffuser_png_files[0],
        path_in_repo="lensless_example.png",
        repo_id=repo_id,
        repo_type="dataset",
        token=hf_token,
    )
    upload_file(
        path_or_fileobj=lensed_png_files[0],
        path_in_repo="lensed_example.png",
        repo_id=repo_id,
        repo_type="dataset",
        token=hf_token,
    )

    # delete PNG files
    os.system(f"rm -rf {dir_diffuser_png}")
    os.system(f"rm -rf {dir_lensed_png}")

    # total time in minutes
    print(f"Total time: {(time.time() - start_time) / 60} minutes")


if __name__ == "__main__":
    upload_dataset()
