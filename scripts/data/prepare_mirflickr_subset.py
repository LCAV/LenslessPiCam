"""
Prepare a subset of the DiffuserCam dataset: https://github.com/Waller-Lab/LenslessLearning
which also includes viewable/postprocessed images.

Note that you need the dataset locally, which is about 100 GB.

```bash
python scripts/data/prepare_mirflickr_subset.py \
--data ~/Documents/DiffuserCam/DiffuserCam_Mirflickr_Dataset
```

"""

import hydra
from hydra.utils import get_original_cwd
from lensless.utils.dataset import MeasuredDataset, DiffuserCamMirflickr
import tqdm

import glob
import numpy as np
from datetime import datetime
import os
import pathlib as plib
from shutil import copyfile
from PIL import Image
from lensless.recon.mirflickr import postprocess
from torch.utils.data import Subset


# @click.command()
# @click.option(
#     "--data",
#     type=str,
#     help="Path to original dataset.",
# )
# @click.option(
#     "--n_files",
#     type=int,
#     default=200,
#     help="Number of files in subset.",
# )
# @click.option(
#     "--seed",
#     type=int,
#     default=11,
# )
# @click.option(
#     "--output_dir_path",
#     type=str,
#     help="Output directory. Otherwise save subset in running directory.",
# )
# def subset_mirflickr(data, n_files, seed, output_dir_path):
@hydra.main(
    version_base=None, config_path="../../configs", config_name="diffusercam_mirflickr_subset"
)
def subset_mirflickr(config):

    original_path = os.path.join(get_original_cwd(), config.original_path)
    psf_path = os.path.join(get_original_cwd(), config.psf_path)

    # load dataset
    dataset = DiffuserCamMirflickr(
        dataset_dir=original_path,
        psf_path=psf_path,
    )

    # take subset
    seed = config.seed
    n_files = config.n_files
    if seed is not None:
        np.random.seed(seed)
        indices = np.random.choice(dataset.allowed_idx, n_files, replace=False)
    else:
        indices = dataset.allowed_idx[dataset.allowed_idx <= n_files]

    subset = Subset(dataset, indices)
    for i in tqdm.tqdm(range(len(subset))):
        lensless, lensed = subset[i]

    # h = dataset[0]

    dataset = MeasuredDataset(
        root_dir=original_path,
        psf=psf_path,
        lensless_fn="diffuser_images",
        lensed_fn="ground_truth_lensed",
        downsample=config.downsample,
    )

    # take subset
    seed = config.seed
    n_files = config.n_files
    if seed is not None:
        np.random.seed(seed)
        indices = np.random.choice(len(dataset), n_files, replace=False)
    else:
        indices = range(n_files)

    subset = Subset(dataset, indices)

    diffuser_dir = os.path.join(os.path.join(original_path, "dataset"), "diffuser_images")
    lensed_dir = os.path.join(os.path.join(original_path, "dataset"), "ground_truth_lensed")
    psf_path = os.path.join(psf_path, "psf.tiff")
    output_dir_path = None

    # create output directory
    timestamp = datetime.now().strftime("%d%m%Y_%Hh%M")
    output_dir_fn = f"DiffuserCam_Mirflickr_{n_files}_{timestamp}_seed{seed}"
    if output_dir_path is not None:
        assert os.path.isdir(output_dir_path)
        output_dir = os.path.join(output_dir_path, output_dir_fn)
    else:
        output_dir = output_dir_fn
    output_plib = plib.Path(output_dir)
    output_plib.mkdir(exist_ok=False)
    diffuser_out = os.path.join(output_dir, "diffuser")
    os.makedirs(diffuser_out)
    lensed_out = os.path.join(output_dir, "lensed")
    os.makedirs(lensed_out)
    print(f"Created output directory : {output_dir}")

    # shuffle and take first few
    diffuser_files = glob.glob(diffuser_dir + "/*.npy")
    np.random.seed(seed)
    np.random.shuffle(diffuser_files)
    subset = diffuser_files[:n_files]

    # copy over files
    copyfile(psf_path, os.path.join(output_dir, os.path.basename(psf_path)))
    for fn in subset:
        bn = os.path.basename(fn)

        # copy raw data and viewable format
        copyfile(fn, os.path.join(diffuser_out, bn))
        image_data = postprocess(np.load(fn))
        image_data = (image_data * 255).astype(np.uint8)
        viewable_file = os.path.join(diffuser_out, bn.split(".")[0] + ".tif")
        im = Image.fromarray(image_data)
        im.save(viewable_file)

        # copy lensed data and viewable format
        lensed_fp = os.path.join(lensed_dir, bn)
        copyfile(lensed_fp, os.path.join(lensed_out, bn))
        image_data = (postprocess(np.load(lensed_fp)) * 255).astype(np.uint8)
        viewable_file = os.path.join(lensed_out, bn.split(".")[0] + ".tif")
        im = Image.fromarray(image_data)
        im.save(viewable_file)


if __name__ == "__main__":
    subset_mirflickr()
