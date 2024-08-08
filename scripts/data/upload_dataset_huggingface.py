"""
Push dataset measured with LenslessPiCam to HuggingFace.

```bash
# install
pip install datasets
pip install huggingface_hub

# make a write token on HuggingFace

# run
python scripts/data/upload_dataset_huggingface.py \
hf_token=... \
```
"""

import hydra
import time
import os
import glob
import numpy as np
from datasets import Dataset, DatasetDict, Image
from huggingface_hub import upload_file
from lensless.utils.dataset import natural_sort
from tqdm import tqdm
from lensless.utils.io import save_image
import cv2
from joblib import Parallel, delayed


@hydra.main(
    version_base=None, config_path="../../configs", config_name="upload_dataset_huggingface"
)
def upload_dataset(config):

    start_time = time.time()

    # parameters
    repo_id = config.repo_id
    hf_token = config.hf_token
    n_files = config.n_files
    multimask = config.multimask
    n_jobs = config.n_jobs
    assert hf_token is not None, "Please provide a HuggingFace token."
    assert repo_id is not None, "Please provide a HuggingFace repo_id."
    assert config.lensless.dir is not None, "Please provide a lensless directory."
    assert config.lensed.dir is not None, "Please provide a lensed directory."
    assert (
        config.lensless.ext is not None
    ), "Please provide a lensless file extension, e.g. .png, .jpg, .tiff"
    assert (
        config.lensed.ext is not None
    ), "Please provide a lensed file extension, e.g. .png, .jpg, .tiff"

    # get masks
    files_masks = []
    n_masks = 0
    if multimask:
        files_masks = glob.glob(os.path.join(config.lensless.dir, "masks", "*.npy"))
        files_masks = natural_sort(files_masks)
        n_masks = len(files_masks)

    # get lensless files
    files_lensless = glob.glob(os.path.join(config.lensless.dir, "*" + config.lensless.ext))
    files_lensless = natural_sort(files_lensless)
    print(f"Number of lensless files: {len(files_lensless)}")
    if n_files is not None:
        print(f"Only keeping {n_files} files...")
        files_lensless = files_lensless[:n_files]

    # get lensed files
    files_lensed = glob.glob(os.path.join(config.lensed.dir, "*" + config.lensed.ext))

    # only keep if in both
    bn_lensless = [os.path.basename(f).split(".")[0] for f in files_lensless]
    bn_lensed = [os.path.basename(f).split(".")[0] for f in files_lensed]
    common_files = list(set(bn_lensless).intersection(bn_lensed))
    common_files = natural_sort(common_files)
    print(f"Number of common files: {len(common_files)}")

    # get file paths
    lensless_files = [
        os.path.join(config.lensless.dir, f + config.lensless.ext) for f in common_files
    ]
    lensed_files = [os.path.join(config.lensed.dir, f + config.lensed.ext) for f in common_files]

    if config.lensless.downsample is not None:

        tmp_dir = config.lensless.dir + "_tmp"
        os.makedirs(tmp_dir, exist_ok=True)

        def downsample(f, output_dir):
            img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(
                img,
                (0, 0),
                fx=1 / config.lensless.downsample,
                fy=1 / config.lensless.downsample,
                interpolation=cv2.INTER_LINEAR,
            )
            new_fp = os.path.join(output_dir, os.path.basename(f))
            new_fp = new_fp.split(".")[0] + config.lensless.ext
            save_image(img, new_fp, normalize=False)

        Parallel(n_jobs=n_jobs)(delayed(downsample)(f, tmp_dir) for f in tqdm(lensless_files))
        lensless_files = glob.glob(os.path.join(tmp_dir, f"*{config.lensless.ext[1:]}"))

    # convert to normalized 8 bit
    if config.lensless.eight_norm:

        tmp_dir = config.lensless.dir + "_tmp"
        os.makedirs(tmp_dir, exist_ok=True)

        # -- parallelize with joblib
        def save_8bit(f, output_dir, normalize=True):
            img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            new_fp = os.path.join(output_dir, os.path.basename(f))
            new_fp = new_fp.split(".")[0] + config.lensless.ext
            save_image(img, new_fp, normalize=normalize)

        Parallel(n_jobs=n_jobs)(delayed(save_8bit)(f, tmp_dir) for f in tqdm(lensless_files))
        lensless_files = glob.glob(os.path.join(tmp_dir, f"*{config.lensless.ext[1:]}"))

    # check for attribute
    df_attr = None
    if "celeba_attr" in config.lensed.keys():
        # load attribute txt file with pandas
        import pandas as pd

        fp = config.lensed.celeba_attr
        df = pd.read_csv(fp, sep=r"\s+", header=1, index_col=0)
        df_attr = df[: len(common_files)]
        # convert -1 to 0
        df_attr = df_attr.replace(-1, 0)
        # convert to boolean
        df_attr = df_attr.astype(bool)
        # to dict
        df_attr = df_attr.to_dict(orient="list")
    if multimask:
        # add label according to mask
        mask_labels = [np.arange(n_masks)]
        mask_labels = np.tile(mask_labels, (len(common_files), 1))
        mask_labels = mask_labels.flatten()
        mask_labels = mask_labels[: len(common_files)]
        if df_attr is not None:
            df_attr["mask_label"] = mask_labels
        else:
            df_attr = {"mask_label": mask_labels}

    # step 1: create Dataset objects
    def create_dataset(lensless_files, lensed_files, df_attr=None):
        dataset_dict = {
            "lensless": lensless_files,
            "lensed": lensed_files,
        }
        if df_attr is not None:
            # combine dictionaries
            dataset_dict = {**dataset_dict, **df_attr}
        dataset = Dataset.from_dict(dataset_dict)
        dataset = dataset.cast_column("lensless", Image())
        dataset = dataset.cast_column("lensed", Image())
        return dataset

    # train-test split
    test_size = config.test_size
    if multimask:
        n_mask_test = int(test_size * n_masks)
        # get indices from df_attr
        test_indices = np.where(df_attr["mask_label"] < n_mask_test)[0]
        train_indices = np.where(df_attr["mask_label"] >= n_mask_test)[0]
        # split dict into train-test
        test_dataset = create_dataset(
            [lensless_files[i] for i in test_indices],
            [lensed_files[i] for i in test_indices],
            {k: [v[i] for i in test_indices] for k, v in df_attr.items()},
        )
        train_dataset = create_dataset(
            [lensless_files[i] for i in train_indices],
            [lensed_files[i] for i in train_indices],
            {k: [v[i] for i in train_indices] for k, v in df_attr.items()},
        )
    elif isinstance(config.split, int):
        n_test_split = int(test_size * config.split)

        # get all indices
        n_splits = len(lensless_files) // config.split
        test_idx = np.array([])
        for i in range(n_splits):
            test_idx = np.append(test_idx, np.arange(n_test_split) + i * config.split)
        test_idx = test_idx.astype(int)

        # get train indices
        train_idx = np.setdiff1d(np.arange(len(lensless_files)), test_idx)
        train_idx = train_idx.astype(int)

        # split dict into train-test
        test_dataset = create_dataset(
            [lensless_files[i] for i in test_idx], [lensed_files[i] for i in test_idx]
        )
        train_dataset = create_dataset(
            [lensless_files[i] for i in train_idx], [lensed_files[i] for i in train_idx]
        )

    else:
        n_test = int(test_size * len(common_files))
        if df_attr is not None:
            # split dict into train-test
            df_attr_test = {k: v[:n_test] for k, v in df_attr.items()}
            df_attr_train = {k: v[n_test:] for k, v in df_attr.items()}
        else:
            df_attr_test = None
            df_attr_train = None
        test_dataset = create_dataset(lensless_files[:n_test], lensed_files[:n_test], df_attr_test)
        train_dataset = create_dataset(
            lensless_files[n_test:], lensed_files[n_test:], df_attr_train
        )
    print(f"Train size: {len(train_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    # step 2: create DatasetDict
    dataset_dict = DatasetDict(
        {
            "train": train_dataset,
            "test": test_dataset,
        }
    )

    # step 3: push to hub
    if config.files is not None:
        for f in config.files:
            fp = config.files[f]
            ext = os.path.splitext(fp)[1]
            remote_fn = f"{f}{ext}"
            upload_file(
                path_or_fileobj=fp,
                path_in_repo=remote_fn,
                repo_id=repo_id,
                repo_type="dataset",
                token=hf_token,
            )

            # viewable version of file
            img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
            local_fp = f"{f}_viewable8bit.png"
            remote_fn = f"{f}_viewable8bit.png"
            save_image(img, local_fp, normalize=True)
            upload_file(
                path_or_fileobj=local_fp,
                path_in_repo=remote_fn,
                repo_id=repo_id,
                repo_type="dataset",
                token=hf_token,
            )

    dataset_dict.push_to_hub(repo_id, token=hf_token)

    upload_file(
        path_or_fileobj=lensless_files[0],
        # path_in_repo=f"lensless_example{config.lensless.ext}" if not config.lensless.eight_norm else f"lensless_example.png",
        path_in_repo=f"lensless_example{config.lensless.ext}",
        repo_id=repo_id,
        repo_type="dataset",
        token=hf_token,
    )
    upload_file(
        path_or_fileobj=lensed_files[0],
        path_in_repo=f"lensed_example{config.lensed.ext}",
        repo_id=repo_id,
        repo_type="dataset",
        token=hf_token,
    )

    for _mask_file in files_masks:
        upload_file(
            path_or_fileobj=_mask_file,
            path_in_repo=os.path.join("masks", os.path.basename(_mask_file)),
            repo_id=repo_id,
            repo_type="dataset",
            token=hf_token,
        )

    # total time in minutes
    print(f"Total time: {(time.time() - start_time) / 60} minutes")

    # delete PNG files
    if config.lensless.eight_norm or config.lensless.downsample:
        os.system(f"rm -rf {tmp_dir}")


if __name__ == "__main__":
    upload_dataset()
