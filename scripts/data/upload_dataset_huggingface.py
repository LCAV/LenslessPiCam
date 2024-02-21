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
from datasets import Dataset, DatasetDict, Image
from huggingface_hub import upload_file
from lensless.utils.dataset import natural_sort


@hydra.main(
    version_base=None, config_path="../../configs", config_name="upload_dataset_huggingface"
)
def upload_dataset(config):

    start_time = time.time()

    # parameters
    repo_id = config.repo_id
    hf_token = config.hf_token
    n_files = config.n_files
    assert hf_token is not None, "Please provide a HuggingFace token."

    # get lensless files
    files_lensless = glob.glob(os.path.join(config.lensless.dir, "*" + config.lensless.ext))
    files_lensless = natural_sort(files_lensless)
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

    # check for attribute
    df_attr = None
    if config.lensed.celeba_attr is not None:
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
    n_test = int(test_size * len(common_files))
    if df_attr is not None:
        # split dict into train-test
        df_attr_test = {k: v[:n_test] for k, v in df_attr.items()}
        df_attr_train = {k: v[n_test:] for k, v in df_attr.items()}
    else:
        df_attr_test = None
        df_attr_train = None
    test_dataset = create_dataset(lensless_files[:n_test], lensed_files[:n_test], df_attr_test)
    train_dataset = create_dataset(lensless_files[n_test:], lensed_files[n_test:], df_attr_train)

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
    dataset_dict.push_to_hub(repo_id, token=hf_token)

    upload_file(
        path_or_fileobj=lensless_files[0],
        path_in_repo="lensless_example.png",
        repo_id=repo_id,
        repo_type="dataset",
        token=hf_token,
    )
    upload_file(
        path_or_fileobj=lensed_files[0],
        path_in_repo="lensed_example.png",
        repo_id=repo_id,
        repo_type="dataset",
        token=hf_token,
    )

    # total time in minutes
    print(f"Total time: {(time.time() - start_time) / 60} minutes")


if __name__ == "__main__":
    upload_dataset()
