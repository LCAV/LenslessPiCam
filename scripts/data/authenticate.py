"""
Script to compute scores for the authentication experiment.
Scores are computed using the ADMM algorithm.

Default configuration is set in `configs/authen.yaml`.

To run the script:
```
python scripts/data/authenticate.py -cn YOUR CONFIG
```

We would run out of GPU memory so wrote a bash script loop the script:
```
#!/bin/sh
# loop forever
while true
do
    # run the python script
    python scripts/data/authenticate.py -CN YOUR CONFIG \
    cont=PATH_TO_INITIAL_RUN
done
```

A ROC curve can then be plotted from multiple scores with:
```
python scripts/data/authenticate_roc.py
```

"""


from lensless.utils.dataset import DigiCam
import torch
from lensless import ADMM
from lensless.utils.image import rgb2gray
import numpy as np
import time
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from tqdm import tqdm
import hydra
import json
import omegaconf
import os


@hydra.main(version_base=None, config_path="../../configs", config_name="authen")
def authen(config):
    cont = config.cont
    scores_fp = config.scores_fp
    if scores_fp is None and cont is not None:
        print(f"Continuing from {cont}")
        hydra_path = os.path.join(cont, ".hydra/config.yaml")
        config = omegaconf.OmegaConf.load(hydra_path)

    huggingface_repo = config.repo_id
    split = config.split
    n_iter = config.n_iter
    n_files = config.n_files
    grayscale = config.grayscale
    device = config.torch_device
    rotate = True
    downsample = 1

    if scores_fp is None:

        # load multimask dataset
        if split == "all":
            train_set = DigiCam(
                huggingface_repo=huggingface_repo,
                split="train",
                rotate=rotate,
                downsample=downsample,
                return_mask_label=True,
            )
            test_set = DigiCam(
                huggingface_repo=huggingface_repo,
                split="test",
                rotate=rotate,
                downsample=downsample,
                return_mask_label=True,
            )
            n_train_psf = len(train_set.psf)
            n_test_psf = len(test_set.psf)
            if n_files is not None:
                # subset train and test set
                train_set = torch.utils.data.Subset(train_set, range(n_files * n_train_psf))
                test_set = torch.utils.data.Subset(test_set, range(n_files * n_test_psf))
            all_set = torch.utils.data.ConcatDataset([test_set, train_set])

            # prepare PSFs
            if n_files is not None:
                train_psfs = train_set.dataset.psf
                test_psfs = test_set.dataset.psf
            else:
                train_psfs = train_set.psf
                test_psfs = test_set.psf
            # -- combine into one dict
            psfs = dict()
            for psf_idx in test_psfs:
                psfs[psf_idx] = test_psfs[psf_idx]
            for psf_idx in train_psfs:
                psfs[psf_idx] = train_psfs[psf_idx]
            n_psf = len(psfs)
            print(f"Number of PSFs: {n_psf}")

            # interleave test and train so go through equal number
            n_files_per_mask = int(len(all_set) / n_psf)
            file_idx = []
            test_files_offet = n_files_per_mask * n_test_psf
            for i in range(n_files_per_mask):
                file_idx += list(np.arange(n_test_psf) + i * n_test_psf)
                file_idx += list(np.arange(n_train_psf) + i * n_train_psf + test_files_offet)

        else:
            all_set = DigiCam(
                huggingface_repo=huggingface_repo,
                split=split,
                rotate=rotate,
                downsample=downsample,
                return_mask_label=True,
            )
            psfs = all_set.psf
            n_psf = len(psfs)
            if n_files is not None:
                all_set = torch.utils.data.Subset(all_set, range(n_files * n_psf))
            print(f"Number of PSFs: {n_psf}")

            file_idx = np.arange(len(all_set))

        n_files = len(all_set)
        print("Number of images to process: ", n_files)

        for i in range(n_psf):
            if grayscale:
                psfs[i] = rgb2gray(psfs[i])
            # normalize
            psfs[i] = psfs[i] / psfs[i].norm()

        fn = f"scores_{n_iter}_grayscale{grayscale}_down{downsample}_nfiles{n_files}.json"
        if cont is not None:
            fn = os.path.join(cont, fn)
            n_files_complete = 0
            # load scores
            with open(fn, "r") as f:
                scores = json.load(f)
            for psf_idx in scores:
                n_files_complete += len(scores[psf_idx])
            file_idx = file_idx[n_files_complete:]
            print(f"Completed {n_files_complete} files, {len(file_idx)} remaining")
        else:
            # initialize scores dict
            scores = dict()
            for psf in psfs:
                scores[str(psf)] = []

        # loop over dataset
        start_time = time.time()
        for i in tqdm(file_idx):

            # save progress
            # if i % n_psf == 0:
            with open(fn, "w") as f:
                json.dump(scores, f, indent=4)

            # -- from dataset
            lensless, _, mask_label = all_set[i]

            # prepare input
            if grayscale:
                lensless = rgb2gray(lensless, keepchanneldim=True)

            # normalize
            lensless = lensless - torch.min(lensless)
            lensless = lensless / torch.max(lensless)

            # reconstruct
            scores_i = []
            for psf_idx in psfs:
                recon = ADMM(psf=psfs[psf_idx].to(device))
                recon.set_data(lensless.to(device))
                recon.apply(disp_iter=None, plot=False, n_iter=n_iter)
                scores_i.append(recon.reconstruction_error().item())
                del recon
            scores[str(mask_label)].append(np.array(scores_i).tolist())
            del lensless
            torch.cuda.empty_cache()

        proc_time = time.time() - start_time
        print(f"Processing time [m]: {proc_time / 60}")

        # save scores as JSON
        with open(fn, "w") as f:
            json.dump(scores, f, indent=4)

        confusion_fn = (
            f"confusion_matrix_{n_iter}_grayscale{grayscale}_down{downsample}_nfiles{n_files}.png"
        )

    else:

        # load scores
        with open(scores_fp, "r") as f:
            scores = json.load(f)
        n_psf = len(scores)

        # extract basename
        basename = scores_fp.split("/")[-1]
        confusion_fn = f"confusion_matrix_{basename}.png"

    # compute and plot confusion matrix
    confusion_matrix = np.zeros((n_psf, n_psf))
    accuracy = np.zeros(n_psf)
    for psf_idx in scores:
        # print(psf_idx, len(scores[psf_idx]))
        confusion_matrix[int(psf_idx)] = np.mean(np.array(scores[psf_idx]), axis=0)

        # compute accuracy for each PSF
        detected_mask = np.argmin(scores[psf_idx], axis=1)
        accuracy[int(psf_idx)] = np.mean(detected_mask == int(psf_idx))
    total_accuracy = np.mean(accuracy)
    print("Total accuracy: ", total_accuracy)

    df_cm = pd.DataFrame(
        confusion_matrix, index=[i for i in range(n_psf)], columns=[i for i in range(n_psf)]
    )
    plt.figure(figsize=(10, 7))
    # set font scale
    sn.set(font_scale=config.font_scale)
    sn.heatmap(df_cm, annot=False, cbar=True)

    # save plot
    plt.savefig(confusion_fn, bbox_inches="tight")
    print(f"Confusion matrix saved as {confusion_fn}")


if __name__ == "__main__":
    authen()
