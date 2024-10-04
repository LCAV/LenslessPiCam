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


from lensless.utils.dataset import HFDataset
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
from lensless.utils.io import save_image
from lensless.recon.model_dict import download_model, load_model


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
    save_idx = config.save_idx
    downsample = config.huggingface.downsample

    # compute scores
    if scores_fp is None:

        # load multimask dataset
        if split == "all":
            train_set = HFDataset(
                huggingface_repo=huggingface_repo,
                split="train",
                return_mask_label=True,
                rotate=config.huggingface.rotate,
                display_res=config.huggingface.image_res,
                flipud=config.huggingface.flipud,
                flip_lensed=config.huggingface.flip_lensed,
                downsample=config.huggingface.downsample,
                alignment=config.huggingface.alignment,
                simulation_config=config.simulation,
            )
            test_set = HFDataset(
                huggingface_repo=huggingface_repo,
                split="test",
                return_mask_label=True,
                rotate=config.huggingface.rotate,
                display_res=config.huggingface.image_res,
                flipud=config.huggingface.flipud,
                flip_lensed=config.huggingface.flip_lensed,
                downsample=config.huggingface.downsample,
                alignment=config.huggingface.alignment,
                simulation_config=config.simulation,
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
            all_set = HFDataset(
                huggingface_repo=huggingface_repo,
                split=split,
                return_mask_label=True,
                rotate=config.huggingface.rotate,
                display_res=config.huggingface.image_res,
                flipud=config.huggingface.flipud,
                flip_lensed=config.huggingface.flip_lensed,
                downsample=config.huggingface.downsample,
                alignment=config.huggingface.alignment,
                simulation_config=config.simulation,
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

        # load model, initialize with first PSF
        algo = config.algo
        inv_output = False
        if algo == "admm":
            model = ADMM(psf=psfs[0].to(device), n_iter=n_iter)
        elif "hf" in algo:
            param = algo.split(":")
            assert (
                len(param) == 4
            ), "hf model requires following format: hf:camera:dataset:model_name"
            camera = param[1]
            dataset = param[2]
            model_name = param[3]
            algo_config = config.get(algo)
            if algo_config is not None:
                skip_pre = algo_config.get("skip_pre", False)
                skip_post = algo_config.get("skip_post", False)
            else:
                skip_pre = False
                skip_post = False

            model_path = download_model(camera=camera, dataset=dataset, model=model_name)
            model = load_model(
                model_path, psfs[0].to(device), device, skip_pre=skip_pre, skip_post=skip_post
            )
            model.eval()
            inv_output = config.inv_output
        else:
            raise ValueError(f"Unsupported algorithm : {algo}")

        # initialize scores
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
        # TODO in batches over multiple GPU
        start_time = time.time()
        for i in tqdm(file_idx):

            if i in save_idx:
                # make folder
                save_dir = str(i)
                os.makedirs(save_dir, exist_ok=True)

            # save progress
            with open(fn, "w") as f:
                json.dump(scores, f, indent=4)

            # -- from dataset
            lensless, _, mask_label = all_set[i]

            # prepare input
            if grayscale:
                lensless = rgb2gray(lensless, keepchanneldim=True)

            # reconstruct
            scores_i = []
            for psf_idx in psfs:
                model._set_psf(psfs[psf_idx].to(device))
                model.set_data(lensless.to(device))
                with torch.no_grad():
                    if inv_output:
                        _, res, _ = model.apply(
                            disp_iter=None, plot=False, output_intermediate=True
                        )
                    else:
                        res = model.apply(disp_iter=None, plot=False, output_intermediate=False)
                        res = res[0]

                scores_i.append(
                    model.reconstruction_error(
                        prediction=res / res.max(), lensless=lensless.to(device) / lensless.max()
                    ).item()
                )

                if i in save_idx:
                    res_np = res[0].cpu().numpy()
                    res_np = res_np / res_np.max()
                    fp = os.path.join(save_dir, f"{psf_idx}.png")
                    save_image(res_np, fp)

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
