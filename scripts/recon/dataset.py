"""
Apply ADMM reconstruction to a dataset downloaded from HuggingFace.

By default, to 25 files from DiffuserCam MirFlickr test set: https://huggingface.co/datasets/bezzam/DiffuserCam-Lensless-Mirflickr-Dataset/viewer/default/test

```
python scripts/recon/dataset.py
```

To apply to CelebA measured with DigiCam: https://huggingface.co/datasets/bezzam/DigiCam-CelebA-10K/viewer/default/test
You can run the following command:
```python
python scripts/recon/dataset.py -cn recon_celeba_digicam
```

To run APGD, use the following command:
```
# (first-time): pip install git+https://github.com/matthieumeo/pycsou.git@38e9929c29509d350a7ff12c514e2880fdc99d6e
python scripts/recon/dataset.py algo=apgd
```

To just copy resized raw data, use the following command:
```
python scripts/recon/dataset.py algo=null data_dim=[48,64]
```

"""

import hydra
import os
import time
import torch
from lensless.utils.io import save_image
from lensless import ADMM
from tqdm import tqdm
from joblib import Parallel, delayed
import numpy as np
from lensless.utils.dataset import DiffuserCamMirflickrHF, HFDataset
from lensless.eval.metric import psnr, lpips
from lensless.utils.image import resize


@hydra.main(version_base=None, config_path="../../configs", config_name="recon_dataset")
def recon_dataset(config):

    algo = config.algo
    if config.dataset == "diffusercam":
        dataset = DiffuserCamMirflickrHF(split=config.split, downsample=config.downsample)
    else:
        dataset = HFDataset(
            huggingface_repo=config.dataset,
            split=config.split,
            downsample=config.downsample,
            alignment=config.alignment,
            rotate=config.rotate,
            psf=config.psf_fn,
        )

    psf = dataset.psf.to(config.torch_device)
    data_dim = dataset.psf.shape
    n_files = len(dataset)
    if config.n_files is not None:
        n_files = min(n_files, config.n_files)
    print(f"Reconstructing {n_files} files...")

    # create output folder
    output_folder = config.output_folder
    if output_folder is None:
        output_folder = os.path.join(os.getcwd(), os.path.basename(config.dataset))
    if algo == "apgd":
        output_folder = output_folder + f"_apgd{config.apgd.max_iter}"
    elif algo == "admm":
        output_folder = output_folder + f"_admm{config.admm.n_iter}"
    else:
        output_folder = output_folder + "_raw"
    output_folder = output_folder + f"_{data_dim[-3]}x{data_dim[-2]}"
    os.makedirs(output_folder, exist_ok=True)
    print(f"Output folder: {output_folder}")

    # -- apply reconstruction
    psnr_scores = []
    lpips_scores = []
    recon = None
    if algo == "apgd":

        from lensless.recon.apgd import APGD

        psf = psf.cpu().numpy()

        start_time = time.time()

        def recover(i):

            # reconstruction object
            recon = APGD(psf=psf, **config.apgd)

            lensless, lensed = dataset[i]
            lensless = lensless.numpy()
            lensed = lensed.numpy()

            # apply reconstruction
            recon.set_data(lensless)
            res = recon.apply(
                disp_iter=config.display.disp,
                gamma=config.display.gamma,
                plot=config.display.plot,
            )

            # -- extract region of interest and save
            if config.dataset != "diffusercam":
                res, lensed = dataset.extract_roi(res[np.newaxis], lensed)
                res = res[0]

            # compute metrics
            scores = psnr(lensed[0], res), lpips(lensed[0], res)

            output_fp = os.path.join(output_folder, f"{i}.png")
            save_image(res, output_fp)
            return scores

        n_jobs = config.apgd.n_jobs
        if n_jobs > 1:
            scores = Parallel(n_jobs=n_jobs)(delayed(recover)(i) for i in range(n_files))
            psnr_scores = [s[0] for s in scores]
            lpips_scores = [s[1] for s in scores]
        else:
            for i in tqdm(range(n_files)):
                scores = recover(i)
                psnr_scores.append(scores[0])
                lpips_scores.append(scores[1])

    else:

        # create reconstruction object
        if config.algo == "admm":
            recon = ADMM(psf, **config.admm)

        # loop over files and apply reconstruction
        start_time = time.time()
        for i in tqdm(range(n_files)):

            # load and prepare data
            lensless, lensed = dataset[i]

            if recon is not None:

                # set data
                recon.set_data(lensless.to(psf.device))

                # apply reconstruction
                res = recon.apply(
                    n_iter=config.admm.n_iter,
                    disp_iter=config.display.disp,
                    gamma=config.display.gamma,
                    plot=config.display.plot,
                )

                # -- extract region of interest
                if config.dataset != "diffusercam":
                    res, lensed = dataset.extract_roi(res, lensed)
                if recon is not None:
                    # compute metrics
                    lensed_np = lensed.cpu().numpy()
                    res_np = res.cpu().numpy()
                    psnr_scores.append(psnr(lensed_np, res_np))
                    lpips_scores.append(lpips(lensed_np[0], res_np[0]))

            else:

                # copy resized raw data
                data_dim = list(config.data_dim) + [psf.shape[-1]]
                res = resize(lensless.cpu().numpy(), shape=data_dim)

            # save reconstruction as PNG
            # -- take first depth
            if isinstance(res, torch.Tensor):
                img = res[0].cpu().numpy()
            else:
                img = res[0]
            output_fp = os.path.join(output_folder, f"{i}.png")
            save_image(img, output_fp)

    if len(psnr_scores) > 0:
        # print average metrics
        print(f"Avg PSNR: {np.mean(psnr_scores)}")
        print(f"Avg LPIPS: {np.mean(lpips_scores)}")

    print(f"Processing time : {time.time() - start_time} s")
    # time per file
    print(f"Time per file : {(time.time() - start_time) / n_files} s")
    print("Files saved to: ", output_folder)


if __name__ == "__main__":
    recon_dataset()
