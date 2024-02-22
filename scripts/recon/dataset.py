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
python scripts/recon/dataset.py algo=null preprocess.data_dim=[48,64]
```

"""

import hydra
import os
import time
from lensless.utils.io import load_psf, save_image
from lensless import ADMM
import torch
from tqdm import tqdm
from joblib import Parallel, delayed
import numpy as np
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from lensless.utils.image import resize


def prep_data(
    data,
    psf,
    bg=None,
    flip_ud=False,
    flip_lr=False,
    use_torch=False,
    torch_dtype=None,
    torch_device=None,
):
    data = np.array(data)
    if flip_ud:
        data = np.flipud(data)
    if flip_lr:
        data = np.fliplr(data)
    data = data / data.max()
    if data.shape[:2] != psf.shape[1:3]:
        data = resize(data, shape=psf.shape)
    if bg is not None:
        data = data - bg
        data = np.clip(data, a_min=0, a_max=data.max())
    if use_torch:
        data = torch.from_numpy(data).type(torch_dtype).to(torch_device)
    return data


@hydra.main(version_base=None, config_path="../../configs", config_name="recon_dataset")
def recon_dataset(config):

    repo_id = config.repo_id
    algo = config.algo

    # load dataset
    dataset = load_dataset(repo_id, split=config.split)
    n_files = len(dataset)
    if config.n_files is not None:
        n_files = min(n_files, config.n_files)
    print(f"Reconstructing {n_files} files...")

    # load PSF
    psf_fp = hf_hub_download(repo_id=repo_id, filename=config.psf_fn, repo_type="dataset")
    dtype = config.input.dtype
    print("\nPSF:")
    psf, bg = load_psf(
        psf_fp,
        verbose=True,
        downsample=config.preprocess.downsample,
        return_bg=True,
        flip_lr=config.preprocess.flip_lr,
        flip_ud=config.preprocess.flip_ud,
        dtype=dtype,
    )
    print(f"Downsampled PSF shape: {psf.shape}")

    data_dim = None
    if config.preprocess.data_dim is not None:
        data_dim = tuple(config.preprocess.data_dim) + (psf.shape[-1],)
    else:
        data_dim = psf.shape

    # create output folder
    output_folder = config.output_folder
    if output_folder is None:
        output_folder = os.path.join(os.getcwd(), os.path.basename(repo_id))
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
    if algo == "apgd":

        from lensless.recon.apgd import APGD

        start_time = time.time()

        def recover(i):

            # reconstruction object
            recon = APGD(psf=psf, **config.apgd)

            data = dataset[i]["lensless"]
            data = prep_data(
                data,
                psf,
                bg=bg,
                flip_ud=config.preprocess.flip_ud,
                flip_lr=config.preprocess.flip_lr,
                use_torch=False,
            )

            # apply reconstruction
            recon.set_data(data)
            img = recon.apply(
                disp_iter=config.display.disp,
                gamma=config.display.gamma,
                plot=config.display.plot,
            )

            # -- extract region of interest and save
            if config.roi is not None:
                roi = config.roi
                img = img[roi[0] : roi[2], roi[1] : roi[3]]

            output_fp = os.path.join(output_folder, f"{i}.png")
            save_image(img, output_fp)

        n_jobs = config.apgd.n_jobs
        if n_jobs > 1:
            Parallel(n_jobs=n_jobs)(delayed(recover)(i) for i in range(n_files))
        else:
            for i in tqdm(range(n_files)):
                recover(i)

    else:

        if config.torch:
            torch_dtype = torch.float32
            torch_device = config.torch_device
            psf = torch.from_numpy(psf).type(torch_dtype).to(torch_device)

        # create reconstruction object
        recon = None
        if config.algo == "admm":
            recon = ADMM(psf, **config.admm)

        # loop over files and apply reconstruction
        start_time = time.time()

        for i in tqdm(range(n_files)):

            # load and prepare data
            data = dataset[i]["lensless"]

            data = prep_data(
                data,
                psf,
                bg=bg,
                flip_ud=config.preprocess.flip_ud,
                flip_lr=config.preprocess.flip_lr,
                use_torch=config.torch,
                torch_dtype=torch_dtype,
                torch_device=torch_device,
            )

            if recon is not None:

                # set data
                recon.set_data(data)

                # apply reconstruction
                res = recon.apply(
                    n_iter=config.admm.n_iter,
                    disp_iter=config.display.disp,
                    gamma=config.display.gamma,
                    plot=config.display.plot,
                )

            else:

                # copy resized raw data
                res = data

            # save reconstruction as PNG
            # -- take first depth
            if config.torch:
                img = res[0].cpu().numpy()
            else:
                img = res[0]

            # -- extract region of interest
            if config.roi is not None:
                img = img[config.roi[0] : config.roi[2], config.roi[1] : config.roi[3]]

            output_fp = os.path.join(output_folder, f"{i}.png")
            save_image(img, output_fp)

        print(f"Processing time : {time.time() - start_time} s")
        # time per file
        print(f"Time per file : {(time.time() - start_time) / n_files} s")
        print("Files saved to: ", output_folder)


if __name__ == "__main__":
    recon_dataset()
