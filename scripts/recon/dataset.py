"""
Apply ADMM reconstruction to folder.


```
python scripts/recon/dataset.py
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
from hydra.utils import to_absolute_path
import os
import time
from lensless.utils.io import load_psf, load_image, save_image
from lensless import ADMM
import torch
import glob
from tqdm import tqdm
from joblib import Parallel, delayed


@hydra.main(version_base=None, config_path="../../configs", config_name="recon_dataset")
def admm_dataset(config):

    algo = config.algo

    # get raw data file paths
    dataset = to_absolute_path(config.input.raw_data)
    if not os.path.isdir(dataset):
        print(f"No dataset found at {dataset}")
        try:
            from torchvision.datasets.utils import download_and_extract_archive
        except ImportError:
            exit()
        msg = "Do you want to download the sample CelebA dataset measured with a random Adafruit LCD pattern (1.2 GB)?"

        # default to yes if no input is given
        valid = input("%s (Y/n) " % msg).lower() != "n"
        if valid:
            url = "https://drive.switch.ch/index.php/s/m89D1tFEfktQueS/download"
            filename = "celeba_adafruit_random_2mm_20230720_1K.zip"
            download_and_extract_archive(
                url, os.path.dirname(dataset), filename=filename, remove_finished=True
            )
    data_fps = sorted(glob.glob(os.path.join(dataset, "*.png")))
    if config.n_files is not None:
        data_fps = data_fps[: config.n_files]
    n_files = len(data_fps)

    # load PSF
    psf_fp = to_absolute_path(config.input.psf)
    flip = config.preprocess.flip
    dtype = config.input.dtype
    print("\nPSF:")
    psf, bg = load_psf(
        psf_fp,
        verbose=True,
        downsample=config.preprocess.downsample,
        return_bg=True,
        flip=flip,
        dtype=dtype,
    )
    print(f"Downsampled PSF shape: {psf.shape}")

    data_dim = None
    if config.preprocess.data_dim is not None:
        data_dim = tuple(config.preprocess.data_dim) + (psf.shape[-1],)
    else:
        data_dim = psf.shape

    # -- create output folder
    output_folder = to_absolute_path(config.output_folder)
    if algo == "apgd":
        output_folder = output_folder + f"_apgd{config.apgd.max_iter}"
    elif algo == "admm":
        output_folder = output_folder + f"_admm{config.admm.n_iter}"
    else:
        output_folder = output_folder + "_raw"
    output_folder = output_folder + f"_{data_dim[-3]}x{data_dim[-2]}"
    os.makedirs(output_folder, exist_ok=True)

    # -- apply reconstruction
    if algo == "apgd":

        from lensless.recon.apgd import APGD

        start_time = time.time()

        def recover(i):

            # reconstruction object
            recon = APGD(psf=psf, **config.apgd)

            data_fp = data_fps[i]

            # load data
            data = load_image(
                data_fp, flip=flip, bg=bg, as_4d=True, return_float=True, shape=data_dim
            )
            data = data[0]  # first depth

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

            bn = os.path.basename(data_fp)
            output_fp = os.path.join(output_folder, bn)
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
            data_fp = data_fps[i]

            # load data
            data = load_image(
                data_fp, flip=flip, bg=bg, as_4d=True, return_float=True, shape=data_dim
            )

            if config.torch:
                data = torch.from_numpy(data).type(torch_dtype).to(torch_device)

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

            bn = os.path.basename(data_fp)
            output_fp = os.path.join(output_folder, bn)
            save_image(img, output_fp)

        print(f"Processing time : {time.time() - start_time} s")
        # time per file
        print(f"Time per file : {(time.time() - start_time) / n_files} s")
        print("Files saved to: ", output_folder)


if __name__ == "__main__":
    admm_dataset()
