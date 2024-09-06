"""
Apply FISTA for hyperspectral data recovery.

```
python scripts/recon/hyperspectral.py
```

"""

import hydra
from hydra.utils import to_absolute_path
import os
import numpy as np
import time
import pathlib as plib
import matplotlib.pyplot as plt
from lensless.utils.io import load_data
from lensless import (
    GradientDescentUpdate,
    GradientDescent,
    NesterovGradientDescent,
    FISTA,
)


@hydra.main(version_base=None, config_path="../../configs", config_name="defaults_recon")
def gradient_descent(
    config,
):

    # load mask and PSF
    mask_fp = "/root/FORKS/LenslessPiCamNoa/data/mask.npy"
    mask = np.load(mask_fp)
    mask = np.expand_dims(mask, axis=0)
    mask = mask.astype(np.float32)

    # load PSF
    import scipy

    psf_fp = "/root/FORKS/LenslessPiCamNoa/data/psf.mat"
    mat = scipy.io.loadmat(psf_fp)
    psf = mat["psf"][:, :, 0]
    psf = psf.astype(np.float32)
    psf = psf[10:260, 35 : 320 - 35]
    psf = psf / np.linalg.norm(psf)
    psf = np.expand_dims(psf, axis=0)  # add depth
    psf = np.expand_dims(psf, axis=-1)  # add channels

    # load data
    from lensless.utils.io import load_image

    data_fp = "/root/FORKS/LenslessPiCamNoa/data/266_lensless.png"
    data = load_image(
        data_fp,
        return_float=True,
        normalize=False,
        dtype=np.float32,
    )
    # -- add depth and channels dimensions
    data = np.expand_dims(data, axis=0)
    data = np.expand_dims(data, axis=-1)

    # apply gradient descent
    from lensless import HyperSpectralFISTA

    save = config["save"]
    if save:
        save = os.getcwd()

    recon = HyperSpectralFISTA(
        psf,
        mask,
        # norm=None,
        norm="ortho",
    )
    recon.set_data(data)
    res = recon.apply(
        n_iter=500,
        disp_iter=50,
        save=save,
        gamma=1.0,
        plot=False,
    )

    if config.torch:
        img = res[0].cpu().numpy()
    else:
        img = res[0]

    if config["display"]["plot"]:
        plt.show()
    if save:
        np.save(plib.Path(save) / "final_reconstruction.npy", img)
        print(f"Files saved to : {save}")

    raise ValueError

    psf, data = load_data(
        psf_fp=to_absolute_path(config.input.psf),
        data_fp=to_absolute_path(config.input.data),
        dtype=config.input.dtype,
        downsample=config["preprocess"]["downsample"],
        bayer=config["preprocess"]["bayer"],
        blue_gain=config["preprocess"]["blue_gain"],
        red_gain=config["preprocess"]["red_gain"],
        plot=config["display"]["plot"],
        flip=config["preprocess"]["flip"],
        gamma=config["display"]["gamma"],
        gray=config["preprocess"]["gray"],
        single_psf=config["preprocess"]["single_psf"],
        shape=config["preprocess"]["shape"],
        use_torch=config.torch,
        torch_device=config.torch_device,
    )

    disp = config["display"]["disp"]
    if disp < 0:
        disp = None

    save = config["save"]
    if save:
        save = os.getcwd()

    start_time = time.time()

    if config["gradient_descent"]["method"] == GradientDescentUpdate.VANILLA:
        recon = GradientDescent(psf)
    elif config["gradient_descent"]["method"] == GradientDescentUpdate.NESTEROV:
        recon = NesterovGradientDescent(
            psf,
            p=config["gradient_descent"]["nesterov"]["p"],
            mu=config["gradient_descent"]["nesterov"]["mu"],
        )
    else:
        recon = FISTA(
            psf,
            tk=config["gradient_descent"]["fista"]["tk"],
        )

    recon.set_data(data)
    print(f"Setup time : {time.time() - start_time} s")

    start_time = time.time()
    res = recon.apply(
        n_iter=config["gradient_descent"]["n_iter"],
        disp_iter=disp,
        save=save,
        gamma=config["display"]["gamma"],
        plot=config["display"]["plot"],
    )
    print(f"Processing time : {time.time() - start_time} s")

    if config.torch:
        img = res[0].cpu().numpy()
    else:
        img = res[0]

    if config["display"]["plot"]:
        plt.show()
    if save:
        np.save(plib.Path(save) / "final_reconstruction.npy", img)
        print(f"Files saved to : {save}")


if __name__ == "__main__":
    gradient_descent()
