"""
Apply Accelerated Proximal Gradient Descent (APGD) with a desired prior.

Pycsou documentation of (APGD):
https://matthieumeo.github.io/pycsou/html/api/algorithms/pycsou.opt.proxalgs.html?highlight=apgd#pycsou.opt.proxalgs.AcceleratedProximalGradientDescent

Example (default to non-negativity prior):
```
python scripts/recon/apgd_pycsou.py
```

"""

import hydra
from hydra.utils import to_absolute_path
import numpy as np
import time
import matplotlib.pyplot as plt
from lensless.utils.io import load_data
from lensless.recon.apgd import APGD
import os
import pathlib as plib


import logging

# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../configs/recon", config_name="apgd_l1")
def apgd(
    config,
):
    psf, data = load_data(
        psf_fp=to_absolute_path(config["input"]["psf"]),
        data_fp=to_absolute_path(config["input"]["data"]),
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
    )

    disp = config["display"]["disp"]
    if disp < 0:
        disp = None

    save = config["save"]
    if save:
        save = os.getcwd()

    start_time = time.time()
    recon = APGD(psf=psf, **config.apgd)
    recon.set_data(data)
    print(f"Setup time : {time.time() - start_time} s")

    start_time = time.time()
    res = recon.apply(
        n_iter=config["apgd"]["max_iter"],
        disp_iter=disp,
        save=save,
        gamma=config["display"]["gamma"],
        plot=config["display"]["plot"],
    )
    print(f"Processing time : {time.time() - start_time} s")
    final_img = res[0]

    if config["display"]["plot"]:
        plt.show()
    if save:
        np.save(plib.Path(save) / "final_reconstruction.npy", final_img)
        print(f"Files saved to : {save}")


if __name__ == "__main__":
    apgd()
