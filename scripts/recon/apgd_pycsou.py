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
from datetime import datetime
import matplotlib.pyplot as plt
from lensless.io import load_data
from lensless import APGD
import os
import pathlib as plib


@hydra.main(version_base=None, config_path="../../configs", config_name="apgd_thumbs_up")
def apgd(
    config,
):

    psf, data = load_data(
        psf_fp=to_absolute_path(config["files"]["psf"]),
        data_fp=to_absolute_path(config["files"]["data"]),
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
        save = os.path.basename(config["files"]["data"]).split(".")[0]
        timestamp = datetime.now().strftime("_%d%m%d%Y_%Hh%M")
        save = "apgd_" + save + timestamp
        save = plib.Path(__file__).parent / save
        save.mkdir(exist_ok=False)

    start_time = time.time()
    recon = APGD(
        psf=psf,
        max_iter=config["apgd"]["max_iter"],
        acceleration=config["apgd"]["acceleration"],
        diff_penalty=config["apgd"]["diff_penalty"],
        diff_lambda=config["apgd"]["diff_lambda"],
        prox_penalty=config["apgd"]["prox_penalty"],
        prox_lambda=config["apgd"]["prox_lambda"],
    )
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
