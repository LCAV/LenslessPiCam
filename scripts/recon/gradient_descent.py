"""
Apply gradient descent.

```
python scripts/recon/gradient_descent.py
```

"""

import hydra
from hydra.utils import to_absolute_path
import os
import numpy as np
import time
import pathlib as plib
from datetime import datetime
import matplotlib.pyplot as plt
from lensless.io import load_data
from lensless import (
    GradientDescentUpdate,
    GradientDescent,
    NesterovGradientDescent,
    FISTA,
)


@hydra.main(
    version_base=None, config_path="../../configs", config_name="gradient_descent_thumbs_up"
)
def gradient_descent(
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
        save = "gd_" + save + timestamp
        save = plib.Path(__file__).parent / save
        save.mkdir(exist_ok=False)

    start_time = time.time()
    if config["gradient_descent"]["method"] is GradientDescentUpdate.VANILLA:
        recon = GradientDescent(psf)
    elif config["gradient_descent"]["method"] is GradientDescentUpdate.NESTEROV:
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

    if config["display"]["plot"]:
        plt.show()
    if save:
        np.save(plib.Path(save) / "final_reconstruction.npy", res[0])
        print(f"Files saved to : {save}")


if __name__ == "__main__":
    gradient_descent()
