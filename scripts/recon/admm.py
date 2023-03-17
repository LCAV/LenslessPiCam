"""
Apply ADMM reconstruction.
```
python scripts/recon/admm.py
```

"""

import hydra
from hydra.utils import to_absolute_path
import os
import time
import pathlib as plib
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from lensless.io import load_data
from lensless import ADMM


@hydra.main(version_base=None, config_path="../../configs", config_name="defaults_recon")
def admm(config):

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
        torch=config.torch,
        torch_device=config.torch_device,
    )

    disp = config["display"]["disp"]
    if disp < 0:
        disp = None

    save = config["save"]
    if save:
        save = os.getcwd()

    start_time = time.time()
    recon = ADMM(
        psf,
        mu1=config["admm"]["mu1"],
        mu2=config["admm"]["mu2"],
        mu3=config["admm"]["mu3"],
        tau=config["admm"]["tau"],
    )
    recon.set_data(data)
    print(f"Setup time : {time.time() - start_time} s")

    start_time = time.time()
    res = recon.apply(
        n_iter=config["admm"]["n_iter"],
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
    admm()
