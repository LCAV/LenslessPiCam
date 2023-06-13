"""
Apply ADMM reconstruction.
```
python scripts/recon/admm.py
```

"""

import hydra
from hydra.utils import to_absolute_path, get_original_cwd
import os
import time
import pathlib as plib
import matplotlib.pyplot as plt
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

    if config.admm.PnP:
        import torch
        from lensless.util import apply_CWH_denoizer, load_drunet
        from lensless.admmPnP import ADMM_PnP

        device = config.torch_device
        drunet = load_drunet(os.path.join(get_original_cwd(), "data/drunet_color.pth")).to(device)

        def denoiserA(x):
            torch.clip(x, min=0.0, out=x)
            x_max = torch.amax(x, dim=(-2, -3), keepdim=True) + 1e-6
            x_denoized = apply_CWH_denoizer(drunet, x / x_max, noise_level=30, device=device)
            x_denoized = torch.clip(x_denoized, min=0.0) * x_max.to(device)
            return x_denoized

        recon = ADMM_PnP(
            psf,
            denoiserA,
            mu1=config.admm.mu1,
            mu2=config.admm.mu2,
            mu3=config.admm.mu3,
            tau=config.admm.tau,
        )

    else:
        recon = ADMM(
            psf,
            mu1=config.admm.mu1,
            mu2=config.admm.mu2,
            mu3=config.admm.mu3,
            tau=config.admm.tau,
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
