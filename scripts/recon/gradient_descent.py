"""
Apply gradient descent.

```
python scripts/recon/gradient_descent.py
```

"""

import hydra
from hydra.utils import to_absolute_path, get_original_cwd
import os
import numpy as np
import time
import pathlib as plib
import matplotlib.pyplot as plt
from lensless.io import load_data
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

    if config["gradient_descent"]["method"] == GradientDescentUpdate.VANILLA:
        recon = GradientDescent(psf)
    elif config["gradient_descent"]["method"] == GradientDescentUpdate.NESTEROV:
        recon = NesterovGradientDescent(
            psf,
            p=config["gradient_descent"]["nesterov"]["p"],
            mu=config["gradient_descent"]["nesterov"]["mu"],
        )
    elif config["gradient_descent"]["method"] == GradientDescentUpdate.FISTA:
        recon = FISTA(
            psf,
            tk=config["gradient_descent"]["fista"]["tk"],
        )
    elif config["gradient_descent"]["method"] == "vanilla_PnP":
        import torch
        from lensless.util import load_drunet, apply_CWH_denoizer

        psf, data = torch.from_numpy(psf), torch.from_numpy(data)
        drunet = load_drunet(os.path.join(get_original_cwd(), "data/drunet_color.pth"))

        def denoiser(x):
            torch.clip(x, min=0.0, max=1.0, out=x)
            return apply_CWH_denoizer(drunet, x, noise_level=1)

        recon = GradientDescent(psf.cpu(), proj=denoiser)
    elif config["gradient_descent"]["method"] == "fista_PnP":
        import torch
        from lensless.util import load_drunet, apply_CWH_denoizer

        psf, data = torch.from_numpy(psf), torch.from_numpy(data)
        drunet = load_drunet(os.path.join(get_original_cwd(), "data/drunet_color.pth"))

        def denoiser(x):
            torch.clip(x, min=0.0, max=1.0, out=x)
            x_denoized = apply_CWH_denoizer(drunet, x, noise_level=5)
            x_denoized = torch.clip(x_denoized, min=0.0, max=1.0)
            return x + 0.1(x_denoized - x)

        recon = FISTA(psf, tk=config["gradient_descent"]["fista"]["tk"], proj=denoiser)

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
