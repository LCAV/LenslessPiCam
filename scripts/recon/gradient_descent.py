"""
Apply gradient descent.

```
python scripts/recon/gradient_descent.py --psf_fp data/psf/tape_rgb.png -\
-data_fp data/raw_data/thumbs_up_rgb.png --n_iter 300
```

"""

import os
import numpy as np
import time
import pathlib as plib
from datetime import datetime
import click
import matplotlib.pyplot as plt
from lensless.io import load_data
from lensless import (
    GradientDescentUpdate,
    GradientDescient,
    NesterovGradientDescent,
    FISTA,
)


@click.command()
@click.option(
    "--psf_fp",
    type=str,
    help="File name for recorded PSF.",
)
@click.option(
    "--data_fp",
    type=str,
    help="File name for raw measurement data.",
)
@click.option(
    "--n_iter",
    type=int,
    default=100,
    help="Number of iterations.",
)
@click.option(
    "--downsample",
    type=float,
    default=4,
    help="Downsampling factor.",
)
@click.option(
    "--shape",
    default=None,
    nargs=2,
    type=int,
    help="Image shape (height, width) for reconstruction.",
)
@click.option(
    "--method",
    default=GradientDescentUpdate.FISTA,
    type=click.Choice(GradientDescentUpdate.all_values()),
    help="Gradient descent update method.",
)
@click.option(
    "--disp",
    default=25,
    type=int,
    help="How many iterations to wait for intermediate plot. Set to negative value for no intermediate plots.",
)
@click.option(
    "--flip",
    type=int,
    is_flag=True,
    help="Whether to flip image.",
)
@click.option(
    "--save",
    is_flag=True,
    help="Whether to save intermediate and final reconstructions.",
)
@click.option(
    "--gray",
    is_flag=True,
    help="Whether to perform construction with grayscale.",
)
@click.option(
    "--bayer",
    is_flag=True,
    help="Whether image is raw bayer data.",
)
@click.option(
    "--no_plot",
    is_flag=True,
    help="Whether to no plot.",
)
@click.option(
    "--bg",
    type=float,
    help="Blue gain.",
)
@click.option(
    "--rg",
    type=float,
    help="Red gain.",
)
@click.option(
    "--gamma",
    default=None,
    type=float,
    help="Gamma factor for plotting.",
)
@click.option(
    "--single_psf",
    is_flag=True,
    help="Same PSF for all channels (sum) or unique PSF for RGB.",
)
def gradient_descent(
    psf_fp,
    data_fp,
    n_iter,
    downsample,
    method,
    disp,
    flip,
    gray,
    bayer,
    bg,
    rg,
    gamma,
    save,
    no_plot,
    single_psf,
    shape,
):
    psf, data = load_data(
        psf_fp=psf_fp,
        data_fp=data_fp,
        downsample=downsample,
        bayer=bayer,
        blue_gain=bg,
        red_gain=rg,
        plot=not no_plot,
        flip=flip,
        gamma=gamma,
        gray=gray,
        single_psf=single_psf,
        shape=shape,
    )

    if disp < 0:
        disp = None
    if save:
        save = os.path.basename(data_fp).split(".")[0]
        timestamp = datetime.now().strftime("_%d%m%Y_%Hh%M")
        save = "gd_" + save + timestamp
        save = plib.Path(__file__).parent / save
        save.mkdir(exist_ok=False)

    start_time = time.time()
    if method is GradientDescentUpdate.VANILLA:
        recon = GradientDescient(psf)
    elif method is GradientDescentUpdate.NESTEROV:
        recon = NesterovGradientDescent(psf)
    else:
        recon = FISTA(psf)
    recon.set_data(data)
    print(f"Setup time : {time.time() - start_time} s")

    start_time = time.time()
    res = recon.apply(n_iter=n_iter, disp_iter=disp, save=save, gamma=gamma, plot=not no_plot)
    print(f"Processing time : {time.time() - start_time} s")

    if not no_plot:
        plt.show()
    if save:
        np.save(plib.Path(save) / "final_reconstruction.npy", res[0])
        print(f"Files saved to : {save}")


if __name__ == "__main__":
    gradient_descent()
