"""
Apply Accelerated Proximal Gradient Descent (APDG) with a non-negativity prior
for grayscale reconstruction. Example using Pycsou:
https://matthieumeo.github.io/pycsou/html/api/algorithms/pycsou.opt.proxalgs.html?highlight=apgd#pycsou.opt.proxalgs.AcceleratedProximalGradientDescent

```
python scripts/apgd.py --psf_fp data/psf/diffcam_rgb.png \
--data_fp data/raw_data/thumbs_up_rgb.png
```

"""

import numpy as np
import time
from datetime import datetime
import click
from copy import deepcopy
import matplotlib.pyplot as plt
from diffcam.io import load_data
from diffcam.plot import plot_image
from pycsou.opt.proxalgs import APGD
from pycsou.func.loss import SquaredL2Loss
from pycsou.func.penalty import NonNegativeOrthant
from pycsou.linop.conv import Convolve2D
import os
import pathlib as plib


@click.command()
@click.option(
    "--psf_fp",
    type=str,
    default="data/psf_sample.tif",
    help="File name for recorded PSF.",
)
@click.option(
    "--data_fp",
    type=str,
    default="data/rawdata_hand_sample.tif",
    help="File name for raw measurement data.",
)
@click.option(
    "--max_iter",
    type=int,
    default=500,
    help="Maximum number of iterations.",
)
@click.option(
    "--downsample",
    type=float,
    default=4,
    help="Downsampling factor.",
)
@click.option(
    "--disp",
    default=50,
    type=int,
    help="How many iterations to wait for intermediate plot. Set to negative value for no intermediate plots.",
)
@click.option(
    "--flip",
    is_flag=True,
    help="Whether to flip image.",
)
@click.option(
    "--save",
    is_flag=True,
    help="Whether to save intermediate and final reconstructions.",
)
@click.option(
    "--bayer",
    is_flag=True,
    help="Whether image is raw bayer data.",
)
@click.option(
    "--no_plot",
    is_flag=True,
    help="Whether to not plot between iterations.",
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
def apgd(
    psf_fp,
    data_fp,
    max_iter,
    downsample,
    disp,
    flip,
    bayer,
    bg,
    rg,
    gamma,
    save,
    no_plot,
    single_psf,
):

    plot_pause = 0.2
    plot = not no_plot
    psf, data = load_data(
        psf_fp=psf_fp,
        data_fp=data_fp,
        downsample=downsample,
        bayer=bayer,
        blue_gain=bg,
        red_gain=rg,
        plot=plot,
        flip=flip,
        gamma=gamma,
        gray=True,
        single_psf=single_psf,
    )

    if save:
        save = os.path.basename(data_fp).split(".")[0]
        timestamp = datetime.now().strftime("_%d%m%Y_%Hh%M")
        save = "apgd_" + save + timestamp
        save = plib.Path(__file__).parent / save
        save.mkdir(exist_ok=False)

    start_time = time.time()
    # Convoluion operator
    H = Convolve2D(size=data.size, filter=psf, shape=data.shape, dtype=np.float32)
    H.compute_lipschitz_cst()

    # Cost function
    l22_loss = (1 / 2) * SquaredL2Loss(dim=H.shape[0], data=data.ravel())
    F = l22_loss * H
    G = NonNegativeOrthant(dim=H.shape[1])
    apgd = APGD(dim=G.shape[1], F=F, G=G, acceleration="BT")
    # BT Big O(1/k^2), CD  Small o(1/K^2), CD should be faster but BT gives better results

    # -- setup to print progress report
    apgd.old_iterand = deepcopy(apgd.init_iterand)
    apgd.update_diagnostics()
    gen = apgd.iterates(n=max_iter)
    print(f"Setup time : {time.time() - start_time} s")

    # -- apply optimization
    ax = None
    if plot or save:
        ax = plot_image(data, gamma=gamma)
    start_time = time.time()
    for i, iter in enumerate(gen):

        if (i + 1) % disp == 0:
            # -- progress report
            apgd.update_diagnostics()
            apgd.old_iterand = deepcopy(apgd.iterand)
            apgd.print_diagnostics()
            image_est = apgd.iterand["iterand"]

            if plot or save:
                plot_image(image_est.reshape(data.shape), gamma=gamma, ax=ax)
                ax.set_title("Reconstruction after iteration {}".format(apgd.iter))
                if save:
                    plt.savefig(plib.Path(save) / f"{i + 1}.png")
                if plot:
                    plt.draw()
                    plt.pause(plot_pause)

    proc_time = time.time() - start_time
    print(f"Processing time : {proc_time} seconds")

    if plot:
        image_est = apgd.iterand["iterand"]
        plot_image(image_est.reshape(data.shape), gamma=gamma, ax=ax)
        ax.set_title("Final reconstruction")
        plt.show()
    if save:
        np.save(plib.Path(save) / "final_reconstruction.npy", image_est.reshape(data.shape))
        print(f"Files saved to : {save}")


if __name__ == "__main__":
    apgd()
