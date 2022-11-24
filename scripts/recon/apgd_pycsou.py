"""
Apply Accelerated Proximal Gradient Descent (APGD) with a desired prior. 

Pycsou documentation of (APGD):
https://matthieumeo.github.io/pycsou/html/api/algorithms/pycsou.opt.proxalgs.html?highlight=apgd#pycsou.opt.proxalgs.AcceleratedProximalGradientDescent

Example (default to non-negativity prior):
```
python scripts/recon/apgd_pycsou.py --psf_fp data/psf/tape_rgb.png --data_fp \
data/raw_data/thumbs_up_rgb.png
```
Note that RGB reconstruction will not plot intermediate results as each channel
is solved separately.

A faster approach can be applied by implementing `RealFFTConvolve2D` such that
the real-valued FFT is used and the 2-D FFT simulateneously applied across
channels
```
python scripts/recon/apgd_pycsou.py --psf_fp data/psf/tape_rgb.png \
--data_fp data/raw_data/thumbs_up_rgb.png --real_conv
```
Note that `RealFFTConvolve2D` has to be implemented in `lensless/realfftconv.py`.

If you are an instructor and/or would like the solution, please send an email to 
eric[dot]bezzam[at]epfl[dot]ch.

"""

import numpy as np
import time
from datetime import datetime
import click
import matplotlib.pyplot as plt
from lensless.io import load_data
from lensless.plot import plot_image
from lensless import APGD, APGDPriors
import os
import pathlib as plib


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
    "--prior",
    default=APGDPriors.NONNEG,
    type=click.Choice(APGDPriors.all_values()),
    help="Prior/penalty for APGD.",
)
@click.option(
    "--max_iter",
    type=int,
    default=300,
    help="Maximum number of iterations.",
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
@click.option(
    "--real_conv",
    is_flag=True,
    help="Whether to use real convolution linear operator.",
)
def apgd(
    psf_fp,
    data_fp,
    prior,
    gray,
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
    real_conv,
    shape,
):

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
        gray=gray,
        single_psf=single_psf,
        shape=shape,
    )

    if save:
        save = os.path.basename(data_fp).split(".")[0]
        timestamp = datetime.now().strftime("_%d%m%Y_%Hh%M")
        save = "apgd_" + save + timestamp
        save = plib.Path(__file__).parent / save
        save.mkdir(exist_ok=False)

    if prior == APGDPriors.L2:
        diff_penalty = prior
        prox_penalty = None
    else:
        diff_penalty = None
        prox_penalty = prior

    start_time = time.time()

    if False and (real_conv or gray): #TODO

        # for `real_conv` parallelize RGB channels with custom operator
        recon = APGD(
            psf=psf,
            max_iter=max_iter,
            diff_penalty=diff_penalty,
            prox_penalty=prox_penalty,
            realconv=real_conv,
        )
        recon.set_data(data)
        print(f"Setup time : {time.time() - start_time} s")

        start_time = time.time()
        res = recon.apply(n_iter=max_iter, disp_iter=disp, save=save, gamma=gamma, plot=not no_plot)
        print(f"Processing time : {time.time() - start_time} s")

        final_img = res[0]

    else:
        # loop over RGB channels (naive approach with complex-valued FFT)
        recon = [
            [
                APGD(
                    psf=psf[dep, :, :, col],
                    max_iter=max_iter,
                    diff_penalty=diff_penalty,
                    prox_penalty=prox_penalty,
                    realconv=real_conv,
                )
                for col in range(psf.shape[3])
            ]
            for dep in range(psf.shape[0])
        ]

        [
            [recon[dep][col].set_data(data[dep, :, :, col]) for col in range(data.shape[3])]
            for dep in range(data.shape[0])
        ]
        print(f"Setup time : {time.time() - start_time} s")
        print(data.shape)
        start_time = time.time()
        final_img = []
        print("Looping over depths")
        for dep in range(data.shape[0]):
            print(
                f"-- depth {dep}",
            )
            print("Looping over channels...")
            for col in range(data.shape[3]):
                print(f"-- channel {col}", end="")
                final_img.append(
                    recon[dep][col].apply(
                        n_iter=max_iter, disp_iter=max_iter + 1, save=False, gamma=gamma, plot=False
                    )
                )
                print(f", {time.time() - start_time} s")

        print(f"Processing time : {time.time() - start_time} s")

        print(np.array(final_img).shape)
        final_img = np.transpose(np.array(final_img), (1, 2, 0))
        ax = plot_image(final_img, gamma=gamma)
        ax.set_title("Final reconstruction after {} iterations".format(max_iter))
        if save:
            plt.savefig(plib.Path(save) / "final_reconstruction.png")

    if not no_plot:
        plt.show()
    if save:
        np.save(plib.Path(save) / "final_reconstruction.npy", final_img)
        print(f"Files saved to : {save}")


if __name__ == "__main__":
    apgd()
