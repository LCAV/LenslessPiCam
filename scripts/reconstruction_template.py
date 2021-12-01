"""
This script will load the PSF data and raw measurement for the reconstruction
that can implement afterwards.

```bash
python scripts/reconstruction_template.py --psf_fp data/psf/diffcam_rgb.png \
--data_fp data/raw_data/thumbs_up_rgb.png
```

"""

import os
import time
import pathlib as plib
import click
import matplotlib.pyplot as plt
from datetime import datetime
from diffcam.io import load_data


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
    default=500,
    help="Number of iterations.",
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
    help="How many iterations to wait for intermediate plot/results. Set to negative value for no intermediate plots.",
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
def reconstruction(
    psf_fp,
    data_fp,
    n_iter,
    downsample,
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
    )

    if disp < 0:
        disp = None
    if save:
        save = os.path.basename(data_fp).split(".")[0]
        timestamp = datetime.now().strftime("_%d%m%d%Y_%Hh%M")
        save = "YOUR_RECONSTRUCTION_" + save + timestamp
        save = plib.Path(__file__).parent / save
        save.mkdir(exist_ok=False)

    start_time = time.time()
    # TODO : setup for your reconstruction algorithm
    print(f"setup time : {time.time() - start_time} s")

    start_time = time.time()
    # TODO : apply your reconstruction
    print(f"proc time : {time.time() - start_time} s")

    if not no_plot:
        plt.show()
    if save:
        print(f"Files saved to : {save}")


if __name__ == "__main__":
    reconstruction()
