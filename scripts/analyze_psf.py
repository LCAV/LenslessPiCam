"""
Load and plot measured PSF along with other useful analysis: pixel value
histogram and width.

Example usage
```bash
python scripts/analyze_psf.py --fp data/psf/lens_cardboard_pinhole.png
python scripts/analyze_psf.py --fp data/psf/lens_iris.png
```

For bayer data
```bash
python scripts/analyze_psf.py --fp data/psf/slm_bayer.png --bayer \
--bg 1.85546875 --rg 2.859375 --gamma 2.2
```
"""


import click
import matplotlib.pyplot as plt
from diffcam.util import rgb2gray
from diffcam.plot import plot_image, pixel_histogram, plot_cross_section
from diffcam.io import load_psf


@click.command()
@click.option(
    "--fp",
    type=str,
    help="File path for recorded PSF.",
)
@click.option(
    "--gamma",
    default=None,
    type=float,
    help="Gamma factor for plotting.",
)
@click.option(
    "--width",
    default=10,
    type=float,
    help="dB drop for estimating width",
)
@click.option(
    "--bayer",
    is_flag=True,
    help="Whether image is raw bayer data.",
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
def analyze_psf(fp, gamma, width, bayer, bg, rg):
    assert fp is not None, "Must pass file path."

    _, ax_rgb = plt.subplots(ncols=2, nrows=1, num="RGB", figsize=(15, 5))
    _, ax_gray = plt.subplots(ncols=3, nrows=1, num="Grayscale", figsize=(15, 5))

    # load PSF
    psf = load_psf(fp, return_float=False, verbose=True, bayer=bayer, blue_gain=bg, red_gain=rg)

    # plot RGB and grayscale
    ax = plot_image(psf, gamma=gamma, normalize=True, ax=ax_rgb[0])
    ax.set_title("RGB")

    psf_grey = rgb2gray(psf)
    ax = plot_image(psf_grey, gamma=gamma, normalize=True, ax=ax_gray[0])
    ax.set_title("PSF")

    # plot histogram, TODO as nbits as argument
    ax = pixel_histogram(psf, ax=ax_rgb[1], nbits=12)
    ax.set_title("Histogram")
    ax = pixel_histogram(psf_grey, ax=ax_gray[1], nbits=12)
    ax.set_title("Histogram")

    # determine PSF width
    plot_cross_section(psf_grey, color="gray", plot_db_drop=width, ax=ax_gray[2])
    _, ax_cross = plt.subplots(ncols=3, nrows=1, num="RGB widths", figsize=(15, 5))
    for i, c in enumerate(["r", "g", "b"]):
        ax, _ = plot_cross_section(psf[:, :, i], color=c, ax=ax_cross[i], plot_db_drop=width)
        if i > 0:
            ax.set_ylabel("")

    plt.show()


if __name__ == "__main__":
    analyze_psf()
