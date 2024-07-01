"""

Load and plot measured PSF along with other useful analysis: pixel value
histogram and width.

Analyze PSF of lensless camera, namely looking at autocorrelations:
```python
python scripts/measure/analyze_image.py --fp data/psf/tape_rgb.png --gamma 2.2 --lensless
```

Example usage (data can be downloaded here: https://drive.switch.ch/index.php/s/NdgHlcDeHVDH5ww)
```bash
python scripts/measure/analyze_image.py --fp data/psf/lens_cardboard.png --plot_width 100 --lens
python scripts/measure/analyze_image.py --fp data/psf/lens_iris.png --plot_width 100 --lens
```

For Bayer data
```bash
python scripts/measure/analyze_image.py --fp data/psf/tape_bayer.png --bayer \
--gamma 2.2 --rg 2.1 --bg 1.3
```

To plot autocorrelations of lensless camera PSF
```bash
python scripts/measure/analyze_image.py --fp data/psf/tape_bayer.png --bayer \
--gamma 2.2 --rg 2.1 --bg 1.3 --lensless
```

Save RGB data from bayer
```
python scripts/measure/analyze_image.py --fp data/psf/tape_bayer.png --bayer \
--gamma 2.2 --rg 2.1 --bg 1.3 --save data/psf/tape_rgb.png
python scripts/measure/analyze_image.py --fp data/raw_data/thumbs_up_bayer.png --bayer \
--gamma 2.2 --rg 2.1 --bg 1.3 --save data/raw_data/thumbs_up_rgb.png
```

"""


import click
import cv2
import numpy as np
import matplotlib.pyplot as plt
from lensless.utils.image import rgb2gray, gamma_correction, resize
from lensless.utils.plot import plot_image, pixel_histogram, plot_cross_section, plot_autocorr2d
from lensless.utils.io import load_image, load_psf, save_image


@click.command()
@click.option(
    "--fp",
    type=str,
    help="File path for measurement.",
)
@click.option(
    "--gamma",
    default=2.2,
    type=float,
    help="Gamma factor for plotting.",
)
@click.option(
    "--width",
    default=3,
    type=float,
    help="dB drop for estimating width",
)
@click.option(
    "--bayer",
    is_flag=True,
    help="Whether image is raw bayer data.",
)
@click.option(
    "--lens",
    is_flag=True,
    help="Whether measurement is PSF of lens, in that case plot cross-section of PSF.",
)
@click.option(
    "--lensless",
    is_flag=True,
    help="Whether measurement is PSF of a lensless camera, in that case plot cross-section of autocorrelation.",
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
    "--plot_width",
    type=int,
    help="Width for cross-section.",
)
@click.option(
    "--save",
    type=str,
    help="File name to save color correct bayer as RGB.",
)
@click.option(
    "--nbits",
    default=None,
    type=int,
    help="Number of bits for output. Only used for Bayer data",
)
@click.option(
    "--back",
    type=str,
    help="File path for background image, e.g. for screen.",
)
def analyze_image(fp, gamma, width, bayer, lens, lensless, bg, rg, plot_width, save, nbits, back):
    assert fp is not None, "Must pass file path."

    # initialize plotting axis
    fig_rgb, ax_rgb = plt.subplots(ncols=2, nrows=1, num="RGB", figsize=(15, 5))
    if lens:
        fig_gray, ax_gray = plt.subplots(ncols=3, nrows=1, num="Grayscale", figsize=(15, 5))
    else:
        fig_gray, ax_gray = plt.subplots(ncols=2, nrows=1, num="Grayscale", figsize=(15, 5))

    # load PSF/image
    if lensless:
        img = load_psf(
            fp,
            verbose=True,
            bayer=bayer,
            blue_gain=bg,
            red_gain=rg,
            nbits_out=nbits,
            return_float=False,
        )[0]
    else:
        img = load_image(
            fp,
            verbose=True,
            bayer=bayer,
            blue_gain=bg,
            red_gain=rg,
            nbits_out=nbits,
            back=back,
        )
    if nbits is None:
        nbits = int(np.ceil(np.log2(img.max())))

    # plot RGB and grayscale
    ax = plot_image(img, gamma=gamma, normalize=True, ax=ax_rgb[0])
    ax.set_title("RGB")
    ax = pixel_histogram(img, ax=ax_rgb[1], nbits=nbits)
    ax.set_title("Histogram")
    fig_rgb.savefig("rgb_analysis.png")

    img_grey = rgb2gray(img[None, ...])
    ax = plot_image(img_grey, gamma=gamma, normalize=True, ax=ax_gray[0])
    ax.set_title("Grayscale")
    ax = pixel_histogram(img_grey, ax=ax_gray[1], nbits=nbits)
    ax.set_title("Histogram")
    fig_gray.savefig("grey_analysis.png")

    img_grey = img_grey.squeeze()
    img = img.squeeze()

    if lens:
        # determine PSF width
        plot_cross_section(
            img_grey, color="gray", plot_db_drop=width, ax=ax_gray[2], plot_width=plot_width
        )
        _, ax_cross = plt.subplots(ncols=3, nrows=1, num="RGB widths", figsize=(15, 5))
        for i, c in enumerate(["r", "g", "b"]):
            ax, _ = plot_cross_section(
                img[:, :, i],
                color=c,
                ax=ax_cross[i],
                plot_db_drop=width,
                max_val=2**nbits - 1,
                plot_width=plot_width,
            )
            if i > 0:
                ax.set_ylabel("")

    elif lensless:

        # plot autocorrelations and width
        # -- grey
        _, ax_auto = plt.subplots(ncols=4, nrows=2, num="Autocorrelations", figsize=(15, 5))
        _, autocorr_grey = plot_autocorr2d(img_grey, ax=ax_auto[0][0])
        plot_cross_section(
            autocorr_grey, color="gray", plot_db_drop=width, ax=ax_auto[1][0], plot_width=plot_width
        )
        # -- rgb
        for i, c in enumerate(["r", "g", "b"]):
            _, autocorr_c = plot_autocorr2d(img[:, :, i], ax=ax_auto[0][i + 1])

            ax, _ = plot_cross_section(
                autocorr_c,
                color=c,
                ax=ax_auto[1][i + 1],
                plot_db_drop=width,
                plot_width=plot_width,
            )
            ax.set_ylabel("")

    if bayer and save is not None:
        cv2.imwrite(save, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"\nColor-corrected RGB image saved to: {save}")

        # save 8bit version for visualization
        if gamma is not None:
            img = img / img.max()
            img = gamma_correction(img, gamma=gamma)
        # -- downsample
        img = resize(img, factor=1 / 4)
        print(img.shape)
        save_8bit = save.replace(".png", "_8bit.png")
        save_image(img, save_8bit, normalize=True)
        print(f"\n8bit version saved to: {save_8bit}")

    plt.show()


if __name__ == "__main__":
    analyze_image()
