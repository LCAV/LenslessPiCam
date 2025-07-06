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
import os
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
    "--save_auto",
    is_flag=True,
    help="Save autocorrelation instead of pop-up window.",
)
@click.option(
    "--nbits",
    default=None,
    type=int,
    help="Number of bits for output. Only used for Bayer data",
)
@click.option(
    "--down",
    default=1,
    type=int,
    help="Factor by which to downsample.",
)
@click.option(
    "--back",
    type=str,
    help="File path for background image, e.g. for screen.",
)
@click.option(
    "--gain_search",
    type=int,
    default=0,
    help="Number of auto-gain iterations to run (0 = disable).",
)

def analyze_image(
    fp, gamma, width, bayer, lens, lensless, bg, rg, plot_width, save, save_auto, nbits, down, back, gain_search
):
    assert fp is not None, "Must pass file path."

    # initialize plotting axis
    fig_rgb, ax_rgb = plt.subplots(ncols=2, nrows=1, num="RGB", figsize=(15, 5))
    if lens:
        fig_gray, ax_gray = plt.subplots(ncols=3, nrows=1, num="Grayscale", figsize=(15, 5))
    else:
        fig_gray, ax_gray = plt.subplots(ncols=2, nrows=1, num="Grayscale", figsize=(15, 5))

    
    if bayer and gain_search > 0 :
        return auto_gain_locally(
            fp, gamma, width, bayer, lens, lensless,
            plot_width, save, save_auto, nbits, down, back,
            rg, bg, gain_search
        )
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
            downsample=down,
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
            downsample=down,
        )
    if nbits is None:
        nbits = int(np.ceil(np.log2(img.max())))
    # plot RGB and grayscale
    ax = plot_image(img, gamma=gamma, normalize=True, ax=ax_rgb[0])
    ax.set_title("RGB")
    ax = pixel_histogram(img, ax=ax_rgb[1], nbits=nbits)
    ax.set_title("Histogram")
    fig_rgb.savefig(os.path.join(os.path.dirname(fp), "rgb_analysis.png"))

    img_grey = rgb2gray(img[None, ...])
    ax = plot_image(img_grey, gamma=gamma, normalize=True, ax=ax_gray[0])
    ax.set_title("Grayscale")
    ax = pixel_histogram(img_grey, ax=ax_gray[1], nbits=nbits)
    ax.set_title("Histogram")
    fig_gray.savefig(os.path.join(os.path.dirname(fp), "grey_analysis.png"))

    img_grey = img_grey.squeeze()
    img = img.squeeze()

    if lens:
        # determine PSF width
        plot_cross_section(
            img_grey, color="gray", plot_db_drop=width, ax=ax_gray[2], plot_width=plot_width
        )
        fig_auto, ax_cross = plt.subplots(ncols=3, nrows=1, num="RGB widths", figsize=(15, 5))
        for i, c in enumerate(["r", "g", "b"]):
            print(f"-- {c} channel")
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
        fig_auto, ax_auto = plt.subplots(ncols=4, nrows=2, num="Autocorrelations", figsize=(15, 5))
        _, autocorr_grey = plot_autocorr2d(img_grey, ax=ax_auto[0][0])
        print("-- grayscale")
        plot_cross_section(
            autocorr_grey, color="gray", plot_db_drop=width, ax=ax_auto[1][0], plot_width=plot_width
        )
        # -- rgb
        for i, c in enumerate(["r", "g", "b"]):
            _, autocorr_c = plot_autocorr2d(img[:, :, i], ax=ax_auto[0][i + 1])
            print(f"-- {c} channel")
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
        save_8bit = save.replace(".png", "_8bit.png")
        save_image(img, save_8bit, normalize=True)
        print(f"\n8bit version saved to: {save_8bit}")

    if save_auto:
        auto_fp = os.path.join(os.path.dirname(fp), "autocorrelation.png")
        fig_auto.savefig(auto_fp)
        print(f"\nAutocorrelation saved to: {auto_fp}")
    else:
        plt.show()

def fit_gains_by_quantiles(img, nbits=12):
    """
    Fit RG/BG so that R and B match G over a dense set of mid-tail percentiles,
    with a triangular weighting that emphasizes the shoulder (around 80%).
    """
    # 10 percentiles from 0.65 to 0.95
    ps = np.linspace(0.65, 0.95, 10)
    perc = (100 * ps).tolist()

    # compute percentiles for each channel
    qs = {c: np.percentile(img[...,c], perc) for c in (0,1,2)}
    qR, qG, qB = qs[0], qs[1], qs[2]

    # build triangular weights (peak at index 5, i.e. ~80%)
    w = np.arange(1, 11)
    w = np.minimum(w, w[::-1]).astype(float)  # [1,2,3,4,5,5,4,3,2,1]

    wB = w.copy()
    wB[-2:] *= 0.5

    # least-squares with weights: gR = (w·qG·qR)/(w·qR·qR)
    numR = np.dot(w * qG, qR)
    denR = np.dot(w * qR, qR) + 1e-12
    numB = np.dot(w * qG, qB)
    denB = np.dot(w * qB, qB) + 1e-12

    gR = np.clip(numR/denR, 0.5, 2.5)
    gB = np.clip( (numB/denB) * 0.8, 0.5, 2.5 )

    # clamp to sensible bounds
    return gR, gB

def auto_gain_locally(fp, gamma, width, bayer, lens, lensless,
                      plot_width, save, save_auto, _nbits, down, back,
                      rg, bg, gain_search):
    """
    One‐shot quantile‐fitting auto‐gain:
     • Loads image once with unity gains (or supplied).
     • Computes gR,gB by fitting 3 quantiles.
     • Invokes final analysis with these gains.
    """
    img0 = load_image(
        fp, verbose=False, bayer=True,
        red_gain=rg or 1.0, blue_gain=bg or 1.0,
        nbits_out=12, return_float=False,
        downsample=down, back=back
    )

    # fit via many quantiles
    gR, gB = fit_gains_by_quantiles(img0)

    click.echo(f"[AutoGain] fitted gains → rg={gR:.3f}, bg={gB:.3f}")

    # final one-shot
    analyze_image.callback(
        fp=fp, gamma=gamma, width=width,
        bayer=bayer, lens=lens, lensless=lensless,
        bg=gB, rg=gR, plot_width=plot_width,
        save=save, save_auto=save_auto,
        nbits=12, down=down, back=back,
        gain_search=0
    )

if __name__ == "__main__":
    analyze_image()
