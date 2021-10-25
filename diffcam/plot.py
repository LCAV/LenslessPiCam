import numpy as np
import warnings
import matplotlib.pyplot as plt

from diffcam.util import FLOAT_DTYPES, get_max_val, gamma_correction


def plot_image(img, ax=None, gamma=None, normalize=True):
    """ """
    if ax is None:
        _, ax = plt.subplots()

    max_val = img.max()
    if not normalize:
        if img.dtype not in FLOAT_DTYPES:
            max_val = get_max_val(img)
        else:
            max_val = 1

    # need float image for gamma correction and plotting
    img_norm = img / max_val
    if gamma:
        img_norm = gamma_correction(img_norm, gamma=gamma)

    if len(img.shape) == 3:
        ax.imshow(img_norm)
    else:
        ax.imshow(img, cmap="gray")

    return ax


def pixel_histogram(img, nbits=None, ax=None, log_scale=True):
    """
    Plot pixel value histogram.

    Parameters
    ----------
    img : py:class:`~numpy.ndarray`
        2D or 3D image.
    nbits : int, optional
        Bit-depth of camera data.
    ax : :py:class:`~matplotlib.axes.Axes`, optional
            `Axes` object to fill, default is to create one.
    log_scale : bool, optional
        Whether to use log scale in counting number of pixels.

    Return
    ------
    ax : :py:class:`~matplotlib.axes.Axes`
    """
    if ax is None:
        _, ax = plt.subplots()

    # TODO change
    if nbits:
        # max_val = get_max_val(img, nbits)
        max_val = 2 ** nbits - 1
    else:
        max_val = int(img.max())

    if len(img.shape) == 3:
        # 3D image
        color_order = ("r", "g", "b")
        for i, col in enumerate(color_order):
            hist, bins = np.histogram(img[:, :, i].ravel(), bins=max_val, range=[0, max_val + 1])
            ax.plot(hist, color=col)
    else:
        # 2D image
        vals = img.flatten()
        hist, bins = np.histogram(vals, bins=max_val, range=[0, max_val + 1])
        ax.plot(hist, color="gray")
    ax.set_xlim([max_val - 1.1 * max_val, max_val * 1.1])

    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("Pixel value")
    ax.grid()

    return ax


def plot_cross_section(vals, idx=None, ax=None, dB=True, plot_db_drop=3, min_val=1e-4, **kwargs):
    """
    Plot cross-section of a 2-D image.

    Parameters
    ----------
    vals : py:class:`~numpy.ndarray`
        2-D image data.
    idx : int, optional
        Row for which to plot cross-section. Default is to take middle.
    ax : :py:class:`~matplotlib.axes.Axes`, optional
        `Axes` object to fill, default is to create one.
    dB : bool, optional
        Whether to plot in dB scale.

    Return
    ------
    ax : :py:class:`~matplotlib.axes.Axes`
    """

    if ax is None:
        _, ax = plt.subplots()

    if idx is None:
        max_idx = np.unravel_index(np.argmax(vals, axis=None), vals.shape)
        idx = max_idx[0]

    cross_section = vals[idx, :].astype(np.float32)
    cross_section /= np.max(np.abs(cross_section))
    if dB:
        cross_section[cross_section < min_val] = min_val
        cross_section = 10 * np.log10(cross_section)
        min_val = 10 * np.log10(min_val)
        ax.set_ylabel("dB")
    shift_vals = np.arange(len(cross_section))
    shift_vals -= np.argmax(cross_section)
    ax.plot(shift_vals, cross_section, **kwargs)
    ax.set_ylim([min_val, 0])
    ax.grid()

    if dB and plot_db_drop:
        zero_crossings = np.where(np.diff(np.signbit(cross_section + plot_db_drop)))[0]
        if len(zero_crossings) == 2:
            zero_crossings -= np.argmax(cross_section)
            width = zero_crossings[1] - zero_crossings[0]
            ax.set_title(f"-{plot_db_drop}dB width = {width}")
            ax.axvline(x=zero_crossings[0], c="k", linestyle="--")
            ax.axvline(x=zero_crossings[1], c="k", linestyle="--")
        else:
            warnings.warn(
                "Width could not be determined. Did not detect just two -3dB points : {}".format(
                    zero_crossings
                )
            )

    return ax, cross_section
