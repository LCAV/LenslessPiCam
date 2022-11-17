import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from lensless.util import FLOAT_DTYPES, get_max_val, gamma_correction

try:
    from lensless.autocorr_sol import autocorr2d
except:
    from lensless.autocorr import autocorr2d


def plot_image(img, ax=None, gamma=None, normalize=True):
    """
    Plot image data.

    Parameters
    ----------
    img : :py:class:`~numpy.ndarray`
        Data to plot.
    ax :py:class:`~matplotlib.axes.Axes`, optional
        `Axes` object to fill for plotting/saving, default is to create one.
    gamma : float, optional
            Gamma correction factor to apply for plots. Default is None.
    normalize : bool
        Whether to normalize data to maximum range. Default is True.

    Returns
    -------
    ax :py:class:`~matplotlib.axes.Axes`, optional
        `Axes` object to fill for plotting/saving, default is to create one.

    """

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
    if gamma and gamma > 1:
        img_norm = gamma_correction(img_norm, gamma=gamma)

    # full data format : [depth, width, height, color]
    if len(img.shape) == 4:
        if img.shape[3] == 3:  # 3d rgb
            sum_img = np.sum(img_norm, axis=0)
            ax.imshow(sum_img)

        else:
            assert img.shape[3] == 1  # 3d grayscale with color channel extended
            sum_img = np.sum(img_norm[:, :, :, 0], axis=0)
            ax.imshow(sum_img, cmap="gray")

    # data of length 3 means we have to infer whethever depth or color is missing, based on shape.
    elif len(img.shape) == 3:

        if img.shape[2] == 3:  # 2D rgb
            ax.imshow(img_norm)

        elif img.shape[2] == 1:  # 2D grayscale with color channel extended
            ax.imshow(img_norm[:, :, 0], cmap="gray")

        else:  # 3D grayscale
            sum_img = np.sum(img_norm, axis=0)
            ax.imshow(sum_img, cmap="gray")

    # data of length 2 means we have only width and height
    elif len(img.shape) == 2:  # 2D grayscale
        ax.imshow(img_norm, cmap="gray")

    else:
        raise ValueError(f"Unexpected data shape : {img_norm.shape}")

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

    if nbits:
        # max_val = get_max_val(img, nbits)
        max_val = 2**nbits - 1
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


def plot_cross_section(
    vals,
    idx=None,
    ax=None,
    dB=True,
    plot_db_drop=3,
    min_val=1e-4,
    max_val=None,
    plot_width=None,
    **kwargs,
):
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

    # get cross-section
    if idx is None:
        # if no index, take cross-section with maximum value
        max_idx = np.unravel_index(np.argmax(vals, axis=None), vals.shape)
        idx = max_idx[0]

    cross_section = vals[idx, :].astype(np.float32)

    # normalize
    if max_val is None:
        max_val = cross_section.max()
    cross_section /= max_val
    min_val = max(min_val, cross_section.min())

    if dB:
        cross_section[cross_section < min_val] = min_val
        cross_section = 10 * np.log10(cross_section)
        min_val = 10 * np.log10(min_val)
        ax.set_ylabel("dB")
    x_vals = np.arange(len(cross_section))
    x_vals -= np.argmax(cross_section)
    ax.plot(x_vals, cross_section, **kwargs)
    ax.set_ylim([min_val, 0])
    if plot_width is not None:
        half_width = plot_width // 2 + 1
        ax.set_xlim([-half_width, half_width])
    ax.grid()

    if dB and plot_db_drop:
        cross_section -= np.max(cross_section)
        zero_crossings = np.where(np.diff(np.signbit(cross_section + plot_db_drop)))[0]
        if len(zero_crossings) >= 2:
            zero_crossings -= np.argmax(cross_section)
            width = zero_crossings[-1] - zero_crossings[0]
            ax.set_title(f"-{plot_db_drop}dB width = {width}")
            ax.axvline(x=zero_crossings[0], c="k", linestyle="--")
            ax.axvline(x=zero_crossings[-1], c="k", linestyle="--")
        else:
            warnings.warn(
                "Width could not be determined. Did not detect two -{} points : {}".format(
                    plot_db_drop, zero_crossings
                )
            )

    return ax, cross_section


def plot_autocorr2d(vals, pad_mode="reflect", ax=None):
    """
    Plot 2-D autocorrelation of image.

    Parameters
    ----------
    vals : py:class:`~numpy.ndarray`
        2-D image.
    pad_mode : str
        Desired padding. See NumPy documentation: https://numpy.org/doc/stable/reference/generated/numpy.pad.html
    ax : :py:class:`~matplotlib.axes.Axes`, optional
            `Axes` object to fill, default is to create one.

    Return
    ------
    ax : :py:class:`~matplotlib.axes.Axes`
    autocorr : py:class:`~numpy.ndarray`
    """

    nbit_plot = 8
    max_val_plot = 2**nbit_plot - 1

    # compute autocorrelation
    autocorr = autocorr2d(vals, pad_mode=pad_mode)

    # rescale for plotting
    data = autocorr - np.min(autocorr)
    data = data / np.max(np.abs(data))  # normalize the data to 0 - 1
    data = max_val_plot * data  # Now scale by bit depth
    autocorr_img = data.astype(np.uint8)

    if ax is None:
        _, ax = plt.subplots()

    ax.imshow(autocorr_img, cmap="gray", vmin=0, vmax=max_val_plot)
    ax.axis("off")
    return ax, autocorr
