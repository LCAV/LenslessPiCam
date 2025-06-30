# #############################################################################
# plot.py
# =================
# Authors :
# Eric BEZZAM [ebezzam@gmail.com]
# Julien SAHLI [julien.sahli@epfl.ch]
# #############################################################################


import numpy as np
import warnings
import matplotlib.pyplot as plt
import os
import json

from lensless.utils.image import FLOAT_DTYPES, get_max_val, gamma_correction, autocorr2d


def plot_image(img, ax=None, gamma=None, normalize=True):
    """
    Plot image data.

    Parameters
    ----------
    img : :py:class:`~numpy.ndarray`
        Data to plot.
    ax : :py:class:`~matplotlib.axes.Axes`, optional
        `Axes` object to fill for plotting/saving, default is to create one.
    gamma : float, optional
        Gamma correction factor to apply for plots. Default is None.
    normalize : bool, optional
        Whether to normalize data to maximum range. Default is True.

    Returns
    -------
    ax : :py:class:`~matplotlib.axes.Axes`
        Axes on which image is plot.
    """

    # if we have only 1 depth, remove the axis
    if img.shape[0] == 1:
        img = img[0]

    # if we have only 1 color channel, remove the axis
    if img.shape[-1] == 1:
        img = img[..., 0]

    disp_img = None
    cmap = None

    # full 3D RGB format : [depth, width, height, color]
    is_3d = False
    if len(img.shape) == 4:
        disp_img = [np.sum(img, axis=axis) for axis in range(3)]
        cmap = None
        is_3d = True

    # data of length 3 means we have to infer whichever depth or color is missing, based on shape.
    elif len(img.shape) == 3:
        if img.shape[2] == 3:  # 2D rgb
            disp_img = [img]
            cmap = None

        else:  # 3D grayscale
            disp_img = [np.sum(img, axis=axis) for axis in range(3)]
            cmap = "gray"
            is_3d = True

    # data of length 2 means we have only width and height
    elif len(img.shape) == 2:  # 2D grayscale
        disp_img = [img]
        cmap = "gray"

    else:
        raise ValueError(f"Unexpected data shape : {img.shape}")

    max_val = [d.max() for d in disp_img]

    if not normalize:
        for i in range(len(max_val)):
            if disp_img[i].dtype not in FLOAT_DTYPES:
                max_val[i] = get_max_val(disp_img[i])
            else:
                max_val[i] = 1

    assert len(disp_img) == 1 or len(disp_img) == 3

    # need float image for gamma correction and plotting
    img_norm = disp_img.copy()
    for i in range(len(img_norm)):
        img_norm[i] = disp_img[i] / max_val[i]
        if gamma and gamma > 1:
            img_norm[i] = gamma_correction(img_norm[i], gamma=gamma)

    if ax is None:
        if not is_3d:
            _, ax = plt.subplots()
        else:
            _, ax = plt.subplots(2, 2)
    else:
        if is_3d:
            assert len(ax) == 2
            assert len(ax[0]) == 2

    if len(img_norm) == 1:
        ax.imshow(img_norm[0], cmap=cmap)

    else:

        # plot each projection separately
        ax[0, 0].imshow(img_norm[0], cmap=cmap)
        ax[0, 1].imshow(np.swapaxes(img_norm[2], 0, 1), cmap=cmap)
        ax[1, 0].imshow(img_norm[1], cmap=cmap)
        ax[1, 1].axis("off")
        ax[0, 1].set_xlabel("Depth")
        ax[1, 0].set_ylabel("Depth")

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
        Axes on which histogram is plot.
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
        Axes on which cross-section is plot.
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

    ax.set_title("Cross-section")
    if dB and plot_db_drop:
        cross_section -= np.max(cross_section)
        zero_crossings = np.where(np.diff(np.signbit(cross_section + plot_db_drop)))[0]
        if len(zero_crossings) >= 2:
            zero_crossings -= np.argmax(cross_section)

            # width = zero_crossings[-1] - zero_crossings[0]
            # ax.axvline(x=zero_crossings[0], c="k", linestyle="--")
            # ax.axvline(x=zero_crossings[-1], c="k", linestyle="--")

            first_crossing = np.abs(zero_crossings[np.argmin(np.abs(zero_crossings))])
            width = 2 * np.abs(first_crossing)
            ax.axvline(x=-first_crossing, c="k", linestyle="--")
            ax.axvline(x=+first_crossing, c="k", linestyle="--")
            print(f"-{plot_db_drop}dB width = {width} pixels")
            ax.set_title("Cross-section")
            ax.set_xlabel(f"-{plot_db_drop}dB width = {width}")

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
        Axes on which auto-correlation is plot.
    autocorr : py:class:`~numpy.ndarray`
        Auto-correlation.
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


def plot_autocorr_rgb(img, width=3, figsize=None, plot_psf=False, psf_gamma=2.2, verbose=False):
    """
    Plot autocorrelation of each channel of an image.

    Parameters
    ----------
    img : py:class:`~numpy.ndarray`
        2-D image.
    width : int, optional
        Width of cross-section to plot. Default is 3dB.

    """

    assert len(img.shape) == 3, "Image must be 3D"
    assert img.shape[2] == 3, "Image must have 3 color channels"

    if plot_psf:
        _, ax_auto = plt.subplots(ncols=3, nrows=3, num="Autocorrelations", figsize=figsize)
    else:
        _, ax_auto = plt.subplots(ncols=3, nrows=2, num="Autocorrelations", figsize=figsize)

    for i, c in enumerate(["Red", "Green", "Blue"]):
        if plot_psf:
            plot_image(
                img[:, :, i],
                ax=ax_auto[0][i],
                gamma=psf_gamma,
                normalize=True,
            )
            # ax_auto[0][i].imshow(img[:, :, i], cmap="gray")
            ax_auto[0][i].axis("off")
            ax_auto[0][i].set_title(f"{c} PSF")

        # plot autocorrelation
        _, autocorr_c = plot_autocorr2d(img[:, :, i], ax=ax_auto[1 if plot_psf else 0][i])
        ax_auto[1][i].set_title("Autocorrelation")
        # # -- horizontal cross-section
        max_idx = np.unravel_index(np.argmax(autocorr_c, axis=None), autocorr_c.shape)
        idx = max_idx[0]
        # ax_auto[1][i].axhline(y=idx, c=c, linestyle="--")

        ax, cross_section = plot_cross_section(
            autocorr_c,
            idx=idx,
            color=c,
            ax=ax_auto[2 if plot_psf else 1][i],
            plot_db_drop=width,
        )
        if verbose:
            print(f"Maximum drop in {c} channel: {cross_section.max() - cross_section.min()}")
        if i != 0:
            ax.set_ylabel("")
    return ax


def compare_models(model_paths, max_epoch=None, linewidth=2, fontsize=18, metrics=None):
    """
    Plot train and test loss for multiple models, and print metrics for best epoch.

    Parameters
    ----------
    model_paths : dict
        Dictionary of model names and their paths.
    max_epoch : int, optional
        Maximum epoch to plot. Default is None.
    linewidth : int, optional
        Line width for plot. Default is 2.
    fontsize : int, optional
        Font size for plot. Default is 18.
    metrics : list, optional
        List of metrics to print. Default is ["PSNR", "SSIM", "LPIPS_Vgg"].
    """

    if metrics is None:
        metrics = ["PSNR", "SSIM", "LPIPS_Vgg"]

    # plot train and test loss
    import matplotlib.colors as mcolors

    plot_colors = list(mcolors.TABLEAU_COLORS.keys())

    _, ax = plt.subplots()
    for model in model_paths:
        model_path = model_paths[model]
        _metrics_path = os.path.join(model_path, "metrics.json")

        assert os.path.exists(_metrics_path), f"Path {_metrics_path} does not exist"
        _test_metrics = json.load(open(_metrics_path))

        color = plot_colors.pop()
        train_loss = np.array(_test_metrics["LOSS"])
        if max_epoch is not None:
            train_loss = train_loss[: max_epoch + 1]
        ax.plot(
            train_loss, label=model + " (train)", color=color, linestyle="--", linewidth=linewidth
        )

        test_loss = np.array(_test_metrics["MSE"]) + np.array(_test_metrics["LPIPS_Vgg"])
        if max_epoch is not None:
            test_loss = test_loss[: max_epoch + 1]
        ax.plot(test_loss, label=model + " (test)", linestyle="-", color=color, linewidth=linewidth)

        # best_epoch = np.argmin(test_loss)
        best_epoch = _test_metrics["best_epoch"]
        print(f"\n-- {model} --")
        print(f"Best epoch for {model}: {best_epoch} / {len(test_loss)-1}")
        print(f"Best test loss for {model}: {test_loss[best_epoch]}")
        # print metrics
        for _metric in metrics:
            print(f"{_metric}: {np.array(_test_metrics[_metric])[best_epoch]:.3}")

    # set font size
    ax.tick_params(axis="both", which="major", labelsize=fontsize)
    ax.set_xlabel("Epoch", fontsize=fontsize)
    ax.set_title("Train-test loss", fontsize=fontsize)

    # legend outside
    ax.legend(loc="upper right", fontsize=fontsize)
    # ax.set_ylim([0.4, 1]);
    if max_epoch is not None:
        ax.set_xlim([0, max_epoch])

    ax.grid()
    return ax
