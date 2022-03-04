import numpy as np
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
import lpips as lpips_lib
import torch
from scipy.ndimage import rotate
from lensless.util import resize


def mse(true, est, normalize=True):
    """
    Compute the mean-squared error between two images. The closer to 0, the
    closer the match.

    Parameters
    ----------
    true : :py:class:`~numpy.ndarray`
        Ground-truth image, same shape as `est`.
    est : :py:class:`~numpy.ndarray`
        Test image.
    normalize : bool
        Whether to normalize such that maximum value is 1.

    Returns
    -------
    mse : float
        The mean-squared error (MSE) metric.

    """
    if normalize:
        true /= true.max()
        est /= est.max()
    return mean_squared_error(image0=true, image1=est)


def psnr(true, est, normalize=True):
    """
    Compute the peak signal to noise ratio (PSNR) for an image. The higher the
    value, the better the match.

    Parameters
    ----------
    true : :py:class:`~numpy.ndarray`
        Ground-truth image, same shape as `est`.
    est : :py:class:`~numpy.ndarray`
        Test image.
    normalize : bool
        Whether to normalize such that maximum value is 1.

    Returns
    -------
    psnr : float
        The PSNR metric.

    """
    if normalize:
        true /= true.max()
        est /= est.max()
    return peak_signal_noise_ratio(image_true=true, image_test=est)


def ssim(true, est, normalize=True):
    """
    Compute the mean structural similarity index between two images. Values lie
    within [0, 1]. The closer to 1, the closer the match.

    Parameters
    ----------
    true : :py:class:`~numpy.ndarray`
        Ground-truth image, same shape as `est`.
    est : :py:class:`~numpy.ndarray`
        Test image.
    normalize : bool
        Whether to normalize such that maximum value is 1.

    Returns
    -------
    mssim : float
        The mean structural similarity index over the image.

    """
    if normalize:
        true /= true.max()
        est /= est.max()
    return structural_similarity(im1=true, im2=est, channel_axis=2)


def lpips(true, est, normalize=True):
    """
    Compute a perceptual metric (LPIPS) between two images. Values lie within
    [0, 1]. The closer to 0, the closer the match.

    GitHub: https://github.com/richzhang/PerceptualSimilarity

    Parameters
    ----------
    true : :py:class:`~numpy.ndarray`
        Ground-truth image, same shape as `est`.
    est : :py:class:`~numpy.ndarray`
        Test image.
    normalize : bool
        Whether to normalize such that maximum value is 1.

    Returns
    -------
    mssim : float
        The LPIPS metric.

    """
    if normalize:
        true /= true.max()
        est /= est.max()
    loss_fn = lpips_lib.LPIPS(net="alex", verbose=False)
    true = torch.from_numpy(
        np.transpose(true, axes=(2, 0, 1))[
            np.newaxis,
        ].copy()
    )
    est = torch.from_numpy(
        np.transpose(est, axes=(2, 0, 1))[
            np.newaxis,
        ].copy()
    )
    return loss_fn.forward(true, est).squeeze().item()


def extract(estimate, original, vertical_crop, horizontal_crop, rotation, verbose=False):
    """
    Utility function to extract matching region in reconstruction and in original
    image. Later will also be resized to the same dimensions as the estimate.

    Parameters
    ----------
    estimate : :py:class:`~numpy.ndarray`
        Reconstructed image from lensless data.
    original :py:class:`~numpy.ndarray`
        Original image.
    vertical_crop : (int, int)
        Vertical region (in pixels) to keep.
    horizontal_crop : (int, int)
        Horizontal region (in pixels) to keep.
    rotation : float
        Degrees to rotate reconstruction.
    verbose : bool

    Returns
    -------
    estimate : :py:class:`~numpy.ndarray`
        Cropped and rotated image estimate.
    img_resize : :py:class:`~numpy.ndarray`
        Original image resized that the dimensions of `estimate`.

    """

    # crop and rotate estimate image
    estimate = rotate(
        estimate[vertical_crop[0] : vertical_crop[1], horizontal_crop[0] : horizontal_crop[1]],
        angle=rotation,
    )
    estimate /= estimate.max()
    estimate = np.clip(estimate, 0, 1)
    if verbose:
        print("estimate cropped: ")
        print(estimate.shape)
        print(estimate.dtype)
        print(estimate.max())

    # resize original image accordingly
    factor = estimate.shape[1] / original.shape[1]
    if verbose:
        print("resize factor", factor)
    img_resize = np.zeros_like(estimate)
    tmp = resize(original, factor=factor).astype(estimate.dtype)
    img_resize[
        : min(estimate.shape[0], tmp.shape[0]), : min(estimate.shape[1], tmp.shape[1])
    ] = tmp[: min(estimate.shape[0], tmp.shape[0]), : min(estimate.shape[1], tmp.shape[1])]
    if verbose:
        print("\noriginal resized: ")
        print(img_resize.shape)
        print(img_resize.dtype)
        print(img_resize.max())

    return estimate, img_resize
