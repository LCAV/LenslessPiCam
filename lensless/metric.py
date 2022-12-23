# #############################################################################
# metric.py
# =========
# Authors :
# Eric BEZZAM [ebezzam@gmail.com]
# #############################################################################


r"""
Evaluation
==========

After performing reconstruction, we are typically interested in evaluating the
quality of the reconstruction. To this end, four metrics are made available:

* **Mean squared error (MSE)**: lower is better with a minimum of 0.
* **Peak signal-to-noise ratio (PSNR)**: higher is better with values given in decibels (dB).
* **Structural similarity index measure (SSIM)**: higher is better with a maximum of 1.
* **Learned Perceptual Image Patch Similarity (LPIPS)**: perceptual metrics that used a pre-trained neural network on patches. Lower is better with a minimum of 0. *NB: only for RGB!*

On a single file
----------------

The following script can be used to compute the above metrics for a single file, given a reference file:

.. code:: bash

   python scripts/compute_metrics_from_original.py \
   --recon data/reconstruction/admm_thumbs_up_rgb.npy \
   --original data/original/thumbs_up.png \
   --vertical_crop 262 371 \
   --horizontal_crop 438 527 \
   --rotation -0.5

where:

*  ``recon`` is the path to the reconstructed file;
*  ``original`` is the path to the reconstructed file;
*  ``vertical_crop`` specifies the vertical section to crop from the reconstruction;
*  ``horizontal_crop`` specifies the horizontal section to crop from the reconstruction;
*  ``rotation`` specifies a rotate in degrees to align the reconstruction with the original.

More information with an example can be found in
`this Medium article <https://medium.com/@bezzam/image-similarity-metrics-applied-to-diffusercam-21998967af8d>`__.


DiffuserCam Lensless Mirflickr Dataset
--------------------------------------

The `DiffuserCam Lensless Mirflickr Dataset (DLMD) <https://waller-lab.github.io/LenslessLearning/dataset.html>`__
comes with (lensed, lensless) image pairs, namely an image that is captured
with a conventional lensed camera and a corresponding image that is captured
with the diffuser-based camera.

You can run ADMM on DLMD with the following script.

.. code:: bash

   python scripts/evaluate_mirflickr_admm.py --data <FP>

where ``<FP>`` is the path to the dataset.

However, the original dataset is quite large (25000 files, 100 GB). So
we've prepared `this subset <https://drive.switch.ch/index.php/s/vmAZzryGI8U8rcE>`__
(200 files, 725 MB) which you can also pass to the script. It is also
possible to set the number of files.

.. code:: bash

   python scripts/evaluate_mirflickr_admm.py \
   --data DiffuserCam_Mirflickr_200_3011302021_11h43_seed11 \
   --n_files 10 --save

The ``--save`` flag will save a viewable image for each reconstruction.

You can also apply ADMM on a single image and visualize the iterative
reconstruction.

.. code:: bash

   python scripts/apply_admm_single_mirflickr.py \
   --data DiffuserCam_Mirflickr_200_3011302021_11h43_seed11 \
   --fid 172

"""

import numpy as np
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
import lpips as lpips_lib
import torch
from scipy.ndimage import rotate
from lensless.util import resize


def mse(true, est, normalize=True):
    r"""
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
    r"""
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


def ssim(true, est, normalize=True, channel_axis=2, **kwargs):
    r"""
    Compute the mean structural similarity index between two images. Values lie
    within [0, 1]. The closer to 1, the closer the match.

    Parameters
    ----------
    true : :py:class:`~numpy.ndarray`
        Ground-truth image, same shape as `est`.
    est : :py:class:`~numpy.ndarray`
        Test image.
    normalize : bool, optional
        Whether to normalize such that maximum value is 1.
    channel_axis : int or None, optional
        If `None`, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.

    Returns
    -------
    ssim : float
        The mean structural similarity index over the image.

    """
    if normalize:
        true /= true.max()
        est /= est.max()
    return structural_similarity(im1=true, im2=est, channel_axis=channel_axis, **kwargs)


def lpips(true, est, normalize=True):
    r"""
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
    r"""
    Utility function to extract matching region in reconstruction and in original
    image. Later will also be resized to the same dimensions as the estimate.

    Parameters
    ----------
    estimate : :py:class:`~numpy.ndarray`
        Reconstructed image from lensless data.
    original : :py:class:`~numpy.ndarray`
        Original image.
    vertical_crop : (int, int)
        Vertical region (in pixels) to keep.
    horizontal_crop : (int, int)
        Horizontal region (in pixels) to keep.
    rotation : float
        Degrees to rotate reconstruction.
    verbose : bool
        Whether to print extracted and resized shapes.

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
