# #############################################################################
# metric.py
# =========
# Authors :
# Eric BEZZAM [ebezzam@gmail.com]
# #############################################################################


"""
Evaluation
==========

After performing reconstruction, we are typically interested in evaluating the
quality of the reconstruction. To this end, four metrics are made available:

* **Mean squared error (MSE)**: lower is better with a minimum of 0.
* **Peak signal-to-noise ratio (PSNR)**: higher is better with values given in decibels (dB).
* **Structural similarity index measure (SSIM)**: higher is better with a maximum of 1.
* **Learned Perceptual Image Patch Similarity (LPIPS)**: perceptual metrics that used a pre-trained neural network on patches. Lower is better with a minimum of 0. *NB: only for RGB!*

Note that in the examples below, YAML configuration files are read from the ``configs`` directory.
``--help`` can be used to see the available options.

On a single file
----------------

The script ``scripts/compute_metrics_from_original.py`` shows how to compute the above metrics for a
single file by (1) extraction a region of interest and (2) comparing it to a reference file.

After downloading the example files:

.. code:: bash

    wget https://drive.switch.ch/index.php/s/NdgHlcDeHVDH5ww/download -O data.zip
    unzip data.zip -d data
    cp -r data/*/* data/
    rm -rf data/LenslessPiCam_GitHub_data
    rm data.zip

The script can be run with:

.. code:: bash

    python scripts/compute_metrics_from_original.py


Default parameters will be used from the ``configs/compute_metrics_from_original.yaml`` file.

More information can be found in
`this Medium article <https://medium.com/@bezzam/image-similarity-metrics-applied-to-diffusercam-21998967af8d>`__.


DiffuserCam Lensless Mirflickr Dataset (DLMD)
---------------------------------------------

The `DiffuserCam Lensless Mirflickr Dataset (DLMD) <https://waller-lab.github.io/LenslessLearning/dataset.html>`__
comes with (lensed, lensless) image pairs, namely an image that is captured
with a conventional lensed camera and a corresponding image that is captured
with the diffuser-based camera.

The original dataset is quite large (25000 files, 100 GB). So we've prepared
`this subset <https://drive.switch.ch/index.php/s/vmAZzryGI8U8rcE>`__
(200 files, 725 MB).

After downloading the data, you can run ADMM on the subset with the following script.

.. code:: bash

   python scripts/evaluate_mirflickr_admm.py

The default parameters can be found in the ``configs/evaluate_mirflickr_admm.yaml`` file.

It is also possible to set the number of files.

.. code:: bash

   python scripts/evaluate_mirflickr_admm.py n_files=10 save=True


The ``save`` option will save a viewable image for each reconstruction.

You can also apply ADMM on a single image and visualize the iterative
reconstruction.

.. code:: bash

   python scripts/apply_admm_single_mirflickr.py

The default parameters can be found in the ``configs/apply_admm_single_mirflickr.yaml`` file.


Benchmarking with PyTorch
-------------------------

It may be useful to benchmark reconstruction algorithms with PyTorch, e.g.
with a *parallel* dataset of lensless and corresponding lensed images.

:py:class:`~lensless.benchmark.ParallelDataset` is a PyTorch :py:class:`~torch.utils.data.Dataset` object that can be used
to load a parallel dataset of lensless and corresponding lensed images.
The function :py:func:`~lensless.benchmark.benchmark` can be used to evaluate a reconstruction
algorithm on a parallel dataset in batches.

Running the following file will evaluate ADMM on a subset of DLMD:

.. code:: bash

    python lensless/benchmark.py

"""

import numpy as np
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
import lpips as lpips_lib
import torch
import torch.nn.functional as F
from torchmetrics.multimodal import CLIPImageQualityAssessment
from scipy.ndimage import rotate
from lensless.utils.image import resize


# Initialize CLIP-IQA model
clip_iqa_model = CLIPImageQualityAssessment(
    model_name_or_path=("clip_iqa"),
    prompts=("noisiness", ), # TODO change if different metric is required
    ).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))


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
        true = np.array(true, dtype=np.float32)
        est = np.array(est, dtype=np.float32)
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
        true = np.array(true, dtype=np.float32)
        est = np.array(est, dtype=np.float32)
        true /= true.max()
        est /= est.max()
    return peak_signal_noise_ratio(image_true=true, image_test=est)


def ssim(true, est, normalize=True, channel_axis=2, data_range=None, **kwargs):
    """
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
    data_range : float or None, optional
        The data range of the input image (distance between minimum and maximum
        possible values). By default, this is estimated from the image data-type.

    Returns
    -------
    ssim : float
        The mean structural similarity index over the image.

    """
    if normalize:
        true = np.array(true, dtype=np.float32)
        est = np.array(est, dtype=np.float32)
        true /= true.max()
        est /= est.max()

    if data_range is None:
        # recommended to explictly pass data range
        data_range = true.max() - true.min()
    return structural_similarity(
        im1=true, im2=est, channel_axis=channel_axis, data_range=data_range, **kwargs
    )


LPIPS_MIN_DIM = 31


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
    lpips : float
        The LPIPS metric.

    """
    if np.min(true.shape[:2]) < LPIPS_MIN_DIM:
        raise ValueError(
            f"LPIPS requires images to be at least {LPIPS_MIN_DIM}x{LPIPS_MIN_DIM} pixels."
        )
    if normalize:
        true = np.array(true, dtype=np.float32)
        est = np.array(est, dtype=np.float32)
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

def extract(
    estimate, original, vertical_crop=None, horizontal_crop=None, rotation=0, verbose=False
):
    """
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

    if vertical_crop is None:
        vertical_crop = (0, estimate.shape[0])
    if horizontal_crop is None:
        horizontal_crop = (0, estimate.shape[1])

    # crop and rotate estimate image
    if rotation:
        estimate = rotate(
            estimate[vertical_crop[0] : vertical_crop[1], horizontal_crop[0] : horizontal_crop[1]],
            angle=rotation,
            mode="nearest",
            reshape=False,
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

def clip_iqa(true, est, normalize=True):
    """
    Computes the CLIP Image Quality Assessment (CLIP-IQA) score between the true and estimated images.
    Args:
        true (Tensor): The ground truth image tensor.
        est (Tensor): The estimated image tensor.
        normalize (bool, optional): If True, normalize the images before computing the CLIP-IQA score. Default is True.
    Returns:
        float: The CLIP-IQA score.
    """
    # if normalize:
    #     true = np.array(true, dtype=np.float32)
    #     est = np.array(est, dtype=np.float32)
    #     true /= true.max()
    #     est /= est.max()

    # Compute CLIP-IQA
    with torch.no_grad():
        # Resize images to 224x224 for CLIP-IQA
        outputs_resized = F.interpolate(
            est, size=(224, 224), mode="bilinear", align_corners=False
        )

        outputs_3d = outputs_resized

        #clip_iqa_scores = self.clip_iqa(outputs_3d)


        return clip_iqa_model(outputs_3d)

        # Compute CLIP-IQA scores over the batch
        clip_iqa = clip_iqa_scores.mean().item()