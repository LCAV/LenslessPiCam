import os.path

import cv2
import numpy as np
from diffcam.util import resize, bayer2rgb
from diffcam.constants import RPI_HQ_CAMERA_CCM_MATRIX, RPI_HQ_CAMERA_BLACK_LEVEL


def load_image(
    fp,
    verbose=False,
    flip=False,
    bayer=False,
    black_level=RPI_HQ_CAMERA_BLACK_LEVEL,
    bg=None,
    rg=None,
    ccm=RPI_HQ_CAMERA_CCM_MATRIX,
):
    """
    Load image as numpy array.

    Parameters
    ----------
    fp : str
        Full path to file.
    verbose : bool, optional
        Whether to plot into about file.
    flip : bool
    bayer : bool
    bg : float
    rg : float

    Returns
    -------
    img :py:class:`~numpy.ndarray`
        RGB image of dimension (height, width, 3).
    """
    assert os.path.isfile(fp)
    img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)

    if bayer:
        assert len(img.shape) == 2, img.shape
        if img.max() > 255:
            # HQ camera
            n_bits = 12
        else:
            n_bits = 8
        img = bayer2rgb(img, nbits=n_bits, bg=bg, rg=rg, black_level=black_level, ccm=ccm)

    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if flip:
        img = np.flipud(img)
        img = np.fliplr(img)

    if verbose:
        # print image properties
        print("dimensions : {}".format(img.shape))
        print("data type : {}".format(img.dtype))
        print("max  : {}".format(img.max()))
        print("min  : {}".format(img.min()))
        print("mean : {}".format(img.mean()))

    return img


def load_psf(
    fp,
    downsample=1,
    return_float=True,
    bg_pix=(5, 25),
    return_bg=False,
    flip=False,
    verbose=False,
    bayer=False,
    blue_gain=None,
    red_gain=None,
):
    """
    Load and process PSF for analysis or for reconstruction.

    Basic steps are:
    - Load image.
    - (Optionally) subtract background. Recommended.
    - (Optionally) resize to more manageable size
    - (Optionally) normalize within [0, 1] if using for reconstruction; otherwise cast back to uint for analysis.

    Parameters
    ----------
    fp : str
        Full path to file.
    downsample : int, optional
        Downsampling factor. Recommended for image reconstruction.
    return_float : bool, optional
        Whether to return PSF as float array, or unsigned int.
    bg_pix : tuple, optional
        Section of pixels to take from top left corner to remove background level. Set to `None` to omit this
        step, althrough it is highly recommended.
    return_bg : bool, optional
        Whether to return background level, for removing from data for reconstruction.
    flip : bool, optional
        Whether to flip up-down and left-right.
    verbose

    Returns
    -------
    psf :py:class:`~numpy.ndarray`
        2-D array of PSF.
    """

    # load image data and extract necessary channels
    psf = load_image(fp, verbose=verbose, flip=flip, bayer=bayer, bg=blue_gain, rg=red_gain)

    original_dtype = psf.dtype
    psf = np.array(psf, dtype="float32")

    # subtract background, assume black edges
    bg = np.zeros(3)
    if bg_pix is not None:
        bg = []
        for i in range(3):
            bg_i = np.mean(psf[bg_pix[0] : bg_pix[1], bg_pix[0] : bg_pix[1], i])
            psf[:, :, i] -= bg_i
            bg.append(bg_i)
        psf = np.clip(psf, a_min=0, a_max=psf.max())
        bg = np.array(bg)

    # resize
    if downsample != 1:
        psf = resize(psf, 1 / downsample)

    # normalize
    if return_float:
        psf /= np.linalg.norm(psf.ravel())
    else:
        psf = psf.astype(original_dtype)

    if return_bg:
        return psf, bg
    else:
        return psf
