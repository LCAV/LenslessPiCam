import os.path
import rawpy
import cv2
import numpy as np
import warnings
from lensless.util import resize, bayer2rgb, rgb2gray, print_image_info
from lensless.plot import plot_image
from lensless.constants import RPI_HQ_CAMERA_CCM_MATRIX, RPI_HQ_CAMERA_BLACK_LEVEL


def load_image(
    fp,
    verbose=False,
    flip=False,
    bayer=False,
    black_level=RPI_HQ_CAMERA_BLACK_LEVEL,
    blue_gain=None,
    red_gain=None,
    ccm=RPI_HQ_CAMERA_CCM_MATRIX,
    back=None,
    nbits_out=None,
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
        Whether to flip data (vertical and horizontal).
    bayer : bool
        Whether input data is Bayer.
    blue_gain : float
        Blue gain for color correction.
    red_gain : float
        Red gain for color correction.
    black_level : float
        Black level. Default is to use that of Raspberry Pi HQ camera.
    ccm : :py:class:`~numpy.ndarray`
        Color correction matrix. Default is to use that of Raspberry Pi HQ camera.
    back : array_like
        Background level to subtract.
    nbits_out : int
        Output bit depth. Default is to use that of input.

    Returns
    -------
    img :py:class:`~numpy.ndarray`
        RGB image of dimension (height, width, 3).
    """
    assert os.path.isfile(fp)
    if "dng" in fp:
        assert bayer
        raw = rawpy.imread(fp)
        img = raw.raw_image
        # TODO : use raw.postprocess?
        ccm = raw.color_matrix[:, :3]
        black_level = np.array(raw.black_level_per_channel[:3]).astype(np.float32)
    else:
        img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)

    if bayer:
        assert len(img.shape) == 2, img.shape
        if img.max() > 255:
            # HQ camera
            n_bits = 12
        else:
            n_bits = 8

        if back:
            back_img = cv2.imread(back, cv2.IMREAD_UNCHANGED)
            dtype = img.dtype
            img = img.astype(np.float32) - back_img.astype(np.float32)
            img = np.clip(img, a_min=0, a_max=img.max())
            img = img.astype(dtype)
        if nbits_out is None:
            nbits_out = n_bits
        img = bayer2rgb(
            img,
            nbits=n_bits,
            blue_gain=blue_gain,
            red_gain=red_gain,
            black_level=black_level,
            ccm=ccm,
            nbits_out=nbits_out,
        )

    else:
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if flip:
        img = np.flipud(img)
        img = np.fliplr(img)

    if verbose:
        print_image_info(img)

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
    dtype=np.float32,
    nbits_out=None,
    single_psf=False,
    shape=None,
    use_3d=False,
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
    verbose : bool
        Whether to print metadata.
    bayer : bool
        Whether input data is Bayer.
    blue_gain : float
        Blue gain for color correction.
    red_gain : float
        Red gain for color correction.
    dtype : float32 or float64
        Data type of returned data.
    nbits_out : int
        Output bit depth. Default is to use that of input.
    single_psf : bool
        Whether to sum RGB channels into single PSF, same across channels. Done
        in "Learned reconstructions for practical mask-based lensless imaging"
        of Kristina Monakhova et. al.

    Returns
    -------
    psf :py:class:`~numpy.ndarray`
        4-D array of PSF [depth, width, height, color]
    """

    # load image data and extract necessary channels
    if use_3d:
        assert os.path.isfile(fp)
        assert fp.endswith(".npy")
        psf = np.load(fp)
    else:
        psf = load_image(
            fp,
            verbose=verbose,
            flip=flip,
            bayer=bayer,
            blue_gain=blue_gain,
            red_gain=red_gain,
            nbits_out=nbits_out,
        )

    original_dtype = psf.dtype
    psf = np.array(psf, dtype=dtype)

    if use_3d:
        if len(psf.shape) == 3:
            grayscale = True
            psf = psf[:, :, :, np.newaxis]
        else:
            grayscale = False
    else:
        if len(psf.shape) == 3:
            grayscale = False
            psf = psf[np.newaxis, :, :, :]
        else:
            grayscale = True
            psf = psf[np.newaxis, :, :, np.newaxis]

    assert len(psf.shape) == 4
    # check that all depths of the psf have the same shape.
    for i in range(len(psf)):
        assert psf[0].shape == psf[i].shape

    # subtract background, assume black edges
    if bg_pix is None:
        bg = np.zeros(len(np.shape(psf[0])))

    else:

        if grayscale:
            bg = np.mean(psf[:, bg_pix[0] : bg_pix[1], bg_pix[0] : bg_pix[1], :])
            psf -= bg
        else:
            bg = []
            for i in range(3):
                bg_i = np.mean(psf[:, bg_pix[0] : bg_pix[1], bg_pix[0] : bg_pix[1], i])
                psf[:, :, :, i] -= bg_i
                bg.append(bg_i)

        psf = np.clip(psf, a_min=0, a_max=psf.max())
        bg = np.array(bg)

    # resize
    if shape:
        psf = np.array([resize(p, shape=shape) for p in psf])
    elif downsample != 1:
        psf = np.array([resize(p, factor=1 / downsample) for p in psf])

    # todo : ideally resize would need to keep dimensions, but it is used in other places of the code so I'd rather not change it for now
    # todo : check
    if len(psf.shape) == 3:
        psf = np.array(psf[:, :, :, np.newaxis])

        if not grayscale:
            # TODO : in Lensless Learning, they sum channels --> `psf_diffuser = np.sum(psf_diffuser,2)`
            # https://github.com/Waller-Lab/LenslessLearning/blob/master/pre-trained%20reconstructions.ipynb
            psf = np.sum(psf, 3)
            psf = psf[:, :, :, np.newaxis]
        else:
            warnings.warn("Notice : single_psf has no effect for grayscale psf")
            single_psf = False

    # normalize
    if return_float:
        # psf /= psf.max()
        psf /= np.linalg.norm(psf.ravel())
    else:
        psf = psf.astype(original_dtype)

    if return_bg:
        return psf, bg
    else:
        return psf


def load_data(
    psf_fp,
    data_fp,
    downsample=None,
    bg_pix=(5, 25),
    plot=True,
    flip=False,
    bayer=False,
    blue_gain=None,
    red_gain=None,
    gamma=None,
    gray=False,
    dtype=np.float32,
    single_psf=False,
    shape=None,
):
    """
    Load data for image reconstruction.

    Parameters
    ----------
    psf_fp : str
        Full path to PSF file.
    data_fp : str
        Full path to measurement file.
    downsample : int or float
        Downsampling factor.
    bg_pix : tuple, optional
        Section of pixels to take from top left corner to remove background
        level. Set to `None` to omit this step, although it is highly
        recommended.
    plot : bool, optional
        Whether or not to plot PSF and raw data.
    flip : bool
        Whether to flip data (vertical and horizontal).
    bayer : bool
        Whether input data is Bayer.
    blue_gain : float
        Blue gain for color correction.
    red_gain : float
        Red gain for color correction.
    gamma : float, optional
        Optional gamma factor to apply, ONLY for plotting. Default is None.
    gray : bool
        Whether to load as grayscale or RGB.
    dtype : float32 or float64
        Data type of returned data.
    single_psf : bool
        Whether to sum RGB channels into single PSF, same across channels. Done
        in "Learned reconstructions for practical mask-based lensless imaging"
        of Kristina Monakhova et. al.

    Returns
    -------
    psf :py:class:`~numpy.ndarray`
        2-D array of PSF.
    data :py:class:`~numpy.ndarray`
        2-D array of raw measurement data.
    """

    assert os.path.isfile(psf_fp)
    assert os.path.isfile(data_fp)
    if shape is None:
        assert downsample is not None

    if psf_fp.endswith(".npy"):
        use_3d = True
    else:
        use_3d = False

    # load and process PSF data

    psf, bg = load_psf(
        psf_fp,
        downsample=downsample,
        return_float=True,
        bg_pix=bg_pix,
        return_bg=True,
        flip=flip,
        bayer=bayer,
        blue_gain=blue_gain,
        red_gain=red_gain,
        dtype=dtype,
        single_psf=single_psf,
        shape=shape,
        use_3d=use_3d,
    )

    # load and process raw measurement
    data = load_image(data_fp, flip=flip, bayer=bayer, blue_gain=blue_gain, red_gain=red_gain)
    data = np.array(data, dtype=dtype)

    data -= bg
    data = np.clip(data, a_min=0, a_max=data.max())

    if data.shape != psf[0].shape:
        # in DiffuserCam dataset, images are already reshaped
        data = resize(data, shape=psf.shape[1:3])

    data /= np.linalg.norm(data.ravel())

    if len(data.shape) == 3:
        data = data[np.newaxis, :, :, :]
    elif len(data.shape) == 2:
        data = data[np.newaxis, :, :, np.newaxis]

    if gray:
        data = np.array(rgb2gray(data), np.newaxis)
        psf = np.array(rgb2gray(psf), np.newaxis)

    if plot:
        ax = plot_image(psf[0], gamma=gamma)
        ax.set_title("PSF of first depth")
        ax = plot_image(data[0], gamma=gamma)
        ax.set_title("Raw data")

    psf = np.array(psf, dtype=dtype)
    data = np.array(data, dtype=dtype)

    return psf, data
