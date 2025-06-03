# #############################################################################
# io.py
# =================
# Authors :
# Eric BEZZAM [ebezzam@gmail.com]
# #############################################################################


import os.path
import warnings

import cv2
import numpy as np
from PIL import Image

from lensless.hardware.constants import RPI_HQ_CAMERA_BLACK_LEVEL, RPI_HQ_CAMERA_CCM_MATRIX
from lensless.utils.image import bayer2rgb_cc, print_image_info, resize, rgb2gray, get_max_val
from lensless.utils.plot import plot_image


def load_image(
    fp,
    verbose=False,
    flip=False,
    flip_ud=False,
    flip_lr=False,
    bayer=False,
    black_level=RPI_HQ_CAMERA_BLACK_LEVEL,
    blue_gain=None,
    red_gain=None,
    ccm=RPI_HQ_CAMERA_CCM_MATRIX,
    back=None,
    nbits_out=None,
    as_4d=False,
    downsample=None,
    bg=None,
    return_float=False,
    shape=None,
    dtype=None,
    normalize=True,
    bgr_input=True,
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
    as_4d : bool
        Add depth and color dimensions if necessary so that image is 4D: (depth,
        height, width, color).
    downsample : int, optional
        Downsampling factor. Recommended for image reconstruction.
    bg : array_like
        Background level to subtract.
    return_float : bool
        Whether to return image as float array, or unsigned int.
    shape : tuple, optional
        Shape (H, W, C) to resize to.
    dtype : str, optional
        Data type of returned data. Default is to use that of input.
    normalize : bool, default True
        If ``return_float``, whether to normalize data to maximum value of 1.

    Returns
    -------
    img : :py:class:`~numpy.ndarray`
        RGB image of dimension (height, width, 3).
    """
    assert os.path.isfile(fp)

    nbits = None  # input bit depth
    if "dng" in fp:
        import rawpy

        assert bayer
        raw = rawpy.imread(fp)
        img = raw.raw_image
        # # # TODO : use raw.postprocess? to much unknown processing...
        # img = raw.postprocess(
        #     adjust_maximum_thr=0,  # default 0.75
        #     no_auto_scale=False,
        #     # no_auto_scale=True,
        #     gamma=(1, 1),
        #     bright=1,  # default 1
        #     exp_shift=1,
        #     no_auto_bright=True,
        #     # use_camera_wb=True,
        #     # use_auto_wb=False,
        #     # -- gives better balance for PSF measurement
        #     use_camera_wb=False,
        #     use_auto_wb=True,  # default is False? f both use_camera_wb and use_auto_wb are True, then use_auto_wb has priority.
        # )

        # if red_gain is None or blue_gain is None:
        #     camera_wb = raw.camera_whitebalance
        #     red_gain = camera_wb[0]
        #     blue_gain = camera_wb[1]

        nbits = int(np.ceil(np.log2(raw.white_level)))
        ccm = raw.color_matrix[:, :3]
        black_level = np.array(raw.black_level_per_channel[:3]).astype(np.float32)
    elif "npy" in fp or "npz" in fp:
        img = np.load(fp)
    else:
        img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)

    if bayer:
        assert len(img.shape) == 2, img.shape
        if nbits is None:
            if img.max() > 255:
                # HQ camera
                nbits = 12
            else:
                nbits = 8

        if back:
            back_img = cv2.imread(back, cv2.IMREAD_UNCHANGED)
            dtype = img.dtype
            img = img.astype(np.float32) - back_img.astype(np.float32)
            img = np.clip(img, a_min=0, a_max=img.max())
            img = img.astype(dtype)
        if nbits_out is None:
            nbits_out = nbits

        img = bayer2rgb_cc(
            img,
            nbits=nbits,
            blue_gain=blue_gain,
            red_gain=red_gain,
            black_level=black_level,
            ccm=ccm,
            nbits_out=nbits_out,
        )

    else:
        if len(img.shape) == 3 and bgr_input:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    original_dtype = img.dtype

    if flip:
        img = np.flipud(img)
        img = np.fliplr(img)
    if flip_ud:
        img = np.flipud(img)
    if flip_lr:
        img = np.fliplr(img)

    if bg is not None:

        # if bg is float vector, turn into int-valued vector
        if bg.max() <= 1 and img.dtype not in [np.float32, np.float64]:
            bg = bg * get_max_val(img)

        img = img - bg
        img = np.clip(img, a_min=0, a_max=img.max())

    if as_4d:
        if len(img.shape) == 3:
            img = img[np.newaxis, :, :, :]
        elif len(img.shape) == 2:
            img = img[np.newaxis, :, :, np.newaxis]

    if downsample is not None or shape is not None:
        if downsample is not None:
            factor = 1 / downsample
        else:
            factor = None
        img = resize(img, factor=factor, shape=shape)

    if return_float:
        if dtype is None:
            dtype = np.float32
        assert dtype == np.float32 or dtype == np.float64
        img = img.astype(dtype)
        if normalize:
            img /= img.max()

    else:
        if dtype is None:
            dtype = original_dtype
        img = img.astype(dtype)

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
    flip_ud=False,
    flip_lr=False,
    verbose=False,
    bayer=False,
    blue_gain=None,
    red_gain=None,
    dtype=np.float32,
    nbits_out=None,
    single_psf=False,
    shape=None,
    use_3d=False,
    bgr_input=True,
    force_rgb=False,
):
    """
    Load and process PSF for analysis or for reconstruction.

    Basic steps are:
    * Load image.
    * (Optionally) subtract background. Recommended.
    * (Optionally) resize to more manageable size
    * (Optionally) normalize within [0, 1] if using for reconstruction; otherwise cast back to uint for analysis.

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
    psf : :py:class:`~numpy.ndarray`
        4-D array of PSF.
    """

    # load image data and extract necessary channels
    if use_3d:
        assert os.path.isfile(fp)
        if fp.endswith(".npy"):
            psf = np.load(fp)
        elif fp.endswith(".npz"):
            archive = np.load(fp)
            if len(archive.files) > 1:
                print("Warning: more than one array in .npz archive, using first")
            elif len(archive.files) == 0:
                raise ValueError("No arrays in .npz archive")
            psf = np.load(fp)[archive.files[0]]
        else:
            raise ValueError("File format not supported")
    else:
        psf = load_image(
            fp,
            verbose=False,
            flip=flip,
            flip_ud=flip_ud,
            flip_lr=flip_lr,
            bayer=bayer,
            blue_gain=blue_gain,
            red_gain=red_gain,
            nbits_out=nbits_out,
            bgr_input=bgr_input,
        )

    original_dtype = psf.dtype
    max_val = get_max_val(psf)
    psf = np.array(psf, dtype=dtype)

    if force_rgb:
        if len(psf.shape) == 2:
            psf = np.stack([psf] * 3, axis=2)
        elif len(psf.shape) == 3:
            pass

    if use_3d:
        if len(psf.shape) == 3:
            grayscale = True
            psf = psf[:, :, :, np.newaxis]
        else:
            assert len(psf.shape) == 4
            grayscale = False

    else:
        if len(psf.shape) == 3:
            grayscale = False
            psf = psf[np.newaxis, :, :, :]
        else:
            assert len(psf.shape) == 2
            grayscale = True
            psf = psf[np.newaxis, :, :, np.newaxis]

    # check that all depths of the psf have the same shape.
    for i in range(len(psf)):
        assert psf[0].shape == psf[i].shape

    # subtract background, assume black edges
    if bg_pix is None:
        bg = np.zeros(len(np.shape(psf)))

    else:
        # grayscale
        if grayscale:
            bg = np.mean(psf[:, bg_pix[0] : bg_pix[1], bg_pix[0] : bg_pix[1], :])
            psf -= bg

        # rgb
        else:
            bg = []
            for i in range(psf.shape[3]):
                bg_i = np.mean(psf[:, bg_pix[0] : bg_pix[1], bg_pix[0] : bg_pix[1], i])
                psf[:, :, :, i] -= bg_i
                bg.append(bg_i)

        psf = np.clip(psf, a_min=0, a_max=psf.max())
        bg = np.array(bg)

    # resize
    if downsample != 1 or shape is not None:
        psf = resize(psf, shape=shape, factor=1 / downsample)

    if single_psf:
        if not grayscale:
            # TODO : in Lensless Learning, they sum channels --> `psf_diffuser = np.sum(psf_diffuser,2)`
            # https://github.com/Waller-Lab/LenslessLearning/blob/master/pre-trained%20reconstructions.ipynb
            psf = np.sum(psf, axis=3)
            psf = psf[:, :, :, np.newaxis]
        else:
            warnings.warn("Notice : single_psf has no effect for grayscale psf")
            single_psf = False

    # normalize
    if return_float:
        # psf /= psf.max()
        psf /= np.linalg.norm(psf.ravel())
        bg /= max_val
    else:
        psf = psf.astype(original_dtype)

    if verbose:
        print_image_info(psf)

    if return_bg:
        return psf, bg
    else:
        return psf


def load_data(
    psf_fp,
    data_fp,
    background_fp=None,
    return_bg=False,
    remove_background=True,
    return_float=True,
    downsample=None,
    bg_pix=(5, 25),
    plot=True,
    flip=False,
    flip_ud=False,
    flip_lr=False,
    bayer=False,
    blue_gain=None,
    red_gain=None,
    gamma=None,
    gray=False,
    dtype=None,
    single_psf=False,
    shape=None,
    use_torch=False,
    torch_device="cpu",
    normalize=False,
    bgr_input=True,
):
    """
    Load data for image reconstruction.

    Parameters
    ----------
    psf_fp : str
        Full path to PSF file.
    data_fp : str
        Full path to measurement file.
    return_float : bool, optional
        Whether to return PSF as float array, or unsigned int.
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
    dtype : float32 or float64, default float32
        Data type of returned data.
    single_psf : bool
        Whether to sum RGB channels into single PSF, same across channels. Done
        in "Learned reconstructions for practical mask-based lensless imaging"
        of Kristina Monakhova et. al.
    normalize : bool default True
        Whether to normalize data to maximum value of 1.

    Returns
    -------
    psf : :py:class:`~numpy.ndarray`
        2-D array of PSF.
    data : :py:class:`~numpy.ndarray`
        2-D array of raw measurement data.
    """

    assert os.path.isfile(psf_fp)
    assert os.path.isfile(data_fp)
    if shape is None:
        assert downsample is not None

    if dtype is None:
        dtype = np.float32
    elif dtype == "float32":
        dtype = np.float32
    elif dtype == "float64":
        dtype = np.float64
    else:
        raise ValueError("dtype must be float32 or float64")

    use_3d = psf_fp.endswith(".npy") or psf_fp.endswith(".npz")

    # load and process PSF data
    bg = None
    res = load_psf(
        psf_fp,
        downsample=downsample,
        return_float=return_float,
        bg_pix=bg_pix,
        return_bg=True if bg_pix is not None else False,
        flip=flip,
        flip_ud=flip_ud,
        flip_lr=flip_lr,
        bayer=bayer,
        blue_gain=blue_gain,
        red_gain=red_gain,
        dtype=dtype,
        single_psf=single_psf,
        shape=shape,
        use_3d=use_3d,
        bgr_input=bgr_input,
    )
    if bg_pix is not None:
        psf, bg = res
    else:
        psf = res

    # load and process raw measurement
    data = load_image(
        data_fp,
        flip=flip,
        flip_ud=flip_ud,
        flip_lr=flip_lr,
        bayer=bayer,
        blue_gain=blue_gain,
        red_gain=red_gain,
        bg=bg,
        as_4d=True,
        return_float=return_float,
        shape=shape,
        normalize=normalize if background_fp is None else False,
        bgr_input=bgr_input,
    )

    if background_fp is not None:
        bg = load_image(
            background_fp,
            flip=flip,
            bayer=bayer,
            blue_gain=blue_gain,
            red_gain=red_gain,
            as_4d=True,
            return_float=return_float,
            shape=shape,
            normalize=False,
            bgr_input=bgr_input,
        )
        assert bg.shape == data.shape

        if remove_background:
            data -= bg

        # clip to 0
        data = np.clip(data, a_min=0, a_max=data.max())

        if normalize:
            data /= data.max()
            bg /= data.max()  # to normalize by the same factor

    if data.shape != psf.shape:
        # in DiffuserCam dataset, images are already reshaped
        data = resize(data, shape=psf.shape)

        if background_fp is not None:
            bg = resize(bg, shape=psf.shape)

    if data.shape[3] > 1 and psf.shape[3] == 1:
        warnings.warn(
            "Warning: loaded a grayscale PSF with RGB data. Repeating PSF across channels."
            "This may be an error as the PSF and the data are likely from different datasets."
        )
        psf = np.repeat(psf, data.shape[3], axis=3)

    if data.shape[3] == 1 and psf.shape[3] > 1:
        warnings.warn(
            "Warning: loaded a RGB PSF with grayscale data. Repeating data across channels."
            "This may be an error as the PSF and the data are likely from different datasets."
        )
        data = np.repeat(data, psf.shape[3], axis=3)

    if data.shape[3] != psf.shape[3]:
        raise ValueError(
            "PSF and data must have same number of channels, check that they are from the same dataset."
        )

    if gray:
        psf = np.array(rgb2gray(psf), np.newaxis)
        data = np.array(rgb2gray(data), np.newaxis)

    if plot:
        ax = plot_image(psf[0], gamma=gamma)
        ax.set_title("PSF of the first depth")
        ax = plot_image(data[0], gamma=gamma)
        ax.set_title("Raw data")

    psf = np.array(psf, dtype=dtype)
    data = np.array(data, dtype=dtype)
    bg = np.array(bg, dtype=dtype)
    if use_torch:
        import torch

        if dtype == np.float32:
            torch_dtype = torch.float32
        elif dtype == np.float64:
            torch_dtype = torch.float64

        psf = torch.from_numpy(psf).type(torch_dtype).to(torch_device)
        data = torch.from_numpy(data).type(torch_dtype).to(torch_device)
        bg = torch.from_numpy(bg).type(torch_dtype).to(torch_device)

    if return_bg:
        return psf, data, bg
    else:
        return psf, data


def save_image(img, fp, max_val=255, normalize=True):
    """Save as uint8 image."""

    img_tmp = img.copy()

    if normalize:

        if img_tmp.dtype == np.uint16 or img_tmp.dtype == np.uint8:
            img_tmp = img_tmp.astype(np.float32)

        img_tmp -= img_tmp.min()
        img_tmp /= img_tmp.max()
        img_tmp *= max_val
        img_tmp = img_tmp.astype(np.uint8)

    else:

        if img_tmp.dtype == np.float64 or img_tmp.dtype == np.float32:
            # check within [0, 1] and convert to uint8

            normalized = False
            if img_tmp.min() < 0:
                img_tmp -= img_tmp.min()
                normalized = True
            if img_tmp.max() > 1:
                img_tmp /= img_tmp.max()
                normalized = True
            if normalized:
                print(f"Warning (out of range): {fp} normalizing data to [0, 1]")
            img_tmp *= max_val
            img_tmp = img_tmp.astype(np.uint8)

    # save
    if len(img_tmp.shape) == 3 and img_tmp.shape[2] == 3:
        # RGB
        img_tmp = Image.fromarray(img_tmp)
    else:
        # grayscale
        img_tmp = Image.fromarray(img_tmp.squeeze())
    img_tmp.save(fp)


def get_dtype(dtype=None, is_torch=False):
    """
    Get dtype for numpy or torch.

    Parameters
    ----------
    dtype : str, optional
        "float32" or "float64", Default is "float32".
    is_torch : bool, optional
        Whether to return torch dtype.
    """
    if dtype is None:
        dtype = "float32"
    assert dtype == "float32" or dtype == "float64"

    if is_torch:
        import torch

    if dtype is None:
        if is_torch:
            dtype = torch.float32
        else:
            dtype = np.float32
    else:
        if is_torch:
            dtype = torch.float32 if dtype == "float32" else torch.float64
        else:
            dtype = np.float32 if dtype == "float32" else np.float64

    return dtype


def get_ctypes(dtype, is_torch):
    if not is_torch:
        if dtype == np.float32 or dtype == np.complex64:
            return np.complex64, np.complex64
        elif dtype == np.float64 or dtype == np.complex128:
            return np.complex128, np.complex128
        else:
            raise ValueError("Unexpected dtype: ", dtype)
    else:
        import torch

        if dtype == np.float32 or dtype == np.complex64:
            return torch.complex64, np.complex64
        elif dtype == np.float64 or dtype == np.complex128:
            return torch.complex128, np.complex128
        elif dtype == torch.float32 or dtype == torch.complex64:
            return torch.complex64, np.complex64
        elif dtype == torch.float64 or dtype == torch.complex128:
            return torch.complex128, np.complex128
        else:
            raise ValueError("Unexpected dtype: ", dtype)
