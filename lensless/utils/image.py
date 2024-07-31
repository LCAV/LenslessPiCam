# #############################################################################
# image.py
# =================
# Authors :
# Eric BEZZAM [ebezzam@gmail.com]
# Julien SAHLI [julien.sahli@epfl.ch]
# #############################################################################


import cv2
import scipy.signal
import numpy as np
from lensless.hardware.constants import RPI_HQ_CAMERA_CCM_MATRIX, RPI_HQ_CAMERA_BLACK_LEVEL

try:
    import torch
    import torchvision.transforms as tf
    from torchvision.transforms.functional import rgb_to_grayscale, rotate

    torch_available = True
except ImportError:
    torch_available = False

SUPPORTED_BIT_DEPTH = np.array([8, 10, 12, 16])
FLOAT_DTYPES = [np.float32, np.float64]


def resize(img, factor=None, shape=None, interpolation=cv2.INTER_CUBIC):
    """
    Resize by given factor.

    Parameters
    ----------
    img : :py:class:`~numpy.ndarray`
        Image to downsample
    factor : int or float
        Resizing factor.
    shape : tuple
        Shape to copy ([depth,] height, width, color). If provided, (height, width) is used.
    interpolation : OpenCV interpolation method
        See https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#cv2.resize

    Returns
    -------
    img : :py:class:`~numpy.ndarray`
        Resized image.
    """
    min_val = img.min()
    max_val = img.max()
    img_shape = np.array(img.shape)[-3:-1]

    assert not ((factor is None) and (shape is None)), "Must specify either factor or shape"
    new_shape = tuple(img_shape * factor) if shape is None else shape[-3:-1]
    new_shape = [int(i) for i in new_shape]

    if np.array_equal(img_shape, new_shape):
        return img

    if torch_available:
        # torch resize expects an input of form [color, depth, width, height]
        tmp = np.moveaxis(img, -1, 0)
        tmp = torch.from_numpy(tmp.copy())
        resized = tf.Resize(size=new_shape, antialias=True)(tmp).numpy()
        resized = np.moveaxis(resized, 0, -1)

    else:
        resized = np.array(
            [
                cv2.resize(img[i], dsize=tuple(new_shape[::-1]), interpolation=interpolation)
                for i in range(img.shape[-4])
            ]
        )
        # OpenCV discards channel dimension if it is 1, put it back
        if len(resized.shape) == 3:
            # resized = resized[:, :, :, np.newaxis]
            resized = np.expand_dims(resized, axis=-1)

    return np.clip(resized, min_val, max_val)


def rotate_HWC(img, angle):

    # to CHW
    img = img.movedim(-1, -3)

    remove_depth_dim = False
    if len(img.shape) == 5:
        # remove singleton depth dimension before, otherwise doesn't work..
        if img.shape[1] == 1:
            img = img.squeeze(1)
            remove_depth_dim = True

    # rotate
    img_rot = rotate(img, angle, expand=False)

    if remove_depth_dim:
        img_rot = img_rot.unsqueeze(1)

    # to HWC
    img_rot = img_rot.movedim(-3, -1)
    return img_rot


def shift_with_pad(img, shift, pad_mode="constant", axis=(0, 1)):
    n_dim = len(img.shape)

    # padding for shifting
    pad_width = [(0, 0) for _ in range(n_dim)]
    for i, s in zip(axis, shift):
        if s < 0:
            pad_width[i] = (0, -s)
        else:
            pad_width[i] = (s, 0)
    pad_width = tuple(pad_width)

    # slice after padding
    slice_obj = [slice(None) for _ in range(n_dim)]
    for i, s in zip(axis, shift):
        if s < 0:
            slice_obj[i] = slice(-s, None)
        elif s == 0:
            slice_obj[i] = slice(None)
        else:
            slice_obj[i] = slice(None, -s)

    if torch_available and isinstance(img, torch.Tensor):

        # concatenate all tuples into single one for torch padding
        # -- in reverse order as pytorch pads dimensions in reverse order
        pad_width_torch = tuple([item for sublist in pad_width[::-1] for item in sublist])
        shifted = torch.nn.functional.pad(img, pad=pad_width_torch, mode=pad_mode)

    else:
        shifted = np.pad(img, pad_width=pad_width, mode=pad_mode)

    shifted = shifted[tuple(slice_obj)]

    return shifted


def is_grayscale(img):
    """
    Check if image is RGB. Assuming image is of shape ([depth,] height, width, color).

    Parameters
    ----------
    img : :py:class:`~numpy.ndarray` or :py:class:`~torch.Tensor`
        Image array.

    Returns
    -------
    bool
        Whether image is RGB.
    """
    return img.shape[-1] == 1


def rgb2gray(rgb, weights=None, keepchanneldim=True):
    """
    Convert RGB array to grayscale.

    Parameters
    ----------
    rgb : :py:class:`~numpy.ndarray` or :py:class:`~torch.Tensor`
        ([Depth,] Height, Width, Channel) image.
    weights : :py:class:`~numpy.ndarray`
        [Optional] (3,) weights to convert from RGB to grayscale. Only used for NumPy arrays.
    keepchanneldim : bool
        Whether to keep the channel dimension. Default is True.

    Returns
    -------
    img :py:class:`~numpy.ndarray`
        Grayscale image of dimension ([depth,] height, width [, 1]).

    """

    use_torch = False
    if torch_available:
        if torch.is_tensor(rgb):
            use_torch = True

    if use_torch:

        # move channel dimension to third to last
        if len(rgb.shape) == 4:
            rgb = rgb.permute(0, 3, 1, 2)
        elif len(rgb.shape) == 3:
            rgb = rgb.permute(2, 0, 1)
        else:
            raise ValueError("Input must be at least 3D.")

        image = rgb_to_grayscale(rgb)

        # move channel dimension to last
        if len(rgb.shape) == 4:
            image = image.permute(0, 2, 3, 1)
        elif len(rgb.shape) == 3:
            image = image.permute(1, 2, 0)

        if not keepchanneldim:
            image = image.squeeze(-1)

        return image

    else:

        if weights is None:
            weights = np.array([0.299, 0.587, 0.114])
        assert len(weights) == 3

        if len(rgb.shape) == 4:
            image = np.tensordot(rgb, weights, axes=((3,), 0))
        elif len(rgb.shape) == 3:
            image = np.tensordot(rgb, weights, axes=((2,), 0))
        else:
            raise ValueError("Input must be at least 3D.")

        if keepchanneldim:
            return image[..., np.newaxis]
        else:
            return image


def gamma_correction(vals, gamma=2.2):
    """
    Apply `gamma correction <https://www.cambridgeincolour.com/tutorials/gamma-correction.htm>`__.

    Parameters
    ----------
    vals : :py:class:`~numpy.ndarray`
        RGB values to gamma correct.
    gamma : float, optional
        Gamma correction factor.

    Returns
    -------
    vals : :py:class:`~numpy.ndarray`
        Gamma-corrected data.

    """

    # Rec. 709 gamma correction
    # http://www.fourmilab.ch/documents/specrend/specrend.c
    cc = 0.018
    inv_gam = 1 / gamma
    clip_val = (1.099 * np.power(cc, inv_gam) - 0.099) / cc
    return np.where(vals < cc, vals * clip_val, 1.099 * np.power(vals, inv_gam) - 0.099)


def get_max_val(img, nbits=None):
    """
    For uint image.

    Parameters
    ----------
    img : :py:class:`~numpy.ndarray`
        Image array.
    nbits : int, optional
        Number of bits per pixel. Detect if not provided.

    Returns
    -------
    max_val : int
        Maximum pixel value.
    """
    assert img.dtype not in FLOAT_DTYPES
    if nbits is None:
        nbits = int(np.ceil(np.log2(img.max())))

    if nbits not in SUPPORTED_BIT_DEPTH:
        nbits = SUPPORTED_BIT_DEPTH[nbits < SUPPORTED_BIT_DEPTH][0]
    max_val = 2**nbits - 1
    if img.max() > max_val:
        new_nbit = int(np.ceil(np.log2(img.max())))
        print(f"Detected pixel value larger than {nbits}-bit range, using {new_nbit}-bit range.")
        max_val = 2**new_nbit - 1
    return max_val


def bayer2rgb_cc(
    img,
    nbits,
    down=None,
    blue_gain=None,
    red_gain=None,
    black_level=RPI_HQ_CAMERA_BLACK_LEVEL,
    ccm=RPI_HQ_CAMERA_CCM_MATRIX,
    nbits_out=None,
):
    """
    Convert raw Bayer data to RGB with the following steps:

    #. Demosaic with bi-linear interpolation, mapping the Bayer array to RGB.
    #. Black level removal.
    #. White balancing, applying gains to red and blue channels.
    #. Color correction matrix.
    #. Clip.

    Parameters
    ----------
    img : :py:class:`~numpy.ndarray`
        2D Bayer data to convert to RGB.
    nbits : int
        Bit depth of input data.
    blue_gain : float
        Blue gain.
    red_gain : float
        Red gain.
    black_level : float
        Black level. Default is to use that of Raspberry Pi HQ camera.
    ccm : :py:class:`~numpy.ndarray`
        Color correction matrix. Default is to use that of Raspberry Pi HQ camera.
    nbits_out : int
        Output bit depth. Default is to use that of input.

    Returns
    -------
    rgb : :py:class:`~numpy.ndarray`
        RGB data.
    """
    assert len(img.shape) == 2, img.shape
    if nbits_out is None:
        nbits_out = nbits
    if nbits_out > 8:
        dtype = np.uint16
    else:
        dtype = np.uint8

    # demosaic Bayer data
    img = cv2.cvtColor(img, cv2.COLOR_BayerRG2RGB)

    # downsample
    if down is not None:
        img = resize(img[None, ...], factor=1 / down, interpolation=cv2.INTER_CUBIC)[0]

    # correction
    img = img - black_level
    if red_gain:
        img[:, :, 0] *= red_gain
    if blue_gain:
        img[:, :, 2] *= blue_gain
    img = img / (2**nbits - 1 - black_level)
    img[img > 1] = 1

    img = (img.reshape(-1, 3, order="F") @ ccm.T).reshape(img.shape, order="F")
    img[img < 0] = 0
    img[img > 1] = 1
    return (img * (2**nbits_out - 1)).astype(dtype)


def print_image_info(img):
    """
    Print dimensions, data type, max, min, mean.
    """
    print("dimensions : {}".format(img.shape))
    print("data type : {}".format(img.dtype))
    print("max  : {}".format(img.max()))
    print("min  : {}".format(img.min()))
    print("mean : {}".format(img.mean()))


def autocorr2d(vals, pad_mode="reflect"):
    """
    Compute 2-D autocorrelation of image via the FFT.

    Parameters
    ----------
    vals : :py:class:`~numpy.ndarray`
        2-D image.
    pad_mode : str
        Desired padding. See NumPy documentation: https://numpy.org/doc/stable/reference/generated/numpy.pad.html

    Return
    ------
    autocorr : :py:class:`~numpy.ndarray`
    """

    shape = vals.shape
    assert len(shape) == 2

    # pad
    vals_padded = np.pad(
        vals,
        pad_width=((shape[0] // 2, shape[0] // 2), (shape[1] // 2, shape[1] // 2)),
        mode=pad_mode,
    )

    # compute autocorrelation via DFT
    X = np.fft.rfft2(vals_padded)
    autocorr = np.fft.ifftshift(np.fft.irfft2(X * X.conj()))

    # remove padding
    return autocorr[shape[0] // 2 : -shape[0] // 2, shape[1] // 2 : -shape[1] // 2]


def corr2d(im1, im2):
    """
    Source: https://stackoverflow.com/a/24769222

    """

    # get rid of the color channels by performing a grayscale transform
    # the type cast into 'float' is to avoid overflows
    im1_gray = np.sum(im1.astype("float"), axis=2)
    im2_gray = np.sum(im2.astype("float"), axis=2)

    # get rid of the averages, otherwise the results are not good
    im1_gray -= np.mean(im1_gray)
    im2_gray -= np.mean(im2_gray)

    # calculate the correlation image; note the flipping of onw of the images
    return scipy.signal.fftconvolve(im1_gray, im2_gray[::-1, ::-1], mode="same")


def rgb2bayer(img, pattern):
    """
    Converting RGB image to separated Bayer channels.

    Parameters
    ----------
    img : :py:class:`~numpy.ndarray`
        Image in RGB format.
    pattern : str
        Bayer pattern: `RGGB`, `BGGR`, `GRBG`, `GBRG`.

    Returns
    -------
    :py:class:`~numpy.ndarray`
        Image converted to the Bayer format `[R, Gr, Gb, B]`. `Gr` and `Gb` are for the green pixels that are on the same line as the red and blue pixels respectively.
    """

    # Verifying that the pattern is a proper Bayer pattern
    pattern = pattern.upper()
    assert pattern in [
        "RGGB",
        "BGGR",
        "GRBG",
        "GBRG",
    ], "Bayer pattern must be in ['RGGB', 'BGGR', 'GRBG', 'GBRG']"

    # Doubling the size of the image to anticipatie shrinking from Bayer transformation
    height, width, _ = img.shape
    resized = resize(img, shape=(height * 2, width * 2, 3))

    # Separating each Bayer channel

    if pattern == "RGGB":
        # RGGB pattern *------*
        #              | R  G |
        #              | G  B |
        #              *------*
        r = resized[::2, ::2, 0]
        gr = resized[1::2, ::2, 1]
        gb = resized[::2, 1::2, 1]
        b = resized[1::2, 1::2, 2]
        img_bayer = np.dstack((r, gr, gb, b))

    elif pattern == "BGGR":
        # BGGR pattern *------*
        #              | B  G |
        #              | G  R |
        #              *------*
        r = resized[1::2, 1::2, 0]
        gr = resized[::2, 1::2, 1]
        gb = resized[1::2, ::2, 1]
        b = resized[::2, ::2, 2]
        img_bayer = np.dstack((b, gb, gr, r))

    elif pattern == "GRBG":
        # GRGB pattern *------*
        #              | G  R |
        #              | B  G |
        #              *------*
        r = resized[1::2, ::2, 0]
        gr = resized[::2, ::2, 1]
        gb = resized[1::2, 1::2, 1]
        b = resized[::2, 1::2, 2]
        img_bayer = np.dstack((gr, r, b, gb))

    else:
        # GBRG pattern *------*
        #              | G  B |
        #              | R  G |
        #              *------*
        r = resized[::2, 1::2, 0]
        gr = resized[1::2, 1::2, 1]
        gb = resized[::2, ::2, 1]
        b = resized[1::2, ::2, 2]
        img_bayer = np.dstack((gb, b, r, gr))

    return img_bayer


def bayer2rgb(X_bayer, pattern):
    """
    Converting 4-channel Bayer image to RGB by averaging the two green channels.

    Parameters
    ----------
    X_bayer : :py:class:`~numpy.ndarray`
        Image in RGB format.
    pattern : str
        Bayer pattern: `RGGB`, `BGGR`, `GRBG`, `GBRG`.

    Returns
    -------
    :py:class:`~numpy.ndarray`
        Image converted to the RGB format.
    """

    # Verifying that the pattern is a proper Bayer pattern
    pattern = pattern.upper()
    assert pattern in [
        "RGGB",
        "BGGR",
        "GRBG",
        "GBRG",
    ], "Bayer pattern must be in ['RGGB', 'BGGR', 'GRBG', 'GBRG']"

    r_channel = [i for i, ltr in enumerate(pattern) if ltr == "R"][0]
    b_channel = [i for i, ltr in enumerate(pattern) if ltr == "B"][0]
    g_channels = [i for i, ltr in enumerate(pattern) if ltr == "G"]

    X_rgb = np.empty(X_bayer.shape[:-1] + (3,))
    X_rgb[:, :, 0] = X_bayer[:, :, r_channel]
    X_rgb[:, :, 1] = np.mean(X_bayer[:, :, g_channels], axis=2)
    X_rgb[:, :, 2] = X_bayer[:, :, b_channel]

    return X_rgb
