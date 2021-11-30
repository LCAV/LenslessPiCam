import cv2
import csv
import numpy as np
from diffcam.constants import RPI_HQ_CAMERA_CCM_MATRIX, RPI_HQ_CAMERA_BLACK_LEVEL


SUPPORTED_BIT_DEPTH = np.array([8, 10, 12, 16])
FLOAT_DTYPES = [np.float32, np.float64]


def resize(img, factor, interpolation=cv2.INTER_CUBIC):
    """
    Resize by given factor.

    Parameters
    ----------
    img :py:class:`~numpy.ndarray`
        Downsampled image.
    factor : int or float
        Resizing factor.
    interpolation : OpenCV interpolation method
        See https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#cv2.resize

    Returns
    -------
    img :py:class:`~numpy.ndarray`
        Resized image.
    """
    min_val = img.min()
    max_val = img.max()
    # new_shape = tuple((np.array(img.shape)[::-1] * factor).astype(int))
    new_shape = tuple((np.array(img.shape)[:2] * factor).astype(int))
    new_shape = new_shape[::-1]
    resized = cv2.resize(img, dsize=new_shape, interpolation=interpolation)
    return np.clip(resized, min_val, max_val)


def rgb2gray(rgb, weights=None):
    """
    Convert RGB array to grayscale.

    Parameters
    ----------
    rgb : :py:class:`~numpy.ndarray`
        (N_height, N_width, N_channel) image.
    weights : :py:class:`~numpy.ndarray`
        [Optional] (3,) weights to convert from RGB to grayscale.

    Returns
    -------
    img :py:class:`~numpy.ndarray`
        Grayscale image of dimension (height, width).

    """
    if weights is None:
        weights = np.array([0.299, 0.587, 0.144])
    assert len(weights) == 3
    return np.tensordot(rgb, weights, axes=((2,), 0))


def gamma_correction(vals, gamma=2.2):
    """
    Tutorials
    - https://www.cambridgeincolour.com/tutorials/gamma-correction.htm
    - (code, for images) https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
        https://lindevs.com/apply-gamma-correction-to-an-image-using-opencv/
    - (code) http://www.fourmilab.ch/documents/specrend/specrend.c

    Parameters
    ----------
    vals : array_like
        RGB values to gamma correct.

    Returns
    -------

    """

    # simple approach
    # return np.power(vals, 1 / gamma)

    # Rec. 709 gamma correction
    # http://www.fourmilab.ch/documents/specrend/specrend.c
    cc = 0.018
    inv_gam = 1 / gamma
    clip_val = (1.099 * np.power(cc, inv_gam) - 0.099) / cc
    return np.where(vals < cc, vals * clip_val, 1.099 * np.power(vals, inv_gam) - 0.099)


def get_max_val(img, nbits=None):
    """For uint image"""
    assert img.dtype not in FLOAT_DTYPES
    if nbits is None:
        nbits = int(np.ceil(np.log2(img.max())))

    if nbits not in SUPPORTED_BIT_DEPTH:
        nbits = SUPPORTED_BIT_DEPTH[nbits < SUPPORTED_BIT_DEPTH][0]
    max_val = 2 ** nbits - 1
    if img.max() > max_val:
        new_nbit = int(np.ceil(np.log2(img.max())))
        print(f"Detected pixel value larger than {nbits}-bit range, using {new_nbit}-bit range.")
        max_val = 2 ** new_nbit - 1
    return max_val


def bayer2rgb(
    img,
    nbits,
    bg=None,
    rg=None,
    black_level=RPI_HQ_CAMERA_BLACK_LEVEL,
    ccm=RPI_HQ_CAMERA_CCM_MATRIX,
    nbits_out=None,
):
    """
    Convert raw Bayer data to RGB with the following steps:
    - Demosaic with bilinear interpolation, mapping the Bayer array to RGB.
    - Black level removal.
    - White balancing, applying gains to red and blue channels.
    - Color correction matrix.
    - Clip

    :param img:
    :param nbits:
    :param bg:
    :param rg:
    :param black_level:
    :param ccm:
    :return:
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

    # correction
    img = img - black_level
    if rg:
        img[:, :, 0] *= rg
    if bg:
        img[:, :, 2] *= bg
    img = img / (2 ** nbits - 1 - black_level)
    img[img > 1] = 1
    img = (img.reshape(-1, 3, order="F") @ ccm.T).reshape(img.shape, order="F")
    img[img < 0] = 0
    img[img > 1] = 1
    return (img * (2 ** nbits_out - 1)).astype(dtype)


def get_distro():
    # https://majornetwork.net/2019/11/get-linux-distribution-name-and-version-with-python/
    RELEASE_DATA = {}
    with open("/etc/os-release") as f:
        reader = csv.reader(f, delimiter="=")
        for row in reader:
            if row:
                RELEASE_DATA[row[0]] = row[1]
    if RELEASE_DATA["ID"] in ["debian", "raspbian"]:
        with open("/etc/debian_version") as f:
            DEBIAN_VERSION = f.readline().strip()
        major_version = DEBIAN_VERSION.split(".")[0]
        version_split = RELEASE_DATA["VERSION"].split(" ", maxsplit=1)
        if version_split[0] == major_version:
            # Just major version shown, replace it with the full version
            RELEASE_DATA["VERSION"] = " ".join([DEBIAN_VERSION] + version_split[1:])
    return f"{RELEASE_DATA['NAME']} {RELEASE_DATA['VERSION']}"


def print_image_info(img):
    print("dimensions : {}".format(img.shape))
    print("data type : {}".format(img.dtype))
    print("max  : {}".format(img.max()))
    print("min  : {}".format(img.min()))
    print("mean : {}".format(img.mean()))
