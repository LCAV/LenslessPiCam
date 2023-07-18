# #############################################################################
# tikhonov.py
# =================
# Authors :
# Aaron FARGEON [aa.fargeon@gmail.com]
# #############################################################################


import numpy as np
from lensless.utils.image import resize
from numpy.linalg import multi_dot
from scipy.linalg import circulant


class CodedApertureReconstruction:
    """
    ddd
    """

    def __init__(self, mask, image_shape, lmbd=3e-4):
        """
        ddd
        """

        self.lmbd = lmbd
        self.P = circulant(np.resize(mask.col, mask.sensor_resolution[0]))[:, : image_shape[0]]
        self.Q = circulant(np.resize(mask.row, mask.sensor_resolution[1]))[:, : image_shape[1]]

    def apply(self, img, color_profile="rgb"):
        """
        ddd
        """

        # Squeezing the image to get rid of extra dimensions
        Y = img.squeeze()

        # Verifying
        assert (
            Y.shape[0] == self.P.shape[1]
        ), f"image height of {Y.shape[0]} does not match the expected size of {self.P.shape[1]}"
        assert (
            Y.shape[1] == self.Q.shape[1]
        ), f"image height of {Y.shape[1]} does not match the expected size of {self.Q.shape[1]}"

        color_profile = color_profile.lower()
        assert color_profile in [
            "grayscale",
            "rgb",
            "bayer_bggr",
        ], "color_profile must be in ['grayscale', 'rgb', 'bayer_bggr]"
        if color_profile == "grayscale":
            assert (
                len(Y.shape) == 2
            ), "for a grayscale image, the squeezed image needs to be in 2 dimensions"
            Y = Y[:, :, np.newaxis]
        elif color_profile == "rgb":
            assert (
                len(Y.shape) == 3
            ), "for an RGB image, the squeezed image needs to be 3 dimensions"
            assert Y.shape[-1] == 3, "for an RGB image, the squeezed image should have 3 channels"
        else:
            assert (
                len(Y.shape) == 3
            ), "for a Bayer image, the squeezed image needs to be 3 dimensions"
            assert Y.shape[-1] == 4, "for a Bayer image, the squeezed image should have 4 channels"

        # Empty matrix for reconstruction
        X = np.empty([self.P.shape[1], self.Q.shape[1], Y.shape[-1]])

        # Applying reconstruction for each channel
        for c in range(Y.shape[-1]):

            # SVD of left matrix
            UL, SL, VLh = np.linalg.svd(self.P, full_matrices=True)
            VL = VLh.T
            DL = np.concatenate((np.diag(SL), np.zeros([self.P.shape[0] - SL.size, SL.size])))
            singLsq = np.square(SL)

            # SVD of right matrix
            UR, SR, VRh = np.linalg.svd(self.Q, full_matrices=True)
            VR = VRh.T
            DR = np.concatenate((np.diag(SR), np.zeros([self.Q.shape[0] - SR.size, SR.size])))
            singRsq = np.square(SR)

            # Applying analytical reconstruction
            Yc = Y[:, :, c]
            inner = multi_dot([DL.T, UL.T, Yc, UR, DR]) / (
                np.outer(singLsq, singRsq) + np.full(X.shape[0:2], self.lmbd)
            )
            X[:, :, c] = multi_dot([VL, inner, VR.T])

        # Non-negativity constraint: setting all negative values to 0
        X = X.clip(min=0)

        return X


def rgb2bayer(img, pattern):
    """
    Converting RGB image to separated Bayer channels
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

    elif pattern == "BGGR":
        # RGGB pattern *------*
        #              | R  G |
        #              | G  B |
        #              *------*
        r = resized[1::2, 1::2, 0]
        gr = resized[::2, 1::2, 1]
        gb = resized[1::2, ::2, 1]
        b = resized[::2, ::2, 2]

    elif pattern == "GBRG":
        # GRGB pattern *------*
        #              | G  R |
        #              | B  G |
        #              *------*
        r = resized[1::2, ::2, 0]
        gr = resized[::2, ::2, 1]
        gb = resized[1::2, 1::2, 1]
        b = resized[::2, 1::2, 2]

    else:
        # GBRG pattern *------*
        #              | G  B |
        #              | R  G |
        #              *------*
        r = resized[::2, 1::2, 0]
        gr = resized[1::2, 1::2, 1]
        gb = resized[::2, ::2, 1]
        b = resized[1::2, ::2, 2]

    # Stacking the Bayer channels, always in the same order s.t. bayer2rgb() works regardless of the pattern
    img_bayer = np.dstack((r, gr, gb, b))

    return img_bayer


def bayer2rgb(X_bayer, normalize=True):
    """
    Converting 4-channel Bayer image to RGB
    """
    X_rgb = np.empty(X_bayer.shape[:-1] + (3,))
    X_rgb[:, :, 2] = X_bayer[:, :, 0]
    X_rgb[:, :, 1] = 0.5 * (X_bayer[:, :, 1] + X_bayer[:, :, 2])
    X_rgb[:, :, 0] = X_bayer[:, :, 3]
    # normalize to be from 0 to 1
    if normalize:
        X_rgb = (X_rgb - X_rgb.min()) / (X_rgb.max() - X_rgb.min())
    return X_rgb
