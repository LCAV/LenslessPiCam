import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from lensless.utils.image import resize
from numpy.linalg import multi_dot
from lensless.hardware.mask import CodedAperture
from scipy.linalg import circulant


def make_separable(Y):
    """
    Making the measurement separable
    """
    rowMeans = Y.mean(axis=1, keepdims=True)
    colMeans = Y.mean(axis=0, keepdims=True)
    allMean = rowMeans.mean()
    Ysep = Y - rowMeans - colMeans + allMean
    return Ysep


def rgb2bayer(img):
    """
    Converting RGB image to separated Bayer channels
    """

    # Doubling the size of the image to anticipatie shrinking from Bayer transformation
    height, width, _ = img.shape
    img = resize(img, shape=(height * 2, width * 2, 3))

    # Separating each Bayer channel (blue, green, green, red)
    b = img[::2, ::2, 2]
    gb = img[1::2, ::2, 1]
    gr = img[::2, 1::2, 1]
    r = img[1::2, 1::2, 0]
    img_bayer = np.dstack((b, gb, gr, r))
    img_bayer = make_separable(img_bayer)

    return img_bayer


def bayer2rgb(X_bayer, normalize=True):
    """
    Converting 4-channel Bayer image to RGB
    """
    X_rgb = np.empty(X_bayer.shape[:-1] + (3,))
    X_rgb[:, :, 0] = X_bayer[:, :, 0]
    X_rgb[:, :, 1] = 0.5 * (X_bayer[:, :, 1] + X_bayer[:, :, 2])
    X_rgb[:, :, 2] = X_bayer[:, :, 3]
    # normalize to be from 0 to 1
    if normalize:
        X_rgb = (X_rgb - X_rgb.min()) / (X_rgb.max() - X_rgb.min())
    return X_rgb


def simulation(img, mask, rgb=True):
    """
    Simulation function
    """
    img_bayer = rgb2bayer(img)
    P = circulant(np.resize(mask.col, mask.mask.shape[0]))[:, : img.shape[0] // 2]
    Q = circulant(np.resize(mask.row, mask.mask.shape[1]))[:, : img.shape[1] // 2]
    Y_bayer = np.dstack([multi_dot([P, img_bayer[:, :, c], Q.T]) for c in range(4)])

    if rgb:
        return bayer2rgb(Y_bayer)
    else:
        return Y_bayer


def fcrecon(Y, P, Q, lmbd):
    """
    Reconstruction algorithm
    """

    # Empty Bayer reconstruction
    X_bayer = np.empty([P.shape[1], Q.shape[1], 4])

    # Applying reconstruction for each Bayer channel
    for c in range(4):

        # SVD of left matrix
        UL, SL, VLh = np.linalg.svd(P, full_matrices=True)
        VL = VLh.T
        DL = np.concatenate((np.diag(SL), np.zeros([P.shape[0] - SL.size, SL.size])))
        singLsq = np.square(SL)

        # SVD of right matrix
        UR, SR, VRh = np.linalg.svd(Q, full_matrices=True)
        VR = VRh.T
        DR = np.concatenate((np.diag(SR), np.zeros([Q.shape[0] - SR.size, SR.size])))
        singRsq = np.square(SR)

        # Applying analytical reconstruction
        Yc = Y[:, :, c]
        inner = multi_dot([DL.T, UL.T, Yc, UR, DR]) / (
            np.outer(singLsq, singRsq) + np.full(X_bayer.shape[0:2], lmbd)
        )
        X_bayer[:, :, c] = multi_dot([VL, inner, VR.T])

    X_bayer = X_bayer.clip(min=0)  # non-negative constraint: set all negative values to 0

    return bayer2rgb(X_bayer, True)  # bring back to RGB and normalize
