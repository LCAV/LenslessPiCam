import numpy as np
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
import lpips as lpips_lib
import torch


def mse(true, est, normalize=True):
    if normalize:
        true /= true.max()
        est /= est.max()
    return mean_squared_error(image0=true, image1=est)


def psnr(true, est, normalize=True):
    if normalize:
        true /= true.max()
        est /= est.max()
    return peak_signal_noise_ratio(image_true=true, image_test=est)


def ssim(true, est, normalize=True):
    if normalize:
        true /= true.max()
        est /= est.max()
    return structural_similarity(im1=true, im2=est, channel_axis=2)


def lpips(true, est, normalize=True):
    # https://github.com/richzhang/PerceptualSimilarity
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
