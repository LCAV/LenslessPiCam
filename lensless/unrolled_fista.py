# #############################################################################
# unrolled_fista.py
# =================
# Authors :
# Yohann PERRON [yohann.perron@gmail.com]
# #############################################################################

import numpy as np
from lensless.trainable_recon import TrainableReconstructionAlgorithm
from lensless.gradient_descent import non_neg
import inspect
from scipy import fft

try:
    import torch

except ImportError:
    raise ImportError("Pytorch is require to use trainable reconstruction algorithms.")


class UnrolledFISTA(TrainableReconstructionAlgorithm):
    """
    Object for applying unrolled projected gradient descent with FISTA (Fast Iterative
    Shrinkage-Thresholding Algorithm) for acceleration.

    FISTA Paper: https://www.ceremade.dauphine.fr/~carlier/FISTA

    """

    def __init__(self, psf, n_iter=5, dtype=None, proj=non_neg, learn_tk=True, tk=1, **kwargs):
        """
        COnstructor for unrolled FISTA algorithm.

        Parameters
        ----------
        psf : :py:class:`~torch.Tensor` of shape (H, W, C)
            The point spread function (PSF) that models forward propagation.
        n_iter : int, optional
            Number of iterations to unrolled, by default 5
        dtype : float32 or float64
            Data type to use for optimization.
        proj : :py:class:`function`, optional
            Projection function to apply at each iteration, by default non_neg
        learn_tk : bool, optional
            whether the tk parameters of FISTA should be learnt, by default True
        tk : int, optional
            Initial value of tk, by default 1
        """

        super(UnrolledFISTA, self).__init__(psf, n_iter=n_iter, dtype=dtype, reset=False, **kwargs)

        self._proj = proj

        # initial guess, half intensity image
        # for online approach could use last reconstruction
        psf_flat = self._psf.reshape(-1, self._n_channels)
        pixel_start = (torch.max(psf_flat, axis=0).values + torch.min(psf_flat, axis=0).values) / 2
        self._image_init = torch.ones_like(self._psf) * pixel_start

        # learnable step size initialize as < 2 / lipschitz
        Hadj_flat = self._convolver._Hadj.reshape(-1, self._n_channels)
        H_flat = self._convolver._H.reshape(-1, self._n_channels)
        self._alpha_p = torch.nn.Parameter(
            torch.ones(n_iter, 3).to(psf.device)
            * (1.8 / torch.max(torch.abs(Hadj_flat * H_flat), axis=0).values)
        )

        # set tk, can be learnt if learn_tk=True
        self._tk_p = [tk]
        for i in range(n_iter):
            self._tk_p.append((1 + np.sqrt(1 + 4 * self._tk_p[i] ** 2)) / 2)
        self._tk_p = torch.Tensor(self._tk_p)
        if learn_tk:
            self._tk_p = torch.nn.Parameter(self._tk_p).to(psf.device)

        self.reset()

    def _form_image(self):
        return self._proj(self._image_est).squeeze()

    def _grad(self):
        diff = self._convolver.convolve(self._image_est) - self._data
        return self._convolver.deconvolve(diff)

    def reset(self, batch_size=1):
        if batch_size == 1:
            self._image_est = self._image_init
            self._xk = self._image_init
        else:
            self._image_est = self._image_init.unsqueeze(0).expand(batch_size, -1, -1, -1)
            self._xk = self._image_est
        self._alpha = torch.abs(self._alpha_p)
        self._tk = torch.abs(self._tk_p)

    def _update(self, iter):
        self._image_est = self._image_est - self._alpha[iter] * self._grad()
        xk = self._proj(self._image_est)
        self._image_est = xk + (self._tk[iter] - 1) / self._tk[iter + 1] * (xk - self._xk)
        self._xk = xk

    def batch_call(self, batch):
        """
        Method for performing iterative reconstruction on a batch of images.
        This implementation is a properly vectorized implementation of FISTA.

        Parameters
        ----------
        batch : :py:class:`~torch.Tensor` of shape (N, C, H, W) or (N, H, W, C)
            The lensless images to reconstruct. If the shape is (N, C, H, W), the images are converted to (N, H, W, C) before reconstruction.

        Returns
        -------
        :py:class:`~torch.Tensor` of shape (N, C, H, W) or (N, H, W, C)
            The reconstructed images. Channels are in the same order as the input.
        """
        self._data = batch
        batch_size = batch.shape[0]

        if self._data.shape[-3] == 3:
            CHW = True
            self._data = self._data.movedim(-3, -1)
        else:
            CHW = False

        self.reset(batch_size)

        for i in range(self.n_iter):
            self._update(i)

        if CHW:
            self._image_est = self._image_est.movedim(-1, -3)
        return self._proj(self._image_est)
