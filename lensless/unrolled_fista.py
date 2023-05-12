# #############################################################################
# trainable_recon.py
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


class unrolled_FISTA(TrainableReconstructionAlgorithm):
    """
    Object for applying projected gradient descent with FISTA (Fast Iterative
    Shrinkage-Thresholding Algorithm) for acceleration.

    Paper: https://www.ceremade.dauphine.fr/~carlier/FISTA

    """

    def __init__(self, psf, n_iter=5, dtype=None, proj=non_neg, learn_tk=True, tk=1, **kwargs):

        super(unrolled_FISTA, self).__init__(psf, n_iter=n_iter, dtype=dtype, **kwargs)

        self._proj = proj

        # initial guess, half intensity image
        # for online approach could use last reconstruction
        psf_flat = self._psf.reshape(-1, self._n_channels)
        pixel_start = (torch.max(psf_flat, axis=0).values + torch.min(psf_flat, axis=0).values) / 2
        self._image_init = torch.ones_like(self._psf) * pixel_start
        self._image_est = self._image_init
        self._xk = self._image_init

        # learnable step size initialize as < 2 / lipschitz
        Hadj_flat = self._convolver._Hadj.reshape(-1, self._n_channels)
        H_flat = self._convolver._H.reshape(-1, self._n_channels)
        self._alpha = torch.nn.Parameter(
            torch.ones(n_iter, 3).to(psf.device)
            * (1.8 / torch.max(torch.abs(Hadj_flat * H_flat), axis=0).values)
        )

        # set tk, can be learnt if learn_tk=True
        self._tk = [tk]
        for i in range(n_iter):
            self._tk.append((1 + np.sqrt(1 + 4 * self._tk[i] ** 2)) / 2)
        self._tk = torch.Tensor(self._tk)
        if learn_tk:
            self._tk = torch.nn.Parameter(self._tk)

    def _form_image(self):
        return self._proj(self._image_est).squeeze()

    def _grad(self):
        diff = self._convolver.convolve(self._image_est) - self._data
        return self._convolver.deconvolve(diff)

    def reset(self):
        # needed because ReconstructionAlgorithm initializer call reset to early
        if hasattr(self, "_image_init"):
            self._image_est = self._image_init
            self._xk = self._image_init

    def _update(self, iter):
        self._image_est = self._image_est - self._alpha[iter] * self._grad()
        xk = self._proj(self._image_est)
        self._image_est = xk + (self._tk[iter] - 1) / self._tk[iter + 1] * (xk - self._xk)
        self._xk = xk

    def batch_call(self, batch):
        self._data = batch
        batch_size = batch.shape[0]

        if self._data.shape[-3] == 3:
            CHW = True
            self._data = self._data.movedim(-3, -1)
        else:
            CHW = False

        self._image_est = self._image_init.unsqueeze(0).expand(batch_size, -1, -1, -1)
        self._xk = self._image_est

        for i in range(self.n_iter):
            self._update(i)

        if CHW:
            self._image_est = self._image_est.movedim(-1, -3)
        return self._proj(self._image_est)
