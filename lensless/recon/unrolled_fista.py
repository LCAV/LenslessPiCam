# #############################################################################
# unrolled_fista.py
# =================
# Authors :
# Yohann PERRON [yohann.perron@gmail.com]
# #############################################################################

import numpy as np
from lensless.recon.trainable_recon import TrainableReconstructionAlgorithm
from lensless.recon.gd import non_neg

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
        psf : :py:class:`~torch.Tensor`
            Point spread function (PSF) that models forward propagation.
            Must be of shape (depth, height, width, channels) even if
            depth = 1 and channels = 1. You can use :py:func:`~lensless.io.load_psf`
            to load a PSF from a file such that it is in the correct format.
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
        psf_flat = self._psf.reshape(-1, self._psf_shape[3])
        pixel_start = (torch.max(psf_flat, axis=0).values + torch.min(psf_flat, axis=0).values) / 2
        self._image_init = torch.ones_like(self._psf[None, ...]) * pixel_start

        # learnable step size initialize as < 2 / lipschitz
        Hadj_flat = self._convolver._Hadj.reshape(-1, self._psf_shape[3])
        H_flat = self._convolver._H.reshape(-1, self._psf_shape[3])
        if not self.skip_unrolled:
            self._alpha_p = torch.nn.Parameter(
                torch.ones(self._n_iter, self._psf_shape[3]).to(psf.device)
                * (1.8 / torch.max(torch.abs(Hadj_flat * H_flat), axis=0).values)
            )
        else:
            self._alpha_p = torch.ones(self._n_iter, self._psf_shape[3]).to(psf.device) * (
                1.8 / torch.max(torch.abs(Hadj_flat * H_flat), axis=0).values
            )

        # set tk, can be learnt if learn_tk=True
        self._tk_p = [tk]
        for i in range(self._n_iter):
            self._tk_p.append((1 + np.sqrt(1 + 4 * self._tk_p[i] ** 2)) / 2)
        self._tk_p = torch.Tensor(self._tk_p)
        if learn_tk and not self.skip_unrolled:
            self._tk_p = torch.nn.Parameter(self._tk_p).to(psf.device)

        self.reset()

    def _form_image(self):
        return self._proj(self._image_est)

    def _grad(self):
        diff = self._convolver.convolve(self._image_est) - self._data
        return self._convolver.deconvolve(diff)

    def reset(self, batch_size=1):
        if self._initial_est is not None:
            self._image_est = self._initial_est
        else:
            self._image_est = self._image_init.expand(batch_size, -1, -1, -1, -1)
        self._xk = self._image_est

        # enforce positivity
        self._alpha = torch.abs(self._alpha_p)
        self._tk = torch.abs(self._tk_p)

    def _update(self, iter):
        self._image_est = self._image_est - self._alpha[iter] * self._grad()
        xk = self._proj(self._image_est)
        self._image_est = xk + (self._tk[iter] - 1) / self._tk[iter + 1] * (xk - self._xk)
        self._xk = xk
