# #############################################################################
# gradient_descent.py
# =================
# Authors :
# Eric BEZZAM [ebezzam@gmail.com]
# Julien SAHLI [julien.sahli@epfl.ch]
# #############################################################################


import numpy as np
from lensless.recon.recon import ReconstructionAlgorithm
import inspect
from lensless.utils.io import load_data
import time

try:
    import torch

    torch_available = True
except ImportError:
    torch_available = False


class GradientDescentUpdate:
    """Gradient descent update techniques."""

    VANILLA = "vanilla"
    NESTEROV = "nesterov"
    FISTA = "fista"

    @staticmethod
    def all_values():
        vals = []
        for i in inspect.getmembers(GradientDescentUpdate):
            # remove private and protected functions, and this function
            if not i[0].startswith("_") and not callable(i[1]):
                vals.append(i[1])
        return vals


def non_neg(xi):
    """
    Clip input so that it is non-negative.

    Parameters
    ----------
    xi : :py:class:`~numpy.ndarray`
        Data to clip.

    Returns
    -------
    nonneg : :py:class:`~numpy.ndarray`
        Non-negative projection of input.

    """
    if torch_available and isinstance(xi, torch.Tensor):
        return torch.maximum(xi, torch.zeros_like(xi))
    else:
        return np.maximum(xi, 0)


class GradientDescent(ReconstructionAlgorithm):
    """
    Object for applying projected gradient descent.
    """

    def __init__(self, psf, dtype=None, proj=non_neg, lip_fact=1.8, **kwargs):
        """

        Parameters
        ----------
        psf : :py:class:`~numpy.ndarray` or :py:class:`~torch.Tensor`
            Point spread function (PSF) that models forward propagation.
            Must be of shape (depth, height, width, channels) even if
            depth = 1 and channels = 1. You can use :py:func:`~lensless.io.load_psf`
            to load a PSF from a file such that it is in the correct format.
        dtype : float32 or float64
            Data type to use for optimization. Default is float32.
        proj : :py:class:`function`
            Projection function to apply at each iteration. Default is
            non-negative.
        """

        assert callable(proj)
        self._proj = proj
        self._lip_fact = lip_fact
        super(GradientDescent, self).__init__(psf, dtype, **kwargs)

        if self._denoiser is not None:
            print("Using denoiser in gradient descent.")
            # redefine projection function
            self._proj = self._denoiser

    def reset(self):
        if self.is_torch:
            if self._initial_est is not None:
                self._image_est = self._initial_est
            else:
                # initial guess, half intensity image
                psf_flat = self._psf.reshape(-1, self._psf_shape[3])
                pixel_start = (
                    torch.max(psf_flat, axis=0).values + torch.min(psf_flat, axis=0).values
                ) / 2
                # initialize image estimate as [Batch, Depth, Height, Width, Channels]
                self._image_est = torch.ones_like(self._psf[None, ...]) * pixel_start

            # set step size as < 2 / lipschitz
            Hadj_flat = self._convolver._Hadj.reshape(-1, self._psf_shape[3])
            H_flat = self._convolver._H.reshape(-1, self._psf_shape[3])
            self._alpha = torch.real(
                self._lip_fact / torch.max(torch.abs(Hadj_flat * H_flat), axis=0).values
            )

        else:
            if self._initial_est is not None:
                self._image_est = self._initial_est
            else:
                psf_flat = self._psf.reshape(-1, self._psf_shape[3])
                pixel_start = (np.max(psf_flat, axis=0) + np.min(psf_flat, axis=0)) / 2
                # initialize image estimate as [Batch, Depth, Height, Width, Channels]
                self._image_est = np.ones_like(self._psf[None, ...]) * pixel_start

            # set step size as < 2 / lipschitz
            Hadj_flat = self._convolver._Hadj.reshape(-1, self._psf_shape[3])
            H_flat = self._convolver._H.reshape(-1, self._psf_shape[3])
            self._alpha = np.real(self._lip_fact / np.max(Hadj_flat * H_flat, axis=0))

    def _grad(self):
        diff = self._convolver.convolve(self._image_est) - self._data
        return self._convolver.deconvolve(diff)

    def _update(self, iter):
        self._image_est -= self._alpha * self._grad()
        self._image_est = self._form_image()

    def _form_image(self):
        if self._denoiser is not None:
            return self._proj(self._image_est, self._denoiser_noise_level)
        else:
            return self._proj(self._image_est)


class NesterovGradientDescent(GradientDescent):
    """
    Object for applying projected gradient descent with Nesterov momentum for
    acceleration.

    A nice tutorial/ blog post on Nesterov momentum can be found
    `here <https://machinelearningmastery.com/gradient-descent-with-nesterov-momentum-from-scratch/>`_.

    """

    def __init__(self, psf, dtype=None, proj=non_neg, p=0, mu=0.9, **kwargs):
        """

        Parameters
        ----------
        psf : :py:class:`~numpy.ndarray` or :py:class:`~torch.Tensor`
            Point spread function (PSF) that models forward propagation.
            Must be of shape (depth, height, width, channels) even if
            depth = 1 and channels = 1. You can use :py:func:`~lensless.io.load_psf`
            to load a PSF from a file such that it is in the correct format.
        dtype : float32 or float64
            Data type to use for optimization. Default is float32.
        proj : :py:class:`function`
            Projection function to apply at each iteration. Default is
            non-negative.
        p : float
            Momentum parameter that keeps track of changes. By default, this
            is initialized to 0.
        mu : float
            Momentum parameter. Default is 0.9.
        """
        self._p = p
        self._mu = mu
        super(NesterovGradientDescent, self).__init__(psf, dtype, proj, **kwargs)

    def reset(self, p=0, mu=0.9):
        self._p = p
        self._mu = mu
        super(NesterovGradientDescent, self).reset()

    def _update(self, iter):
        p_prev = self._p
        self._p = self._mu * self._p - self._alpha * self._grad()
        self._image_est += -self._mu * p_prev + (1 + self._mu) * self._p
        # self._image_est = self._proj(self._image_est)
        self._image_est = self._form_image()


class FISTA(GradientDescent):
    """
    Object for applying projected gradient descent with FISTA (Fast Iterative
    Shrinkage-Thresholding Algorithm) for acceleration.

    Paper: https://www.ceremade.dauphine.fr/~carlier/FISTA

    """

    def __init__(self, psf, dtype=None, proj=non_neg, tk=1.0, **kwargs):
        """

        Parameters
        ----------
        psf : :py:class:`~numpy.ndarray` or :py:class:`~torch.Tensor`
            Point spread function (PSF) that models forward propagation.
            Must be of shape (depth, height, width, channels) even if
            depth = 1 and channels = 1. You can use :py:func:`~lensless.io.load_psf`
            to load a PSF from a file such that it is in the correct format.
        dtype : float32 or float64
            Data type to use for optimization. Default is float32.
        proj : :py:class:`function`
            Projection function to apply at each iteration. Default is
            non-negative.
        tk : float
            Initial step size parameter for FISTA. It is updated at each iteration
            according to Eq. 4.2 of paper. By default, initialized to 1.0.

        """
        self._initial_tk = tk

        super(FISTA, self).__init__(psf, dtype, proj, **kwargs)

        self._tk = tk
        self._xk = self._image_est

    def reset(self, tk=None):
        super(FISTA, self).reset()
        if tk:
            self._tk = tk
        else:
            self._tk = self._initial_tk
        self._xk = self._image_est

    def _update(self, iter):
        self._image_est -= self._alpha * self._grad()
        xk = self._form_image()
        tk = (1 + np.sqrt(1 + 4 * self._tk**2)) / 2
        self._image_est = xk + (self._tk - 1) / tk * (xk - self._xk)
        self._tk = tk
        self._xk = xk


def apply_gradient_descent(psf_fp, data_fp, n_iter, verbose=False, proj=non_neg, **kwargs):

    # load data
    psf, data = load_data(psf_fp=psf_fp, data_fp=data_fp, plot=False, **kwargs)

    # create reconstruction object
    recon = GradientDescent(psf, n_iter=n_iter, proj=proj)

    # set data
    recon.set_data(data)

    # perform reconstruction
    start_time = time.time()
    res = recon.apply(plot=False)
    proc_time = time.time() - start_time

    if verbose:
        print(f"Reconstruction time : {proc_time} s")
        print(f"Reconstruction shape: {res.shape}")
    return res
