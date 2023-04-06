import numpy as np
from lensless.recon import ReconstructionAlgorithm
import inspect
import warnings

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
    def __init__(self, psf, dtype=None, proj=non_neg, initial_est=None, **kwargs):
        """
        Object for applying projected gradient descent.

        Parameters
        ----------
        psf : :py:class:`~numpy.ndarray` or :py:class:`~torch.Tensor`
            Point spread function (PSF) that models forward propagation.
            2D (grayscale) or 3D (RGB) data can be provided and the shape will
            be used to determine which reconstruction (and allocate the
            appropriate memory).
        dtype : float32 or float64
            Data type to use for optimization. Default is float32.
        proj : :py:class:`function`
            Projection function to apply at each iteration. Default is
            non-negative.
        initial_est : :py:class:`~numpy.ndarray`
            Initial estimation for the reconstruction. If none, will create
            a plain gray estimation.
        """
        self._initial_est = initial_est
        super(GradientDescent, self).__init__(psf, dtype)
        assert callable(proj)
        self._proj = proj

    def reset(self):

        if self.is_torch:
            if self._initial_est is not None:
                warnings.warn("Error : use of initial estimations is not supported with Pytorch yet")
            # initial guess, half intensity image
            # for online approach could use last reconstruction
            psf_flat = self._psf.reshape(-1, self._psf_shape[3])
            pixel_start = (
                torch.max(psf_flat, axis=0).values + torch.min(psf_flat, axis=0).values
            ) / 2
            self._image_est = torch.ones_like(self._psf) * pixel_start

            # set step size as < 2 / lipschitz
            Hadj_flat = self._convolver._Hadj.reshape(-1, self._psf_shape[3])
            H_flat = self._convolver._H.reshape(-1, self._psf_shape[3])
            self._alpha = torch.real(1.8 / torch.max(torch.abs(Hadj_flat * H_flat), axis=0).values)

        else:
            # try to determine the type of the inital estimation, if provided
            if self._initial_est is not None:
                if len(self._initial_est.shape) == 3:  # if input we gave is under-dimensioned, determine its type
                    if self._initial_est.shape[0] == 1:
                        self._image_est = np.array(self._initial_est[:, :, :, np.newaxis])
                    elif self._initial_est.shape[2] == 3:  # 2d rgb
                        self._image_est = np.array(self._initial_est[np.newaxis, :, :, :])
                    else:
                        warnings.warn("Error : could not infer data type for initial estimation")
                        self._image_est = None
                elif len(self._initial_est.shape) == 2:  # grayscale 2d
                    self._image_est = np.array(self._initial_est[np.newaxis, :, :, np.newaxis])

            # if no initial estimation is provided, will use half intensity image
            if self._image_est is None:
                self._image_est = np.ones(self._psf_shape, dtype=self._dtype)
            else:
                assert self._initial_est.shape == self._psf_shape

            psf_flat = self._psf.reshape(-1, self._psf_shape[3])
            pixel_start = (np.max(psf_flat, axis=0) + np.min(psf_flat, axis=0)) / 2
            self._image_est *= pixel_start

            # set step size as < 2 / lipschitz
            Hadj_flat = self._convolver._Hadj.reshape(-1, self._psf_shape[3])
            H_flat = self._convolver._H.reshape(-1, self._psf_shape[3])
            self._alpha = np.real(1.8 / np.max(Hadj_flat * H_flat, axis=0))

    def _grad(self):
        diff = self._convolver.convolve(self._image_est) - self._data
        return self._convolver.deconvolve(diff)

    def _update(self):
        self._image_est -= self._alpha * self._grad()
        self._image_est = self._proj(self._image_est)

    def _form_image(self):
        return self._proj(self._image_est).squeeze()


class NesterovGradientDescent(GradientDescent):
    """
    Object for applying projected gradient descent with Nesterov momentum for
    acceleration.

    Tutorial on Nesterov momentum: https://machinelearningmastery.com/gradient-descent-with-nesterov-momentum-from-scratch/

    """

    def __init__(self, psf, dtype=None, proj=non_neg, p=0, mu=0.9, **kwargs):
        self._p = p
        self._mu = mu
        super(NesterovGradientDescent, self).__init__(psf, dtype, proj)

    def reset(self, p=0, mu=0.9):
        self._p = p
        self._mu = mu
        super(NesterovGradientDescent, self).reset()

    def _update(self):
        p_prev = self._p
        self._p = self._mu * self._p - self._alpha * self._grad()
        self._image_est += -self._mu * p_prev + (1 + self._mu) * self._p
        self._image_est = self._proj(self._image_est)


class FISTA(GradientDescent):
    """
    Object for applying projected gradient descent with FISTA (Fast Iterative
    Shrinkage-Thresholding Algorithm) for acceleration.

    Paper: https://www.ceremade.dauphine.fr/~carlier/FISTA

    """

    def __init__(self, psf, dtype=None, proj=non_neg, tk=1, **kwargs):

        super(FISTA, self).__init__(psf, dtype, proj)
        self._tk = tk
        self._xk = self._image_est

    def reset(self, tk=1):
        super(FISTA, self).reset()
        self._tk = tk
        self._xk = self._image_est

    def _update(self):
        self._image_est -= self._alpha * self._grad()
        xk = self._proj(self._image_est)
        tk = (1 + np.sqrt(1 + 4 * self._tk**2)) / 2
        self._image_est = xk + (self._tk - 1) / tk * (xk - self._xk)
        self._tk = tk
        self._xk = xk
