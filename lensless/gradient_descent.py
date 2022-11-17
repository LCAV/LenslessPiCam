import numpy as np
from lensless.recon import ReconstructionAlgorithm
import inspect
from scipy import fft


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
    xi = np.maximum(xi, 0)
    return xi


class GradientDescient(ReconstructionAlgorithm):
    def __init__(self, psf, dtype=np.float32, proj=non_neg, **kwargs):
        """
        Object for applying projected gradient descent.

        Parameters
        ----------
        psf : :py:class:`~numpy.ndarray`
            3D Point spread function (PSF) that models forward propagation.
            3D (grayscale) or 4D (RGB) data can be provided and the shape will
            be used to determine which reconstruction (and allocate the
            appropriate memory).
        dtype : float32 or float64
            Data type to use for optimization.
        proj : :py:class:`function`
            Projection function to apply at each iteration. Default is
            non-negative.
        """

        super(GradientDescient, self).__init__(psf, dtype)
        assert callable(proj)
        self._proj = proj

    def _crop(self, x):
        return x[:, self._start_idx[0] : self._end_idx[0], self._start_idx[1] : self._end_idx[1], :]

    def _pad(self, v):
        vpad = np.zeros(self._padded_shape).astype(v.dtype)
        for i in range(vpad.shape[0]):
            vpad[
                i, self._start_idx[0] : self._end_idx[0], self._start_idx[1] : self._end_idx[1]
            ] = v[i, :, :]
        return vpad

    def reset(self):
        # initial guess, half intensity image
        # for online approach could use last reconstruction
        psf_flat = self._psf.reshape(-1, self._psf_shape[3])
        pixel_start = (np.max(psf_flat, axis=0) + np.min(psf_flat, axis=0)) / 2
        x = np.ones(self._psf_shape, dtype=self._dtype) * pixel_start
        self._image_est = self._pad(x)
        # spatial frequency response
        self._H = fft.rfft2(
            self._pad(self._psf), norm="ortho", axes=(0, 1, 2), s=self._padded_shape[:3]
        )
        self._Hadj = np.conj(self._H)

        Hadj_flat = self._Hadj.reshape(-1, self._psf_shape[3])
        H_flat = self._H.reshape(-1, self._psf_shape[3])
        self._alpha = np.real(1.8 / np.max(Hadj_flat * H_flat, axis=0))

    def _grad(self):
        diff = self._forward() - self._data
        return self._backward(diff)

    def _forward(self):
        Vk = fft.rfft2(self._image_est, axes=(0, 1, 2), s=self._padded_shape[:3])
        return self._crop(
            fft.ifftshift(
                fft.irfft2(self._H * Vk, axes=(0, 1, 2), s=self._padded_shape[:3]), axes=(0, 1, 2)
            )
        )

    def _backward(self, x):
        X = fft.rfft2(self._pad(x), axes=(0, 1, 2), s=self._padded_shape[:3])
        return fft.ifftshift(
            fft.irfft2(self._Hadj * X, axes=(0, 1, 2), s=self._padded_shape[:3]), axes=(0, 1, 2)
        )

    def _update(self):
        self._image_est -= self._alpha * self._grad()
        self._image_est = self._proj(self._image_est)

    def _form_image(self):
        return self._proj(self._crop(self._image_est)).squeeze()


class NesterovGradientDescent(GradientDescient):
    """
    Object for applying projected gradient descent with Nesterov momentum for
    acceleration.

    Tutorial on Nesterov momentum: https://machinelearningmastery.com/gradient-descent-with-nesterov-momentum-from-scratch/

    """

    def __init__(self, psf, dtype=np.float32, proj=non_neg, p=0, mu=0.9, **kwargs):
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


class FISTA(GradientDescient):
    """
    Object for applying projected gradient descent with FISTA (Fast Iterative
    Shrinkage-Thresholding Algorithm) for acceleration.

    Paper: https://www.ceremade.dauphine.fr/~carlier/FISTA

    """

    def __init__(self, psf, dtype=np.float32, proj=non_neg, tk=1, **kwargs):
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
