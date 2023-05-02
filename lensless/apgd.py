from lensless.recon import ReconstructionAlgorithm
from lensless.rfft_convolve import RealFFTConvolve2D as Convolver
import inspect
import numpy as np
from pycsou.func.loss import SquaredL2Loss
from pycsou.func.penalty import NonNegativeOrthant, SquaredL2Norm, L1Norm
from pycsou.opt.proxalgs import APGD as APGD_pyc
from copy import deepcopy
from pycsou.core.linop import LinearOperator
from typing import Union, Optional
from numbers import Number


class APGDPriors:
    """
    Priors (compatible with Pycsou) for APGD.

    See Pycsou documentation for available penalties:
    https://matthieumeo.github.io/pycsou/html/api/functionals/pycsou.func.penalty.html?highlight=penalty#module-pycsou.func.penalty
    """

    L2 = "l2"
    NONNEG = "nonneg"
    L1 = "l1"

    @staticmethod
    def all_values():
        vals = []
        for i in inspect.getmembers(APGDPriors):
            # remove private and protected functions, and this function
            if not i[0].startswith("_") and not callable(i[1]):
                vals.append(i[1])
        return vals


class RealFFTConvolve2D(LinearOperator):
    def __init__(self, filter, dtype: Optional[type] = None, norm: str = "ortho", **kwargs):
        """
        Linear operator that performs convolution in Fourier domain, and assumes
        real-valued signals.

        Parameters
        ----------
        filter :py:class:`~numpy.ndarray`
            2D filter to use. Must be of shape (height, width, channels) even if
            only one channel.
        dtype : float32 or float64
            Data type to use for optimization.
        """
        assert len(filter.shape) == 4, "Filter must be of shape (depth, height, width, channels)"
        self._filter_shape = np.array(filter.shape)
        self._convolver = Convolver(filter, dtype=dtype, norm=norm)

        shape = (int(np.prod(self._filter_shape)), int(np.prod(self._filter_shape)))
        super(RealFFTConvolve2D, self).__init__(shape=shape, dtype=dtype)

    def custom_lipschitz_cst(self):
        for layer in range:
            pass
            """
            H = RealFFTConvolve2D(layer, dtype=dtype)
            H.compute_lipschitz_cst()
            self._H.append(H)"""


    def __call__(self, x: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        # like here: https://github.com/PyLops/pylops/blob/3e7eb22a62ec60e868ccdd03bc4b54806851cb26/pylops/signalprocessing/ConvolveND.py#L103
        y = self._convolver.convolve(np.reshape(x, self._filter_shape))
        return y.ravel()

    def adjoint(self, y: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        x = self._convolver.deconvolve(np.reshape(y, self._filter_shape))
        return x.ravel()


class APGD(ReconstructionAlgorithm):
    def __init__(
        self,
        psf,
        max_iter=500,
        dtype=np.float32,
        diff_penalty=None,
        prox_penalty=APGDPriors.NONNEG,
        acceleration="BT",
        diff_lambda=0.001,
        prox_lambda=0.001,
        **kwargs
    ):
        """
        Wrapper for `Pycsou's APGD <https://matthieumeo.github.io/pycsou/html/api/algorithms/pycsou.opt.proxalgs.html?highlight=apgd#pycsou.opt.proxalgs.AcceleratedProximalGradientDescent>`__
        (accelerated proximal gradient descent) applied to lensless imaging.

        Parameters
        ----------
        psf : :py:class:`~numpy.ndarray`
            PSF that models forward propagation.
        max_iter : int, optional
            Maximal number of iterations.
        dtype : float32 or float64
            Data type to use for optimization.
        diff_penalty : None or str or :py:class:`~pycsou.core.functional.DifferentiableFunctional`
            Differentiable functional to serve as prior / regularization term.
            Default is None. See `Pycsou documentation <https://matthieumeo.github.io/pycsou/html/api/functionals/pycsou.func.penalty.html?highlight=penalty#module-pycsou.func.penalty>`__
            for available penalties.
        prox_penalty : None or str or :py:class:`~pycsou.core.functional.ProximableFunctional`
            Proximal functional to serve as prior / regularization term. Default
            is non-negative prior. See `Pycsou documentation <https://matthieumeo.github.io/pycsou/html/api/functionals/pycsou.func.penalty.html?highlight=penalty#module-pycsou.func.penalty>`__
            for available penalties.
        acceleration : [None, 'BT', 'CD']
            Which acceleration scheme should be used (None for no acceleration).
            "BT" (Beck and Teboule) has convergence `O(1/k^2)`, while "CD"
            (Chambolle and Dossal) has convergence `o(1/K^2)`. So "CD" should be
            faster. but from our experience "BT" gives better results.
        diff_lambda : float
            Weight of differentiable penalty.
        prox_lambda : float
            Weight of proximal penalty.
        """

        # PSF and data are the same size / shape
        self._original_shape = psf.shape
        self._original_size = psf.size

        self._apgd = None
        self._gen = None

        super(APGD, self).__init__(psf, dtype)

        self._max_iter = max_iter

        # Convolution operator
        self._H = RealFFTConvolve2D(psf, dtype=dtype)
        self._H.compute_lipschitz_cst()

        # initialize solvers which will be created when data is set
        if diff_penalty is not None:
            if diff_penalty == APGDPriors.L2:
                self._diff_penalty = diff_lambda * SquaredL2Norm(dim=self._H.shape[1])
            else:
                assert hasattr(diff_penalty, "jacobianT")
                self._diff_penalty = diff_lambda * diff_penalty(dim=self._H.shape[1])
        else:
            self._diff_penalty = None

        if prox_penalty is not None:
            if prox_penalty == APGDPriors.L1:
                self._prox_penalty = prox_lambda * L1Norm(dim=self._H.shape[1])
            elif prox_penalty == APGDPriors.NONNEG:
                self._prox_penalty = prox_lambda * NonNegativeOrthant(dim=self._H.shape[1])
            else:
                try:
                    self._prox_penalty = prox_lambda * prox_penalty(dim=self._H.shape[1])
                except ValueError:
                    print("Unexpected prior.")
        else:
            self._prox_penalty = None

        self._acc = acceleration

    def set_data(self, data):
        """
        For ``APGD``, we use data to initialize problem for Pycsou.

        Parameters
        ----------
        data : :py:class:`~numpy.ndarray`
            Lensless data on which to iterate to recover an estimate of the
             scene. Should match provide PSF, i.e. shape and 2D (grayscale) or
             3D (RGB).

        """
        super(APGD, self).set_data(np.repeat(data, self._original_shape[0],axis=0)) # we repeat the data to match the size of the PSF

        """ Set up problem """
        # Cost function
        loss = (1 / 2) * SquaredL2Loss(dim=self._H.shape[0], data=self._data.ravel())
        F = loss * self._H
        if self._diff_penalty is not None:
            F += self._diff_penalty

        if self._prox_penalty is not None:
            G = self._prox_penalty
            dim = G.shape[1]
        else:
            G = None
            dim = self._data.size

        self._apgd = APGD_pyc(dim=dim, F=F, G=G, acceleration=self._acc)

        # -- setup to print progress report
        self._apgd.old_iterand = deepcopy(self._apgd.init_iterand)
        self._apgd.update_diagnostics()
        self._gen = self._apgd.iterates(n=self._max_iter)

    def reset(self):
        self._image_est = np.zeros(self._original_size, dtype=self._dtype)
        if self._apgd is not None:
            self._apgd.reset()

            # -- setup to print progress report
            self._apgd.old_iterand = deepcopy(self._apgd.init_iterand)
            self._apgd.update_diagnostics()
            self._gen = self._apgd.iterates(n=self._max_iter)

    def _progress(self):
        """
        Pycsou has functionality for printing progress that we will make use of
        here.

        """
        self._apgd.update_diagnostics()
        self._apgd.old_iterand = deepcopy(self._apgd.iterand)
        self._apgd.print_diagnostics()

    def _update(self):
        next(self._gen)
        self._image_est[:] = self._apgd.iterand["iterand"]

    def _form_image(self):
        image = self._image_est.reshape(self._original_shape)
        image[image < 0] = 0
        return image
