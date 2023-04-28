from lensless.recon import ReconstructionAlgorithm
import inspect
import numpy as np
from typing import Optional
from lensless.rfft_convolve import RealFFTConvolve2D as Convolver

import pycsou.abc as pyca
import pycsou.operator.func as func
import pycsou.opt.solver as solver
import pycsou.opt.stop as stop
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.ptype as pyct


class APGDPriors:
    """
    Priors (compatible with Pycsou) for APGD.

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


class RealFFTConvolve2D(pyca.LinOp):
    def __init__(
        self, filter: pyct.NDArray, dtype: Optional[type] = None, norm: str = "ortho", **kwargs
    ):
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
        norm : str
            Normalization to use for convolution. See :py:class:`~lensless.rfft_convolve.RealFFTConvolve2D`
        """

        assert len(filter.shape) == 3
        self._filter_shape = np.array(filter.shape)
        self._convolver = Convolver(filter, dtype=dtype, norm=norm)

        shape = (int(np.prod(self._filter_shape)), int(np.prod(self._filter_shape)))
        super(RealFFTConvolve2D, self).__init__(shape=shape)

    @pycrt.enforce_precision(i="x")
    @pycu.vectorize(i="x")
    def apply(self, x: pyct.NDArray) -> pyct.NDArray:
        y = self._convolver.convolve(np.reshape(x, self._filter_shape))
        return y.ravel()

    @pycrt.enforce_precision(i="y")
    @pycu.vectorize(i="y")
    def adjoint(self, y: pyct.NDArray) -> pyct.NDArray:
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
        acceleration=True,
        diff_lambda=0.001,
        prox_lambda=0.001,
        disp=100,
        rel_error=1e-6,
        lipschitz_tight=True,
        lipschitz_tol=1.0,
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
        acceleration : bool, optional
            Whether to use acceleration or not. Default is True.
        diff_lambda : float
            Weight of differentiable penalty.
        prox_lambda : float
            Weight of proximal penalty.
        disp : int, optional
            Display frequency. Default is 100.
        rel_error : float, optional
            Relative error to stop optimization. Default is 1e-6.
        lipschitz_tight : bool, optional
            Whether to use tight Lipschitz constant or not. Default is True.
        lipschitz_tol : float, optional
            Tolerance to compute Lipschitz constant. Default is 1.
        """

        # PSF and data are the same size / shape
        self._original_shape = psf.shape
        self._original_size = psf.size

        self._apgd = None
        self._gen = None

        super(APGD, self).__init__(psf, dtype, max_iter=max_iter, **kwargs)

        self._stop_crit = stop.RelError(eps=rel_error) | stop.MaxIter(max_iter)
        self._disp = disp

        # Convolution operator
        self._H = RealFFTConvolve2D(self._psf, dtype=dtype)
        self._H.lipschitz(tol=lipschitz_tol, tight=lipschitz_tight)

        # initialize solvers which will be created when data is set
        if diff_penalty is not None:
            if diff_penalty == APGDPriors.L2:
                self._diff_penalty = diff_lambda * func.SquaredL2Norm(dim=self._H.shape[1])
            else:
                assert hasattr(diff_penalty, "jacobianT")
                self._diff_penalty = diff_lambda * diff_penalty(dim=self._H.shape[1])
        else:
            self._diff_penalty = None

        if prox_penalty is not None:
            if prox_penalty == APGDPriors.L1:
                self._prox_penalty = prox_lambda * func.L1Norm(dim=self._H.shape[1])
            elif prox_penalty == APGDPriors.NONNEG:
                self._prox_penalty = prox_lambda * func.PositiveOrthant(dim=self._H.shape[1])
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
        if not self._is_rgb:
            assert len(data.shape) == 2
            data = data[:, :, np.newaxis]
        assert len(self._psf_shape) == len(data.shape)
        self._data = data

        """ Set up problem """
        # Cost function
        loss = (1 / 2) * func.SquaredL2Norm(dim=self._H.shape[0]).asloss(self._data.ravel())
        F = loss * self._H
        if self._diff_penalty is not None:
            F += self._diff_penalty

        self._apgd = solver.PGD(
            f=F, g=self._prox_penalty, show_progress=False, verbosity=self._disp
        )

        self._apgd.fit(
            x0=np.zeros(F.shape[1]),
            # x0=rng.normal(size=F.dim),
            stop_crit=self._stop_crit,
            track_objective=True,
            mode=pyca.solver.Mode.MANUAL,
            acceleration=self._acc,
        )

    def reset(self):
        self._image_est = np.zeros(self._original_size, dtype=self._dtype)

    def _update(self):
        res = next(self._apgd.steps())
        self._image_est[:] = res["x"]

    def _form_image(self):
        image = self._image_est.reshape(self._original_shape)
        image[image < 0] = 0
        return image
