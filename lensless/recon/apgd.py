# #############################################################################
# apgd.py
# =================
# Authors :
# Eric BEZZAM [ebezzam@gmail.com]
# Julien SAHLI [julien.sahli@epfl.ch]
# #############################################################################


from lensless.recon.recon import ReconstructionAlgorithm
import inspect
import numpy as np
from typing import Optional
from lensless.utils.image import resize
from lensless.recon.rfft_convolve import RealFFTConvolve2D as Convolver
import cv2

import pycsou.abc as pyca
import pycsou.operator.func as func
import pycsou.opt.solver as solver
import pycsou.opt.stop as stop
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.ptype as pyct
import pycsou.operator.linop as pycl


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
        assert len(filter.shape) == 4, "Filter must be of shape (depth, height, width, channels)"
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
        dtype="float32",
        diff_penalty=None,
        prox_penalty=APGDPriors.NONNEG,
        acceleration=True,
        diff_lambda=0.001,
        prox_lambda=0.001,
        disp=100,
        rel_error=None,
        lipschitz_tight=True,
        lipschitz_tol=1.0,
        img_shape=None,
        **kwargs
    ):
        """
        Wrapper for `Pycsou's PGD <https://github.com/matthieumeo/pycsou/blob/a74b714192821501371c89dbd44eac15a5456a0f/src/pycsou/opt/solver/pgd.py#L17>`__
        (accelerated proximal gradient descent) applied to lensless imaging.

        Parameters
        ----------
        psf : :py:class:`~numpy.ndarray`
            Point spread function (PSF) that models forward propagation.
            Must be of shape (depth, height, width, channels) even if
            depth = 1 and channels = 1. You can use :py:func:`~lensless.io.load_psf`
            to load a PSF from a file such that it is in the correct format.
        max_iter : int, optional
            Maximal number of iterations.
        dtype : float32 or float64
            Data type to use for optimization.
        diff_penalty : None or str or `DiffFunc`
            Differentiable functional to serve as prior / regularization term.
            Default is None. See `DiffFunc <https://github.com/matthieumeo/pycsou/blob/a74b714192821501371c89dbd44eac15a5456a0f/src/pycsou/abc/operator.py#L980>`_.
        prox_penalty : None or str or `ProxFunc`
            Proximal functional to serve as prior / regularization term. Default
            is non-negative prior. See `ProxFunc <https://github.com/matthieumeo/pycsou/blob/a74b714192821501371c89dbd44eac15a5456a0f/src/pycsou/abc/operator.py#L741>`_.
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
        img_shape : tuple, optional
            Shape of measurement (H, W, C). If None, assume shape of PSF.
        """

        assert isinstance(psf, np.ndarray), "PSF must be a numpy array"

        self._original_shape = psf.shape
        self._apgd = None

        self._stop_crit = stop.MaxIter(max_iter)
        if rel_error is not None:
            self._stop_crit = self._stop_crit | stop.RelError(eps=rel_error)
        self._disp = disp

        # Convolution (and optional downsampling) operator
        if img_shape is not None:

            meas_shape = np.array(img_shape[:2])
            rec_shape = np.array(self._original_shape[1:3])
            assert np.all(meas_shape <= rec_shape), "Image shape must be smaller than PSF shape"
            self.downsampling_factor = np.round(rec_shape / meas_shape).astype(int)

            # new PSF shape, must be integer multiple of image shape
            new_shape = tuple(np.array(meas_shape) * self.downsampling_factor) + (psf.shape[-1],)
            psf_re = resize(psf.copy(), shape=new_shape, interpolation=cv2.INTER_CUBIC)

            # combine operations
            conv = RealFFTConvolve2D(psf_re, dtype=dtype)
            ds = pycl.SubSample(
                psf_re.shape,
                slice(None),
                slice(0, -1, self.downsampling_factor[0]),
                slice(0, -1, self.downsampling_factor[1]),
                slice(None),
            )

            self._H = ds * conv

            super(APGD, self).__init__(psf_re, dtype, n_iter=max_iter, **kwargs)

        else:
            self.downsampling_factor = 1
            self._H = RealFFTConvolve2D(psf, dtype=dtype)

            super(APGD, self).__init__(psf, dtype, n_iter=max_iter, **kwargs)

        self._H.lipschitz(tol=lipschitz_tol, tight=lipschitz_tight)

        # initialize solvers which will be created when data is set
        if diff_penalty is not None:
            if diff_penalty == APGDPriors.L2:
                self._diff_penalty = diff_lambda * func.SquaredL2Norm(dim=self._H.shape[1])
            else:
                assert hasattr(diff_penalty, "jacobian")
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

        # super(APGD, self).set_data(
        #     np.repeat(data, self._original_shape[-4], axis=0)
        # )  # we repeat the data for each depth to match the size of the PSF

        data = np.repeat(data, self._original_shape[-4], axis=0)  # repeat for each depth
        assert isinstance(data, np.ndarray)
        assert len(data.shape) >= 3, "Data must be at least 3D: [..., width, height, channel]."

        assert np.all(
            self._psf_shape[-3:-1] == (np.array(data.shape)[-3:-1] * self.downsampling_factor)
        ), "PSF and data shape mismatch"

        if len(data.shape) == 3:
            self._data = data[None, None, ...]
        elif len(data.shape) == 4:
            self._data = data[None, ...]
        else:
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
            # x0=np.random.normal(size=F.shape[1]),
            stop_crit=self._stop_crit,
            track_objective=True,
            mode=pyca.solver.Mode.MANUAL,
            acceleration=self._acc,
        )

    def reset(self):
        if self._initial_est is not None:
            self._image_est = self._initial_est
        else:
            self._image_est = np.zeros(np.prod(self._psf_shape), dtype=self._dtype)

    def _update(self, iter):
        res = next(self._apgd.steps())
        self._image_est[:] = res["x"]

    def _form_image(self):
        image = self._image_est.reshape(self._psf_shape)
        image[image < 0] = 0
        if np.any(self._psf_shape != self._original_shape):
            image = resize(image, shape=self._original_shape)
        return image
