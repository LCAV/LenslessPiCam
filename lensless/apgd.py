from lensless.recon import ReconstructionAlgorithm
import inspect
import numpy as np
from pycsou.linop.conv import Convolve2D
from pycsou.func.loss import SquaredL2Loss
from pycsou.func.penalty import NonNegativeOrthant, SquaredL2Norm, L1Norm
from pycsou.opt.proxalgs import APGD as APGD_pyc
from copy import deepcopy


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


class APGD(ReconstructionAlgorithm):
    def __init__(
        self,
        psf,
        max_iter=500,
        dtype=np.float32,
        prior=APGDPriors.NONNEG,
        acceleration="BT",
        reg_weight=0.001,
    ):
        """
        Wrapper for Pycsou's APGD (accelerated proximal gradient descent)
        applied to lensless imaging.

        Pycsou APGD documentation: https://matthieumeo.github.io/pycsou/html/api/algorithms/pycsou.opt.proxalgs.html?highlight=apgd#pycsou.opt.proxalgs.AcceleratedProximalGradientDescent

        Parameters
        ----------
        psf :py:class:`~numpy.ndarray`
            PSF that models forward propagation.
        max_iter : int, optional
            Maximal number of iterations.
        dtype : float32 or float64
            Data type to use for optimization.
        prior : :py:class:`~pycsou.func.penalty`
            Penalty functional to serve as prior / regularization term. Default
            is non-negative prior. See Pycsou documentation for available
            penalties: https://matthieumeo.github.io/pycsou/html/api/functionals/pycsou.func.penalty.html?highlight=penalty#module-pycsou.func.penalty
        acceleration : [None, 'BT', 'CD']
            Which acceleration scheme should be used (None for no acceleration).
            "BT" (Beck and Teboule) has convergence `O(1/k^2)`, while "CD"
            (Chambolle and Dossal) has convergence `o(1/K^2)`. So "CD" should be
            faster. but from our experience "BT" gives better results.
        reg_weight : float
            Weight of regularization / prior.
        """

        self._max_iter = max_iter

        # PSF and data are the same size / shape
        self._original_shape = psf.shape
        self._original_size = psf.size

        # Convolution operator
        self._H = Convolve2D(size=psf.size, filter=psf, shape=psf.shape, dtype=dtype)
        self._H.compute_lipschitz_cst()

        # initialize solvers which will be created when data is set
        if prior == APGDPriors.L2:
            self._prior = SquaredL2Norm
        elif prior == APGDPriors.L1:
            self._prior = L1Norm
        elif prior == APGDPriors.NONNEG:
            self._prior = NonNegativeOrthant
        else:
            raise ValueError("Unexpected prior.")
        self._acc = acceleration
        self._lambda = reg_weight
        self._apgd = None
        self._gen = None

        # TODO call reset() to initialize matrices?
        super(APGD, self).__init__(psf, dtype)

    def set_data(self, data):
        if not self._is_rgb:
            assert len(data.shape) == 2
            data = data[:, :, np.newaxis]
        assert len(self._psf_shape) == len(data.shape)
        self._data = data

        """ Set up problem """
        # Cost function
        loss = (1 / 2) * SquaredL2Loss(dim=self._H.shape[0], data=self._data.ravel())
        F = loss * self._H
        if self._prior == SquaredL2Norm:
            dim = self._data.size
            F += self._lambda * self._prior(dim=self._H.shape[1])
            G = None
        else:
            G = self._lambda * self._prior(dim=self._H.shape[1])
            dim = G.shape[1]
        self._apgd = APGD_pyc(dim=dim, F=F, G=G, acceleration=self._acc)

        # -- setup to print progress report
        self._apgd.old_iterand = deepcopy(self._apgd.init_iterand)
        self._apgd.update_diagnostics()
        self._gen = self._apgd.iterates(n=self._max_iter)

    def reset(self):
        self._image_est = np.zeros(self._original_size).astype(self._dtype)
        if self._apgd is not None:
            self._apgd.reset()

            # -- setup to print progress report
            self._apgd.old_iterand = deepcopy(self._apgd.init_iterand)
            self._apgd.update_diagnostics()
            self._gen = self._apgd.iterates(n=self._max_iter)

    def _progress(self):
        self._apgd.update_diagnostics()
        self._apgd.old_iterand = deepcopy(self._apgd.iterand)
        self._apgd.print_diagnostics()

    def _update(self):
        next(self._gen)
        self._image_est = self._apgd.iterand["iterand"]

    def _form_image(self):
        return self._image_est.reshape(self._original_shape)
