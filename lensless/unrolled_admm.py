import numpy as np
from lensless.trainable_recon import TrainableReconstructionAlgorithm
from lensless.admm import soft_thresh, finite_diff, finite_diff_adj, finite_diff_gram
from scipy import fft

try:
    import torch

    torch_available = True
except ImportError:
    torch_available = False


class unrolled_ADMM(TrainableReconstructionAlgorithm):
    """
    Object for applying ADMM (Alternating Direction Method of Multipliers) with
    a non-negativity constraint and a total variation (TV) prior.

    Paper about ADMM: https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf

    Slides about ADMM: https://web.stanford.edu/class/ee364b/lectures/admm_slides.pdf

    """

    def __init__(
        self,
        psf,
        dtype=None,
        n_iter=5,
        mu1=1e-6,
        mu2=1e-5,
        mu3=4e-5,
        tau=0.0001,
        psi=None,
        psi_adj=None,
        psi_gram=None,
        **kwargs
    ):
        """

        Parameters
        ----------
        psf : :py:class:`~numpy.ndarray` or :py:class:`~torch.Tensor`
            Point spread function (PSF) that models forward propagation.
            2D (grayscale) or 3D (RGB) data can be provided and the shape will
            be used to determine which reconstruction (and allocate the
            appropriate memory).
        dtype : float32 or float64
            Data type to use for optimization. Default is float32.
        mu1 : float
            Step size for updating primal/dual variables.
        mu2 : float
            Step size for updating primal/dual variables.
        mu3 : float
            Step size for updating primal/dual variables.
        tau : float
            Weight for L1 norm of `psi` applied to the image estimate.
        psi : :py:class:`function`, optional
            Operator to map image to a space that the image is assumed to be
            sparse in (hence L1 norm). Default is to use total variation (TV)
            operator.
        psi_adj : :py:class:`function`
            Adjoint of `psi`.
        psi_gram : :py:class:`function`
            Function to compute gram of `psi`.
        """

        # call reset() to initialize matrices
        super(unrolled_ADMM, self).__init__(
            psf, n_iter=n_iter, dtype=dtype, pad=False, norm="backward"
        )
        # super(ADMM, self).__init__(psf, dtype, pad=False, norm="ortho")

        self._mu1 = torch.nn.Parameter(torch.ones(self.n_iter, device=self._psf.device) * mu1)
        self._mu2 = torch.nn.Parameter(torch.ones(self.n_iter, device=self._psf.device) * mu2)
        self._mu3 = torch.nn.Parameter(torch.ones(self.n_iter, device=self._psf.device) * mu3)
        self._tau = torch.nn.Parameter(torch.ones(self.n_iter, device=self._psf.device) * tau)
        # set prior
        if psi is None:
            # use already defined Psi and PsiT
            self._PsiTPsi = finite_diff_gram(self._padded_shape, self._dtype, self.is_torch)
        else:
            assert psi_adj is not None
            assert psi_gram is not None
            assert callable(psi)
            assert callable(psi_adj)
            assert callable(psi_gram)
            # overwrite already defined Psi and PsiT
            self._Psi = psi
            self._PsiT = psi_adj
            self._PsiTPsi = psi_gram(self._padded_shape)

        self._PsiTPsi = self._PsiTPsi.to(self._psf.device)

        self.reset()

    def _Psi(self, x):
        """
        Operator to map image to space that the image is assumed to be sparse
        in.
        """
        return finite_diff(x)

    def _PsiT(self, U):
        """
        Adjoint of `_Psi`.
        """
        return finite_diff_adj(U)

    def reset(self):
        # needed because ReconstructionAlgorithm initializer call reset to early
        if not hasattr(self, "_mu1"):
            return
        # TODO initialize without padding
        self._image_est = torch.zeros(self._padded_shape, dtype=self._dtype).to(self._psf.device)
        # self._image_est = torch.zeros_like(self._psf)
        self._X = torch.zeros_like(self._image_est)
        self._U = torch.zeros_like(self._Psi(self._image_est))
        self._W = torch.zeros_like(self._X)
        if self._image_est.max():
            # if non-zero
            # self._forward_out = self._forward()
            self._forward_out = self._convolver.convolve(self._image_est)
            self._Psi_out = self._Psi(self._image_est)
        else:
            self._forward_out = torch.zeros_like(self._X)
            self._Psi_out = torch.zeros_like(self._U)

        self._xi = torch.zeros_like(self._image_est)
        self._eta = torch.zeros_like(self._U)
        self._rho = torch.zeros_like(self._X)

        # precompute_R_divmat
        self._R_divmat = 1.0 / (
            self._mu1[:, None, None, None]
            * (torch.abs(self._convolver._Hadj * self._convolver._H))[None, :, :, :]
            + self._mu2[:, None, None, None] * torch.abs(self._PsiTPsi)[None, :, :, :]
            + self._mu3[:, None, None, None]
        ).type(self._complex_dtype)

        # precompute_X_divmat
        self._X_divmat = 1.0 / (
            self._convolver._pad(torch.ones_like(self._psf))[None, :, :, :]
            + self._mu1[:, None, None, None]
        )
        # self._X_divmat = 1.0 / (torch.ones_like(self._psf) + self._mu1)

    def _U_update(self, iter):
        """Total variation update."""
        # to avoid computing sparse operator twice
        self._U = soft_thresh(
            self._Psi_out + self._eta / self._mu2[iter], self._tau[iter] / self._mu2[iter]
        )

    def _X_update(self, iter):
        # to avoid computing forward model twice
        # self._X = self._X_divmat * (self._xi + self._mu1 * self._forward_out + self._data)
        self._X = self._X_divmat[iter] * (
            self._xi + self._mu1[iter] * self._forward_out + self._convolver._pad(self._data)
        )

    def _image_update(self, iter):
        rk = (
            (self._mu3[iter] * self._W - self._rho)
            + self._PsiT(self._mu2[iter] * self._U - self._eta)
            + self._convolver.deconvolve(self._mu1[iter] * self._X - self._xi)
        )

        # rk = self._convolver._pad(rk)

        freq_space_result = self._R_divmat[iter] * torch.fft.rfft2(rk, dim=(0, 1))
        self._image_est = torch.fft.irfft2(freq_space_result, dim=(0, 1))

        # self._image_est = self._convolver._crop(res)

    def _W_update(self, iter):
        """Non-negativity update"""
        self._W = torch.maximum(
            self._rho / self._mu3[iter] + self._image_est, torch.zeros_like(self._image_est)
        )

    def _xi_update(self, iter):
        # to avoid computing forward model twice
        self._xi = self._xi + self._mu1[iter] * (self._forward_out - self._X)

    def _eta_update(self, iter):
        # to avoid finite difference operataion again?
        self._eta = self._eta + self._mu2[iter] * (self._Psi_out - self._U)

    def _rho_update(self, iter):
        self._rho = self._rho + self._mu3[iter] * (self._image_est - self._W)

    def _update(self, iter):

        self._U_update(iter)
        self._X_update(iter)
        self._image_update(iter)

        # update forward and sparse operators
        self._forward_out = self._convolver.convolve(self._image_est)
        self._Psi_out = self._Psi(self._image_est)

        self._W_update(iter)
        self._xi_update(iter)
        self._eta_update(iter)
        self._rho_update(iter)

    def _form_image(self):
        image = self._convolver._crop(self._image_est)

        # # TODO without cropping
        # image = self._image_est

        image[image < 0] = 0
        return image.squeeze()
