# #############################################################################
# unrolled_admm.py
# =================
# Authors :
# Yohann PERRON [yohann.perron@gmail.com]
# Eric BEZZAM [ebezzam@gmail.com]
# #############################################################################

from lensless.recon.trainable_recon import TrainableReconstructionAlgorithm
from lensless.recon.admm import soft_thresh, finite_diff, finite_diff_adj, finite_diff_gram


try:
    import torch

except ImportError:
    raise ImportError("Pytorch is require to use trainable reconstruction algorithms.")


class UnrolledADMM(TrainableReconstructionAlgorithm):
    """
    Object for applying unrolled version of :py:class:`~lensless.ADMM`.

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
        pad=False,
        norm="backward",
        **kwargs,
    ):
        """

        Parameters
        ----------
        psf : :py:class:`~torch.Tensor`
            Point spread function (PSF) that models forward propagation.
            Must be of shape (depth, height, width, channels) even if
            depth = 1 and channels = 1. You can use :py:func:`~lensless.io.load_psf`
            to load a PSF from a file such that it is in the correct format.
        dtype : float32 or float64
            Data type to use for optimization. Default is float32.
        n_iter : int, optional
            Number of iterations to unrolled, by default 5
        mu1 : float
            Initial step size for updating primal/dual variables.
        mu2 : float
            Initial step size for updating primal/dual variables.
        mu3 : float
            Initial step size for updating primal/dual variables.
        tau : float
            Initial weight for L1 norm of `psi` applied to the image estimate.
        psi : :py:class:`function`, optional
            Operator to map image to a space that the image is assumed to be
            sparse in (hence L1 norm). Default is to use total variation (TV)
            operator.
        psi_adj : :py:class:`function`
            Adjoint of `psi`.
        psi_gram : :py:class:`function`
            Function to compute gram of `psi`.
        pad : bool, optional
            Whether to pad the image with zeros before applying the PSF.
        norm : str
            Normalization to use for the convolution. Options are "forward",
            "backward", and "ortho". Default is "backward".
        """

        super(UnrolledADMM, self).__init__(
            psf, n_iter=n_iter, dtype=dtype, pad=pad, norm=norm, reset=False, **kwargs
        )

        if not self.skip_unrolled:
            self._mu1_p = torch.nn.Parameter(
                torch.ones(self._n_iter, device=self._psf.device) * mu1
            )
            self._mu2_p = torch.nn.Parameter(
                torch.ones(self._n_iter, device=self._psf.device) * mu2
            )
            self._mu3_p = torch.nn.Parameter(
                torch.ones(self._n_iter, device=self._psf.device) * mu3
            )
            self._tau_p = torch.nn.Parameter(
                torch.ones(self._n_iter, device=self._psf.device) * tau
            )
        else:
            self._mu1_p = torch.ones(self._n_iter, device=self._psf.device) * mu1
            self._mu2_p = torch.ones(self._n_iter, device=self._psf.device) * mu2
            self._mu3_p = torch.ones(self._n_iter, device=self._psf.device) * mu3
            self._tau_p = torch.ones(self._n_iter, device=self._psf.device) * tau

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

    def reset(self, batch_size=1):

        if self._data is not None:
            device = self._data.device
        else:
            device = self._convolver._H.device

        # ensure that mu1, mu2, mu3, tau are positive
        self._mu1 = torch.abs(self._mu1_p).to(device)
        self._mu2 = torch.abs(self._mu2_p).to(device)
        self._mu3 = torch.abs(self._mu3_p).to(device)
        self._tau = torch.abs(self._tau_p).to(device)

        # TODO initialize without padding
        if self._initial_est is not None:
            self._image_est = self._initial_est.to(device)
        else:
            self._image_est = torch.zeros([1] + self._padded_shape, dtype=self._dtype).to(device)

        self._X = torch.zeros_like(self._image_est)
        self._U = torch.zeros_like(self._Psi(self._image_est))
        self._W = torch.zeros_like(self._X)
        if self._image_est.max():
            # if non-zero
            self._forward_out = self._convolver.convolve(self._image_est)
            self._Psi_out = self._Psi(self._image_est)
        else:
            self._forward_out = torch.zeros_like(self._X)
            self._Psi_out = torch.zeros_like(self._U)

        self._xi = torch.zeros_like(self._image_est)
        self._eta = torch.zeros_like(self._U)
        self._rho = torch.zeros_like(self._X)

        # precompute_R_divmat [iter, batch, depth, height, width, channels]
        self._R_divmat = 1.0 / (
            self._mu1[:, None, None, None, None, None]
            * (torch.abs(self._convolver._Hadj * self._convolver._H))[None, ...]
            + self._mu2[:, None, None, None, None, None] * torch.abs(self._PsiTPsi).to(device)
            + self._mu3[:, None, None, None, None, None]
        ).type(self._complex_dtype)

        # precompute_X_divmat [iter, batch, depth, height, width, channels]
        self._X_divmat = 1.0 / (
            self._convolver._pad(torch.ones_like(self._convolver._psf))[None, ...]
            + self._mu1[:, None, None, None, None, None]
        )

    def _U_update(self, iter):
        """Total variation update."""
        # to avoid computing sparse operator twice
        self._U = soft_thresh(
            self._Psi_out + self._eta / self._mu2[iter], self._tau[iter] / self._mu2[iter]
        )

    def _X_update(self, iter):
        # to avoid computing forward model twice
        self._X = self._X_divmat[iter] * (
            self._xi + self._mu1[iter] * self._forward_out + self._convolver._pad(self._data)
        )

    def _image_update(self, iter):
        rk = (
            (self._mu3[iter] * self._W - self._rho)
            + self._PsiT(self._mu2[iter] * self._U - self._eta)
            + self._convolver.deconvolve(self._mu1[iter] * self._X - self._xi)
        )
        freq_space_result = self._R_divmat[iter] * torch.fft.rfft2(rk, dim=(-3, -2))
        self._image_est = torch.fft.irfft2(
            freq_space_result, dim=(-3, -2), s=self._convolver._padded_shape[-3:-1]
        )

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
        self._W_update(iter)
        self._image_update(iter)

        # update forward and sparse operators
        self._forward_out = self._convolver.convolve(self._image_est)
        self._Psi_out = self._Psi(self._image_est)

        self._xi_update(iter)
        self._eta_update(iter)
        self._rho_update(iter)

    def _form_image(self):
        image = self._convolver._crop(self._image_est)
        # image = torch.clamp(image, min=0)
        image = torch.clip(image, min=0.0)
        return image
