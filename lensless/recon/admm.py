# #############################################################################
# admm.py
# =================
# Authors :
# Eric BEZZAM [ebezzam@gmail.com]
# Julien SAHLI [julien.sahli@epfl.ch]
# #############################################################################


import numpy as np
from lensless.recon.recon import ReconstructionAlgorithm
from scipy import fft
from lensless.utils.io import load_data
import time

try:
    import torch

    torch_available = True
except ImportError:
    torch_available = False


class ADMM(ReconstructionAlgorithm):
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
        mu1=1e-6,
        mu2=1e-5,
        mu3=4e-5,
        tau=0.0001,
        psi=None,
        psi_adj=None,
        psi_gram=None,
        pad=False,
        norm="backward",
        # PnP
        denoiser=None,
        **kwargs,
    ):
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
        pad : bool
            Whether to pad the image with zeros before applying the PSF. Default
            is False, as optimized data is already padded.
        norm : str
            Normalization to use for the convolution. Options are "forward",
            "backward", and "ortho". Default is "backward".
        """
        self._mu1 = mu1
        self._mu2 = mu2
        self._mu3 = mu3
        self._tau = tau

        # 3D ADMM is not supported yet
        assert len(psf.shape) == 4, "PSF must be 4D: (depth, height, width, channels)."
        if psf.shape[0] > 1:
            raise NotImplementedError(
                "3D ADMM is not supported yet, use gradient descent or APGD instead."
            )

        # call reset() to initialize matrices
        self._proj = self._Psi
        super(ADMM, self).__init__(
            psf, dtype, pad=pad, norm=norm, denoiser=denoiser, reset=False, **kwargs
        )

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

            # - need to reset with new projector
            self._proj = self._Psi

        # precompute_R_divmat (self._H computed by constructor with reset())
        if self.is_torch:
            self._PsiTPsi = self._PsiTPsi.to(self._psf.device)

        # check denoiser for PnP
        if self._denoiser is not None:
            self._denoiser_use_dual = denoiser["use_dual"]

            # - need to reset with new projector
            self._proj = self._denoiser
            # identify function
            self._PsiT = lambda x: x

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
        if self.is_torch:
            # TODO initialize without padding
            # initialize image estimate as [Batch, Depth, Height, Width, Channels]
            if self._initial_est is not None:
                self._image_est = self._initial_est
            else:
                self._image_est = torch.zeros([1] + self._padded_shape, dtype=self._dtype).to(
                    self._psf.device
                )

            # self._image_est = torch.zeros_like(self._psf)
            self._X = torch.zeros_like(self._image_est)
            # self._U = torch.zeros_like(self._Psi(self._image_est))
            if self._denoiser is not None:
                # PnP
                self._U = torch.zeros_like(
                    self._denoiser(self._image_est, self._denoiser_noise_level)
                )
            else:
                self._U = torch.zeros_like(self._proj(self._image_est))
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

            # precompute _R_divmat
            self._R_divmat = 1.0 / (
                self._mu1 * (torch.abs(self._convolver._Hadj * self._convolver._H))
                + self._mu2 * torch.abs(self._PsiTPsi)
                + self._mu3
            ).type(self._complex_dtype)

            # precompute_X_divmat
            self._X_divmat = 1.0 / (self._convolver._pad(torch.ones_like(self._psf)) + self._mu1)
            # self._X_divmat = 1.0 / (torch.ones_like(self._psf) + self._mu1)

        else:
            if self._initial_est is not None:
                self._image_est = self._initial_est
            else:
                self._image_est = np.zeros([1] + self._padded_shape, dtype=self._dtype)

            # self._U = np.zeros(np.r_[self._padded_shape, [2]], dtype=self._dtype)
            self._X = np.zeros_like(self._image_est)
            # self._U = np.zeros_like(self._Psi(self._image_est))
            self._U = np.zeros_like(self._proj(self._image_est))
            self._W = np.zeros_like(self._X)
            if self._image_est.max():
                # if non-zero
                # self._forward_out = self._forward()
                self._forward_out = self._convolver.convolve(self._image_est)
                self._Psi_out = self._Psi(self._image_est)
            else:
                self._forward_out = np.zeros_like(self._X)
                self._Psi_out = np.zeros_like(self._U)

            self._xi = np.zeros_like(self._image_est)
            self._eta = np.zeros_like(self._U)
            self._rho = np.zeros_like(self._X)

            # precompute R_divmat
            self._R_divmat = 1.0 / (
                self._mu1 * (np.abs(self._convolver._Hadj * self._convolver._H))
                + self._mu2 * np.abs(self._PsiTPsi)
                + self._mu3
            ).astype(self._complex_dtype)

            # precompute_X_divmat
            self._X_divmat = 1.0 / (
                self._convolver._pad(np.ones(self._psf_shape, dtype=self._dtype)) + self._mu1
            )

    def _U_update(self):
        """Total variation update."""
        # to avoid computing sparse operator twice
        if self._denoiser is not None:
            # PnP
            if self._denoiser_use_dual:
                self._U = self._denoiser(
                    self._U + self._eta / self._mu2,
                    self._denoiser_noise_level,
                )
            else:
                self._U = self._denoiser(self._image_est, self._denoiser_noise_level)
        else:
            self._U = soft_thresh(
                self._Psi_out + self._eta / self._mu2, thresh=self._tau / self._mu2
            )

    def _X_update(self):
        # to avoid computing forward model twice
        # self._X = self._X_divmat * (self._xi + self._mu1 * self._forward_out + self._data)
        self._X = self._X_divmat * (
            self._xi + self._mu1 * self._forward_out + self._convolver._pad(self._data)
        )

    def _W_update(self):
        """Non-negativity update"""
        if self.is_torch:
            self._W = torch.maximum(
                self._rho / self._mu3 + self._image_est, torch.zeros_like(self._image_est)
            )
        else:
            self._W = np.maximum(self._rho / self._mu3 + self._image_est, 0)

    def _image_update(self):
        if self._denoiser is not None:
            # PnP
            rk = (
                (self._mu3 * self._W - self._rho)
                # + self._mu2 * self._U
                + self._mu2 * self._U - self._eta
                if self._denoiser_use_dual
                else self._mu2 * self._U
                + self._convolver.deconvolve(self._mu1 * self._X - self._xi)
            )
        else:
            rk = (
                (self._mu3 * self._W - self._rho)
                + self._PsiT(self._mu2 * self._U - self._eta)
                + self._convolver.deconvolve(self._mu1 * self._X - self._xi)
            )

        # rk = self._convolver._pad(rk)

        if self.is_torch:
            freq_space_result = self._R_divmat * torch.fft.rfft2(rk, dim=(-3, -2))
            self._image_est = torch.fft.irfft2(
                freq_space_result, dim=(-3, -2), s=self._convolver._padded_shape[-3:-1]
            )
        else:
            freq_space_result = self._R_divmat * fft.rfft2(rk, axes=(-3, -2))
            self._image_est = fft.irfft2(
                freq_space_result, axes=(-3, -2), s=self._convolver._padded_shape[-3:-1]
            )

        # self._image_est = self._convolver._crop(res)

    def _xi_update(self):
        # to avoid computing forward model twice
        self._xi += self._mu1 * (self._forward_out - self._X)

    def _eta_update(self):
        # to avoid finite difference operataion again?
        if self._denoiser is not None:
            # PnP
            self._eta += self._mu2 * (self._image_est - self._U)
        else:
            self._eta += self._mu2 * (self._Psi_out - self._U)

    def _rho_update(self):
        self._rho += self._mu3 * (self._image_est - self._W)

    def _update(self, iter):
        self._U_update()
        self._X_update()
        self._W_update()
        self._image_update()

        # update forward and sparse operators
        self._forward_out = self._convolver.convolve(self._image_est)
        if self._denoiser is None:
            self._Psi_out = self._Psi(self._image_est)

        self._xi_update()
        if self._denoiser is None:
            self._eta_update()
        elif self._denoiser_use_dual:
            self._eta_update()
        self._rho_update()

    def _form_image(self):
        image = self._convolver._crop(self._image_est)

        # # TODO without cropping
        # image = self._image_est

        image[image < 0] = 0
        return image


def soft_thresh(x, thresh):
    if torch_available and isinstance(x, torch.Tensor):
        return torch.sign(x) * torch.max(torch.abs(x) - thresh, torch.zeros_like(x))
    else:
        # numpy automatically applies functions to each element of the array
        return np.sign(x) * np.maximum(0, np.abs(x) - thresh)


def finite_diff(x):
    """Gradient of image estimate, approximated by finite difference. Space where image is assumed sparse."""
    if torch_available and isinstance(x, torch.Tensor):
        return torch.stack(
            (torch.roll(x, 1, dims=-3) - x, torch.roll(x, 1, dims=-2) - x), dim=len(x.shape)
        )
    else:
        return np.stack(
            (np.roll(x, 1, axis=-3) - x, np.roll(x, 1, axis=-2) - x),
            axis=len(x.shape),
        )


def finite_diff_adj(x):
    """Adjoint of finite difference operator."""
    if torch_available and isinstance(x, torch.Tensor):
        diff1 = torch.roll(x[..., 0], -1, dims=-3) - x[..., 0]
        diff2 = torch.roll(x[..., 1], -1, dims=-2) - x[..., 1]
    else:
        diff1 = np.roll(x[..., 0], -1, axis=-3) - x[..., 0]
        diff2 = np.roll(x[..., 1], -1, axis=-2) - x[..., 1]
    return diff1 + diff2


def finite_diff_gram(shape, dtype=None, is_torch=False):
    """Gram matrix of finite difference operator."""
    if is_torch:
        if dtype is None:
            dtype = torch.float32
        gram = torch.zeros(shape, dtype=dtype)

    else:
        if dtype is None:
            dtype = np.float32
        gram = np.zeros(shape, dtype=dtype)

    if shape[0] == 1:
        gram[0, 0, 0] = 4
        gram[0, 0, 1] = gram[0, 0, -1] = gram[0, 1, 0] = gram[0, -1, 0] = -1
    else:
        gram[0, 0, 0] = 6
        gram[0, 0, 1] = gram[0, 0, -1] = gram[0, 1, 0] = gram[0, -1, 0] = gram[1, 0, 0] = gram[
            -1, 0, 0
        ] = -1

    if is_torch:
        return torch.fft.rfft2(gram, dim=(-3, -2))
    else:
        return fft.rfft2(gram, axes=(-3, -2))


def apply_admm(psf_fp, data_fp, n_iter, verbose=False, **kwargs):

    # load data
    psf, data = load_data(psf_fp=psf_fp, data_fp=data_fp, plot=False, **kwargs)

    # create reconstruction object
    recon = ADMM(psf, n_iter=n_iter)

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
