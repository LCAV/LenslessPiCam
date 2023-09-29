# #############################################################################
# rfft_convolve.py
# =================
# Authors :
# Eric BEZZAM [ebezzam@gmail.com]
# Julien SAHLI [julien.sahli@epfl.ch]
# #############################################################################


"""
2D convolution in Fourier domain, with same real-valued kernel.
"""

import numpy as np
from scipy import fft
from scipy.fftpack import next_fast_len

try:
    import torch

    torch_available = True
except ImportError:
    torch_available = False


class RealFFTConvolve2D:
    def __init__(self, psf, dtype=None, pad=True, norm="ortho", **kwargs):
        """
        Linear operator that performs convolution in Fourier domain, and assumes
        real-valued signals.

        Parameters
        ----------
        psf :py:class:`~numpy.ndarray` or :py:class:`~torch.Tensor`
            Point spread function (PSF) that models forward propagation.
            Must be of shape (depth, height, width, channels) even if
            depth = 1 and channels = 1. You can use :py:func:`~lensless.io.load_psf`
            to load a PSF from a file such that it is in the correct format.
        dtype : float32 or float64
            Data type to use for optimization.
        pad : bool, optional
            Whether data needs to be padded prior to convolution. User may wish to
            optimize padded data and set this to False, as is done for :py:class:`~lensless.ADMM`.
            Defaults to True.
        norm : str, optional
            Normalization to use for FFT. Defaults to 'ortho'.
        """

        self.is_torch = False
        if torch_available and isinstance(psf, torch.Tensor):
            self.is_torch = True

        # prepare shapes for reconstruction

        assert len(psf.shape) == 4, "Expected 4D PSF of shape (depth, width, height, channels)"
        self._use_3d = psf.shape[0] != 1
        self._is_rgb = psf.shape[3] == 3
        assert self._is_rgb or psf.shape[3] == 1

        # save normalization
        self.norm = norm

        # set dtype
        if dtype is None:
            if self.is_torch:
                dtype = torch.float32
            else:
                dtype = np.float32

        if self.is_torch:
            self._psf = psf.type(dtype)
        else:
            self._psf = psf.astype(dtype)

        self._psf_shape = np.array(self._psf.shape)

        # cropping / padding indexes
        self._padded_shape = 2 * self._psf_shape[-3:-1] - 1
        self._padded_shape = np.array([next_fast_len(i) for i in self._padded_shape])
        self._padded_shape = list(
            np.r_[self._psf_shape[-4], self._padded_shape, self._psf_shape[-1]]
        )
        self._start_idx = (self._padded_shape[-3:-1] - self._psf_shape[-3:-1]) // 2
        self._end_idx = self._start_idx + self._psf_shape[-3:-1]
        self.pad = pad  # Whether necessary to pad provided data

        # precompute filter in frequency domain
        if self.is_torch:
            self._H = torch.fft.rfft2(
                self._pad(self._psf), norm=norm, dim=(-3, -2), s=self._padded_shape[-3:-1]
            )
            self._Hadj = torch.conj(self._H)
            self._padded_data = (
                None  # This must be reinitialized each time to preserve differentiability
            )
        else:
            self._H = fft.rfft2(self._pad(self._psf), axes=(-3, -2), norm=norm)
            self._Hadj = np.conj(self._H)
            self._padded_data = np.zeros(self._padded_shape).astype(dtype)

        self.dtype = dtype

    def _crop(self, x):
        return x[
            ..., self._start_idx[0] : self._end_idx[0], self._start_idx[1] : self._end_idx[1], :
        ]

    def _pad(self, v):
        if len(v.shape) == 5:
            batch_size = v.shape[0]
        elif len(v.shape) == 4:
            batch_size = 1
        else:
            raise ValueError("Expected 4D or 5D tensor")
        shape = [batch_size] + self._padded_shape
        if self.is_torch:
            vpad = torch.zeros(size=shape, dtype=v.dtype, device=v.device)
        else:
            vpad = np.zeros(shape).astype(v.dtype)
        vpad[
            ..., self._start_idx[0] : self._end_idx[0], self._start_idx[1] : self._end_idx[1], :
        ] = v
        return vpad

    def convolve(self, x):
        """
        Convolve with pre-computed FFT of provided PSF.
        """
        if self.pad:
            self._padded_data = self._pad(x)
        else:
            if self.is_torch:
                self._padded_data = x  # .type(self.dtype).to(self._psf.device)
            else:
                self._padded_data[:] = x  # .astype(self.dtype)

        if self.is_torch:
            conv_output = torch.fft.ifftshift(
                torch.fft.irfft2(
                    torch.fft.rfft2(self._padded_data, dim=(-3, -2)) * self._H, dim=(-3, -2)
                ),
                dim=(-3, -2),
            )

        else:
            conv_output = fft.ifftshift(
                fft.irfft2(fft.rfft2(self._padded_data, axes=(-3, -2)) * self._H, axes=(-3, -2)),
                axes=(-3, -2),
            )
        if self.pad:
            return self._crop(conv_output)
        else:
            return conv_output

    def deconvolve(self, y):
        """
        Deconvolve with adjoint of pre-computed FFT of provided PSF.
        """
        if self.pad:
            self._padded_data = self._pad(y)
        else:
            if self.is_torch:
                self._padded_data = y  # .type(self.dtype).to(self._psf.device)
            else:
                self._padded_data[:] = y  # .astype(self.dtype)
        if self.is_torch:
            deconv_output = torch.fft.ifftshift(
                torch.fft.irfft2(
                    torch.fft.rfft2(self._padded_data, dim=(-3, -2)) * self._Hadj, dim=(-3, -2)
                ),
                dim=(-3, -2),
            )

        else:
            deconv_output = fft.ifftshift(
                fft.irfft2(fft.rfft2(self._padded_data, axes=(-3, -2)) * self._Hadj, axes=(-3, -2)),
                axes=(-3, -2),
            )

        if self.pad:
            return self._crop(deconv_output)
        else:
            return deconv_output
