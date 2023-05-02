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
            2D filter to use.
        dtype : float32 or float64
            Data type to use for optimization.
        """

        self.is_torch = False
        if torch_available and isinstance(psf, torch.Tensor):
            self.is_torch = True

        # prepare shapes for reconstruction

        assert len(psf.shape) == 4
        self._use_3d = psf.shape[0] != 1
        self._is_rgb = psf.shape[3] == 3
        assert self._is_rgb or psf.shape[3] == 1
        self._psf_shape = np.array(self._psf.shape) #is it still used ?


        # set dtype
        if dtype is None:
            if self.is_torch:
                dtype = torch.float32
            else:
                dtype = np.float32
        else:
            if self.is_torch:
                self._psf = psf.type(dtype)
            else:
                self._psf = psf.astype(dtype)

        self._psf_shape = np.array(self._psf.shape)

        # cropping / padding indexes
        self._padded_shape = 2 * self._psf_shape[1:3] - 1
        self._padded_shape = np.array([next_fast_len(i) for i in self._padded_shape])
        self._padded_shape = list(np.r_[self._psf_shape[0], self._padded_shape, self._psf_shape[3]])
        self._start_idx = (self._padded_shape[1:3] - self._psf_shape[1:3]) // 2
        self._end_idx = self._start_idx + self._psf_shape[1:3]
        self.pad = pad  # Whether necessary to pad provided data

        # precompute filter in frequency domain
        if self.is_torch:
            self._H = torch.fft.rfft2(
                self._pad(self._psf), norm=norm, dim=(1, 2), s=self._padded_shape[1:3]
            )
            self._Hadj = torch.conj(self._H)
            self._padded_data = torch.zeros(size=self._padded_shape, dtype=dtype, device=psf.device)

        else:
            self._H = fft.rfft2(self._pad(self._psf), axes=(1, 2), norm=norm)
            self._Hadj = np.conj(self._H)
            self._padded_data = np.zeros(self._padded_shape).astype(dtype)

    def _crop(self, x):
        return x[:, self._start_idx[0] : self._end_idx[0], self._start_idx[1] : self._end_idx[1]]

    def _pad(self, v):
        if self.is_torch:
            vpad = torch.zeros(size=self._padded_shape, dtype=v.dtype, device=v.device)
        else:
            vpad = np.zeros(self._padded_shape).astype(v.dtype)
        vpad[:, self._start_idx[0] : self._end_idx[0], self._start_idx[1] : self._end_idx[1]] = v
        return vpad

    def convolve(self, x):
        """
        Convolve with pre-computed FFT of provided PSF.
        """
        if self.pad:
            self._padded_data[
                :, self._start_idx[0] : self._end_idx[0], self._start_idx[1] : self._end_idx[1]
            ] = x
        else:
            self._padded_data[:] = x

        if self.is_torch:
            conv_output = torch.fft.ifftshift(
                torch.fft.irfft2(
                    torch.fft.rfft2(self._padded_data, dim=(1, 2)) * self._H, dim=(1, 2)
                ),
                dim=(1, 2),
            )

        else:
            conv_output = fft.ifftshift(
                fft.irfft2(fft.rfft2(self._padded_data, axes=(1, 2)) * self._H, axes=(1, 2)),
                axes=(1, 2),
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
            self._padded_data[
                :, self._start_idx[0] : self._end_idx[0], self._start_idx[1] : self._end_idx[1]
            ] = y
        else:
            self._padded_data[:] = y

        if self.is_torch:

            deconv_output = torch.fft.ifftshift(
                torch.fft.irfft2(
                    torch.fft.rfft2(self._padded_data, dim=(1, 2)) * self._Hadj, dim=(1, 2)
                ),
                dim=(1, 2),
            )

        else:

            deconv_output = fft.ifftshift(
                fft.irfft2(fft.rfft2(self._padded_data, axes=(1, 2)) * self._Hadj, axes=(1, 2)),
                axes=(1, 2),
            )

        if self.pad:
            return self._crop(deconv_output)
        else:
            return deconv_output
