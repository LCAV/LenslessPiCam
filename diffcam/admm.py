import numpy as np
from diffcam.recon import ReconstructionAlgorithm
from scipy import fft


class ADMM(ReconstructionAlgorithm):
    def __init__(
        self,
        psf,
        dtype=np.float32,
        mu1=1e-6,
        mu2=1e-5,
        mu3=4e-5,
        tau=0.0001,
        psi=None,
        psi_adj=None,
        psi_gram=None,
    ):
        self._mu1 = mu1
        self._mu2 = mu2
        self._mu3 = mu3
        self._tau = tau

        # call reset() to initialize matrices
        super(ADMM, self).__init__(psf, dtype)

        # set prior
        if psi is None:
            # use already defined Psi and PsiT
            self._PsiTPsi = finite_diff_gram(self._padded_shape, self._dtype)
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

        # precompute_R_divmat
        self._R_divmat = 1.0 / (
            self._mu1 * (np.abs(np.conj(self._H) * self._H))
            + self._mu2 * np.abs(self._PsiTPsi)
            + self._mu3
        ).astype(self._complex_dtype)

    def _Psi(self, x):
        return finite_diff(x)

    def _PsiT(self, U):
        return finite_diff_adj(U)

    def _crop(self, x):
        return x[self._start_idx[0] : self._end_idx[0], self._start_idx[1] : self._end_idx[1]]

    def _pad(self, v):
        """adjoint of cropping"""
        vpad = np.zeros(self._padded_shape).astype(v.dtype)
        vpad[self._start_idx[0] : self._end_idx[0], self._start_idx[1] : self._end_idx[1]] = v
        return vpad

    def _forward(self):
        """Convolution with frequency response."""
        return fft.ifftshift(
            fft.irfft2(
                fft.rfft2(self._image_est, axes=(0, 1)) * self._H,
                axes=(0, 1),
            ),
            axes=(0, 1),
        )

    def _backward(self, x):
        """adjoint of forward / convolution"""
        return fft.ifftshift(
            fft.irfft2(fft.rfft2(x, axes=(0, 1)) * np.conj(self._H), axes=(0, 1)),
            axes=(0, 1),
        )

    def reset(self):
        # spatial frequency response
        self._H = fft.rfft2(self._pad(self._psf), axes=(0, 1)).astype(self._complex_dtype)

        self._X = np.zeros(self._padded_shape, dtype=self._dtype)
        self._U = np.zeros(np.r_[self._padded_shape, [2]], dtype=self._dtype)
        self._image_est = np.zeros_like(self._X)
        self._W = np.zeros_like(self._X)
        if self._image_est.max():
            # if non-zero
            self._forward_out = self._forward()
            self._Psi_out = self._Psi(self._image_est)
        else:
            self._forward_out = np.zeros_like(self._X)
            self._Psi_out = np.zeros_like(self._U)

        self._xi = np.zeros_like(self._image_est)
        self._eta = np.zeros_like(self._U)
        self._rho = np.zeros_like(self._X)

        # precompute_X_divmat
        self._X_divmat = 1.0 / (self._pad(np.ones(self._psf_shape, dtype=self._dtype)) + self._mu1)

    def _U_update(self):
        """Total variation update."""
        # to avoid computing sparse operator twice
        self._U = soft_thresh(self._Psi_out + self._eta / self._mu2, self._tau / self._mu2)

    def _X_update(self):
        # to avoid computing forward model twice
        self._X = self._X_divmat * (self._xi + self._mu1 * self._forward_out + self._data)

    def _image_update(self):
        rk = (
            (self._mu3 * self._W - self._rho)
            + self._PsiT(self._mu2 * self._U - self._eta)
            + self._backward(self._mu1 * self._X - self._xi)
        )
        freq_space_result = self._R_divmat * fft.rfft2(rk, axes=(0, 1))
        self._image_est = fft.irfft2(freq_space_result, axes=(0, 1))

    def _W_update(self):
        """Non-negativity update"""
        self._W = np.maximum(self._rho / self._mu3 + self._image_est, 0)

    def _xi_update(self):
        # to avoid computing forward model twice
        self._xi += self._mu1 * (self._forward_out - self._X)

    def _eta_update(self):
        # to avoid finite difference operataion again?
        self._eta += self._mu2 * (self._Psi_out - self._U)

    def _rho_update(self):
        self._rho += self._mu3 * (self._image_est - self._W)

    def _update(self):
        self._U_update()
        self._X_update()
        self._image_update()

        # update forward and sparse operators
        self._forward_out = self._forward()
        self._Psi_out = self._Psi(self._image_est)

        self._W_update()
        self._xi_update()
        self._eta_update()
        self._rho_update()

    def _form_image(self):
        image = self._crop(self._image_est)
        image[image < 0] = 0
        return image

    def set_data(self, data):
        if not self._is_rgb:
            assert len(data.shape) == 2
            data = data[:, :, np.newaxis]
        assert len(self._psf_shape) == len(data.shape)
        self._data = self._pad(data)
        self.reset()


def soft_thresh(x, thresh):
    # numpy automatically applies functions to each element of the array
    return np.sign(x) * np.maximum(0, np.abs(x) - thresh)


def finite_diff(x):
    """Gradient of image estimate, approximated by finite difference. Space where image is assumed sparse."""
    return np.stack(
        (np.roll(x, 1, axis=0) - x, np.roll(x, 1, axis=1) - x),
        axis=len(x.shape),
    )


def finite_diff_adj(x):
    diff1 = np.roll(x[..., 0], -1, axis=0) - x[..., 0]
    diff2 = np.roll(x[..., 1], -1, axis=1) - x[..., 1]
    return diff1 + diff2


def finite_diff_gram(shape, dtype=np.float32):
    gram = np.zeros(shape, dtype=dtype)
    gram[0, 0] = 4
    gram[0, 1] = gram[1, 0] = gram[0, -1] = gram[-1, 0] = -1
    return fft.rfft2(gram, axes=(0, 1))
