import numpy as np
from lensless.recon import ReconstructionAlgorithm
from scipy import fft


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
        dtype=np.float32,
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
        psf : :py:class:`~numpy.ndarray`
            Point spread function (PSF) that models forward propagation.
            2D (grayscale) or 3D (RGB) data can be provided and the shape will
            be used to determine which reconstruction (and allocate the
            appropriate memory).
        dtype : float32 or float64
            Data type to use for optimization.
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

    def _crop(self, x):
        return x[:, self._start_idx[0] : self._end_idx[0], self._start_idx[1] : self._end_idx[1], :]

    def _pad(self, v):
        """adjoint of cropping"""
        vpad = np.zeros(np.append(len(v), self._padded_shape[1:])).astype(
            v.dtype
        )  # we keep len(v) as fist axis as sometimes we want to pad original data and sometimes, 3d estimation
        for i in range(vpad.shape[0]):
            vpad[
                i, self._start_idx[0] : self._end_idx[0], self._start_idx[1] : self._end_idx[1]
            ] = v[i, :, :]
        return vpad

    def _forward(self):
        """Convolution with frequency response."""
        return fft.ifftshift(
            fft.irfft2(
                fft.rfft2(self._image_est, axes=(0, 1, 2), s=self._padded_shape[:3]) * self._H,
                axes=(0, 1, 2),
                s=self._padded_shape[:3],
            ),
            axes=(0, 1, 2),
        )

    def _backward(self, x):
        """adjoint of forward / convolution"""
        return fft.ifftshift(
            fft.irfft2(
                fft.rfft2(x, axes=(0, 1, 2), s=self._padded_shape[:3]) * np.conj(self._H),
                axes=(0, 1, 2),
                s=self._padded_shape[:3],
            ),
            axes=(0, 1, 2),
        )

    def reset(self):
        # spatial frequency response
        print("res psf :", np.array(self._psf).shape)
        print("pad : ", self._pad(self._psf).shape)
        print("padsh : ", self._padded_shape)
        self._H = fft.rfft2(self._pad(self._psf), axes=(1, 2), s=self._padded_shape[1:3]).astype(
            self._complex_dtype
        )
        print("res H :", np.array(self._H).shape)

        self._X = np.zeros(self._padded_shape, dtype=self._dtype)
        # self._U = np.zeros(np.r_[self._padded_shape, [2]], dtype=self._dtype)
        self._image_est = np.zeros_like(self._X)
        self._U = np.zeros_like(self._Psi(self._image_est))
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
        freq_space_result = self._R_divmat * fft.rfft2(rk, axes=(0, 1, 2), s=self._padded_shape[:3])
        self._image_est = fft.irfft2(freq_space_result, axes=(0, 1, 2), s=self._padded_shape[:3])

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
        return image.squeeze()

    def set_data(self, data):
        super(ADMM, self).set_data(data)
        self._data = self._pad(self._data)
        self.reset()


def soft_thresh(x, thresh):
    # numpy automatically applies functions to each element of the array
    return np.sign(x) * np.maximum(0, np.abs(x) - thresh)


def finite_diff(x):
    """Gradient of image estimate, approximated by finite difference. Space where image is assumed sparse."""
    return np.stack(
        (np.roll(x, 1, axis=0) - x, np.roll(x, 1, axis=1) - x, np.roll(x, 1, axis=2) - x),
        axis=len(x.shape),
    )


def finite_diff_adj(x):
    diff0 = np.roll(x[..., 0], -1, axis=0) - x[..., 0]
    diff1 = np.roll(x[..., 1], -1, axis=1) - x[..., 1]
    diff2 = np.roll(x[..., 2], -1, axis=2) - x[..., 2]
    return diff0 + diff1 + diff2


def finite_diff_gram(shape, dtype=np.float32):
    gram = np.zeros(shape, dtype=dtype)
    gram[0, 0, 0] = 4
    gram[0, 0, 1] = gram[0, 1, 0] = gram[0, 0, -1] = gram[0, -1, 0] = -1
    return fft.rfft2(gram, axes=(0, 1, 2))
