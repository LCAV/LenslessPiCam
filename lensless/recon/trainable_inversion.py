# #############################################################################
# trainable_inversion.py
# ======================
# Authors :
# Eric BEZZAM [ebezzam@gmail.com]
# #############################################################################

from lensless.recon.trainable_recon import TrainableReconstructionAlgorithm


class TrainableInversion(TrainableReconstructionAlgorithm):
    def __init__(self, psf, dtype=None, K=1e-4, **kwargs):
        """
        Constructor for trainable inversion component as proposed in
        the FlatNet work: https://siddiquesalman.github.io/flatnet/

        Their implementation: https://github.com/siddiquesalman/flatnet/blob/d1dc179666a08df58c4bdf83c4274310ba3cd186/models/fftlayer.py#L70

        Parameters
        ----------
        psf : :py:class:`~torch.Tensor`
            Point spread function (PSF) that models forward propagation.
            Must be of shape (depth, height, width, channels) even if
            depth = 1 and channels = 1. You can use :py:func:`~lensless.io.load_psf`
            to load a PSF from a file such that it is in the correct format.
        dtype : float32 or float64
            Data type to use for optimization.
        K : float
            Regularization parameter.

        """

        super(TrainableInversion, self).__init__(psf, n_iter=1, dtype=dtype, reset=False, **kwargs)
        self._convolver._Hadj = self._convolver._Hadj / (self._convolver._H.norm() ** 2 + K)

        self.reset()

    def _form_image(self):
        self._image_est[self._image_est < 0] = 0
        return self._image_est

    def _set_psf(self, psf):
        return super()._set_psf(psf)

    def reset(self, batch_size=1):
        # no state variables
        return

    def _update(self, iter):
        self._image_est = self._convolver.deconvolve(self._data)
