# #############################################################################
# trainable_recon.py
# =================
# Authors :
# Yohann PERRON [yohann.perron@gmail.com]
# #############################################################################

from lensless.recon import ReconstructionAlgorithm

try:
    import torch

except ImportError:
    raise ImportError("Pytorch is require to use trainable reconstruction algorithms.")


class TrainableReconstructionAlgorithm(ReconstructionAlgorithm, torch.nn.Module):
    """
    Abstract class for defining lensless imaging reconstruction algorithms with trainable parameters.

    The following abstract methods need to be defined:

    * ``_update``: updating state variables at each iterations.
    * ``reset``: reset state variables.
    * ``_form_image``: any pre-processing that needs to be done in order to view the image estimate, e.g. reshaping or clipping.

    One advantage of deriving from this abstract class is that functionality for
    iterating, saving, and visualization is already implemented, namely in the
    ``apply`` method.

    Consequently, using a reconstruction algorithm that derives from it boils down
    to four steps:

    # . Creating an instance of the reconstruction algorithm.
    # . Training the algorithm
    # . Setting the data.
    # . Applying the algorithm.


    """

    def __init__(self, psf, dtype=None, n_iter=1, **kwargs):
        """
        Base constructor. Derived constructor may define new state variables
        here and also reset them in `reset`.

        Parameters
        ----------

            psf : :py:class:`~torch.Tensor`
                Point spread function (PSF) that models forward propagation.
                2D (grayscale) or 3D (RGB) data can be provided and the shape will
                be used to determine which reconstruction (and allocate the
                appropriate memory).
            dtype : float32 or float64
                Data type to use for optimization.
            n_iter : int
                Number of iterations for unrolled algorithm.
        """
        assert isinstance(psf, torch.Tensor)
        self.is_torch = True

        self.n_iter = n_iter
        super(TrainableReconstructionAlgorithm, self).__init__(psf, dtype=dtype, n_iter=1, **kwargs)

    def batch_call(self, batch):
        result = []
        for image in batch:
            self.reset()
            self.set_data(image)
            result.append(self.apply(plot=False))
        return torch.stack(result, dim=0)

    def apply(self, disp_iter=10, plot_pause=0.2, plot=True, save=False, gamma=None, ax=None):
        """
        Method for performing iterative reconstruction. Contrary to none trainable reconstruction algorithm, the number of iteration isn't requiered. Note that `set_data`
        must be called beforehand.

        Parameters
        ----------
        disp_iter : int
            How often to display and/or intermediate reconstruction (in number
            of iterations). If `None` OR `plot` or `save` are False, no
            intermediate reconstruction will be plotted/saved.
        plot_pause : float
            Number of seconds to pause after displaying reconstruction.
        plot : bool
            Whether to plot final result, and intermediate results if
            `disp_iter` is not None.
        save : bool
            Whether to save final result (as PNG), and intermediate results if
            `disp_iter` is not None.
        gamma : float, optional
            Gamma correction factor to apply for plots. Default is None.
        ax : :py:class:`~matplotlib.axes.Axes`, optional
            `Axes` object to fill for plotting/saving, default is to create one.

        Returns
        -------
        final_im : :py:class:`~numpy.ndarray`
            Final reconstruction.
        ax : :py:class:`~matplotlib.axes.Axes`
            `Axes` object on which final reconstruction is displayed. Only
            returning if `plot` or `save` is True.

        """
        # apply expect H W C (This check should probably be moved to set_data)
        if self._data.shape[0] == 3:
            CHW = True
            self._data = self._data.permute(1, 2, 0)

        im = super(TrainableReconstructionAlgorithm, self).apply(
            n_iter=self.n_iter,
            disp_iter=10,
            plot_pause=0.2,
            plot=plot,
            save=save,
            gamma=gamma,
            ax=ax,
        )

        if CHW:
            if isinstance(im, tuple):
                im = im[0].permute(2, 0, 1), im[1]
            else :
                im = im.permute(2, 0, 1)
        return im
