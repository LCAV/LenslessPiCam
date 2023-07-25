# #############################################################################
# trainable_recon.py
# ==================
# Authors :
# Yohann PERRON [yohann.perron@gmail.com]
# #############################################################################

import abc
import pathlib as plib
from matplotlib import pyplot as plt
from lensless.recon.recon import ReconstructionAlgorithm
from lensless.utils.plot import plot_image

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
    * ``batch_call``: method for performing iterative reconstruction on a batch of images.

    One advantage of deriving from this abstract class is that functionality for
    iterating, saving, and visualization is already implemented, namely in the
    ``apply`` method.

    Consequently, using a reconstruction algorithm that derives from it boils down
    to four steps:

    1. Creating an instance of the reconstruction algorithm.
    2. Training the algorithm
    3. Setting the data.
    4. Applying the algorithm.


    """

    def __init__(
        self,
        psf,
        dtype=None,
        n_iter=1,
        pre_process=None,
        post_process=None,
        **kwargs,
    ):
        """
        Base constructor. Derived constructor may define new state variables
        here and also reset them in `reset`.

        Parameters
        ----------

            psf : :py:class:`~torch.Tensor`
                Point spread function (PSF) that models forward propagation.
                Must be of shape (depth, height, width, channels) even if
                depth = 1 and channels = 1. You can use :py:func:`~lensless.io.load_psf`
                to load a PSF from a file such that it is in the correct format.
            dtype : float32 or float64
                Data type to use for optimization.
            n_iter : int
                Number of iterations for unrolled algorithm.
            pre_process : :py:class:`function` or :py:class:`~torch.nn.Module`, optional
                If :py:class:`function` : Function to apply to the image estimate before algorithm. Its input most be (image to process, noise_level), where noise_level is a learnable parameter. If it include aditional learnable parameters, they will not be added to the parameter list of the algorithm. To allow for traning, the function must be autograd compatible.
                If :py:class:`~torch.nn.Module` : A DruNet compatible network to apply to the image estimate before algorithm. See ``utils.image.apply_denoiser`` for more details. The network will be included as a submodule of the algorithm and its parameters will be added to the parameter list of the algorithm. If this isn't intended behavior, set requires_grad=False.
            post_process : :py:class:`function` or :py:class:`~torch.nn.Module`, optional
                If :py:class:`function` : Function to apply to the image estimate after the whole algorithm. Its input most be (image to process, noise_level), where noise_level is a learnable parameter. If it include aditional learnable parameters, they will not be added to the parameter list of the algorithm. To allow for traning, the function must be autograd compatible.
                If :py:class:`~torch.nn.Module` : A DruNet compatible network to apply to the image estimate after the whole algorithm. See ``utils.image.apply_denoiser`` for more details. The network will be included as a submodule of the algorithm and its parameters will be added to the parameter list of the algorithm. If this isn't intended behavior, set requires_grad=False.
        """
        assert isinstance(psf, torch.Tensor), "PSF must be a torch.Tensor"
        super(TrainableReconstructionAlgorithm, self).__init__(
            psf, dtype=dtype, n_iter=n_iter, **kwargs
        )

        # pre processing
        (
            self.pre_process,
            self.pre_process_model,
            self.pre_process_param,
        ) = self._prepare_process_block(pre_process)

        # post processing
        (
            self.post_process,
            self.post_process_model,
            self.post_process_param,
        ) = self._prepare_process_block(post_process)

    def _prepare_process_block(self, process):
        """
        Method for preparing the pre or post process block.
        Parameters
        ----------
            process : :py:class:`function` or :py:class:`~torch.nn.Module`, optional
                Pre or post process block to prepare.
        """
        if isinstance(process, torch.nn.Module):
            # If the post_process is a torch module, we assume it is a DruNet like network.
            from lensless.utils.image import process_with_DruNet

            process_model = process
            process_function = process_with_DruNet(process_model, self._psf.device, mode="train")
        elif process is not None:
            # Otherwise, we assume it is a function.
            assert callable(process), "pre_process must be a callable function"
            process_function = process
            process_model = None
        else:
            process_function = None
            process_model = None
        if process_function is not None:
            process_param = torch.nn.Parameter(torch.tensor([1.0], device=self._psf.device))
        else:
            process_param = None

        return process_function, process_model, process_param

    def batch_call(self, batch):
        """
        Method for performing iterative reconstruction on a batch of images.
        This implementation is a properly vectorized implementation of FISTA.

        Parameters
        ----------
        batch : :py:class:`~torch.Tensor` of shape (batch, depth, channels, height, width)
            The lensless images to reconstruct.

        Returns
        -------
        :py:class:`~torch.Tensor` of shape (batch, depth, channels, height, width)
            The reconstructed images.
        """
        self._data = batch
        assert (
            len(self._data.shape) == 5
        ), f"batch must be of shape (N, D, C, H, W), current shape {self._data.shape}"
        batch_size = batch.shape[0]

        # pre process data
        if self.pre_process is not None:
            self._data = self.pre_process(self._data, self.pre_process_param)

        self.reset(batch_size=batch_size)

        for i in range(self._n_iter):
            self._update(i)

        image_est = self._form_image()

        if self.post_process is not None:
            image_est = self.post_process(image_est, self.post_process_param)

        return image_est

    def apply(
        self,
        disp_iter=10,
        plot_pause=0.2,
        plot=True,
        save=False,
        gamma=None,
        ax=None,
        reset=True,
        output_intermediate=False,
    ):
        """
        Method for performing iterative reconstruction. Contrary to non-trainable reconstruction
        algorithm, the number of iteration isn't required. Note that `set_data` must be called
        beforehand.

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
        output_intermediate : bool, optional
            Whether to output intermediate reconstructions after preprocessing and before postprocessing.

        Returns
        -------
        final_im : :py:class:`~torch.Tensor`
            Final reconstruction.
        ax : :py:class:`~matplotlib.axes.Axes`
            `Axes` object on which final reconstruction is displayed. Only
            returning if `plot` or `save` is True.

        """
        # pre process data
        pre_processed_image = None
        if self.pre_process is not None:
            self._data = self.pre_process(self._data, self.pre_process_param)
            if output_intermediate:
                pre_processed_image = self._data[0, ...].clone()

        im = super(TrainableReconstructionAlgorithm, self).apply(
            n_iter=self._n_iter,
            disp_iter=disp_iter,
            plot_pause=plot_pause,
            plot=plot,
            save=save,
            gamma=gamma,
            ax=ax,
            reset=reset,
        )

        # remove plot if returned
        if plot:
            im, _ = im

        # post process data
        pre_post_process_image = None
        if self.post_process is not None:
            # apply post process
            if output_intermediate:
                pre_post_process_image = im.clone()
            im = self.post_process(im, self.post_process_param)[0, ...]

        if plot:
            ax = plot_image(self._get_numpy_data(im), ax=ax, gamma=gamma)
            ax.set_title(
                "Final reconstruction after {} iterations and post process".format(self._n_iter)
            )
            if save:
                plt.savefig(plib.Path(save) / "final.png")

        if output_intermediate:
            return im, pre_processed_image, pre_post_process_image
        elif plot:
            return im, ax
        else:
            return im
