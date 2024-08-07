# #############################################################################
# trainable_recon.py
# ==================
# Authors :
# Yohann PERRON [yohann.perron@gmail.com]
# Eric BEZZAM [ebezzam@gmail.com]
# #############################################################################

import pathlib as plib
from matplotlib import pyplot as plt
from lensless.recon.recon import ReconstructionAlgorithm
from lensless.utils.plot import plot_image
from lensless.recon.rfft_convolve import RealFFTConvolve2D

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
        skip_unrolled=False,
        skip_pre=False,
        skip_post=False,
        return_intermediate=False,
        legacy_denoiser=False,
        compensation=None,
        compensation_residual=True,
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
        skip_unrolled : bool, optional
            Whether to skip the unrolled algorithm and only apply the pre- or post-processor block (e.g. to just use a U-Net for reconstruction).
        return_unrolled_output : bool, optional
            Whether to return the output of the unrolled algorithm if also using a post-processor block.
        compensation : list, optional
            Number of channels for each intermediate output in compensation layer, as in "Robust Reconstruction With Deep Learning to Handle Model Mismatch in Lensless Imaging" (2021).
            Post-processor must be defined if compensation provided.
        compensation_residual : bool, optional
            Whether to use residual connection in compensation layer.
        """

        assert isinstance(psf, torch.Tensor), "PSF must be a torch.Tensor"
        super(TrainableReconstructionAlgorithm, self).__init__(
            psf, dtype=dtype, n_iter=n_iter, **kwargs
        )

        self._legacy_denoiser = legacy_denoiser
        self.set_pre_process(pre_process)
        self.set_post_process(post_process)
        self.skip_unrolled = skip_unrolled
        self.skip_pre = skip_pre
        self.skip_post = skip_post
        self.return_intermediate = return_intermediate
        self.compensation_branch = compensation
        if compensation is not None:
            from lensless.recon.utils import CompensationBranch

            assert (
                post_process is not None
            ), "If compensation_branch is True, post_process must be defined."
            assert (
                len(compensation) == n_iter
            ), "compensation_nc must have the same length as n_iter"
            self.compensation_branch = CompensationBranch(
                compensation, residual=compensation_residual
            )
            self.compensation_branch = self.compensation_branch.to(self._psf.device)

        if self.return_intermediate:
            assert (
                post_process is not None or pre_process is not None
            ), "If return_intermediate is True, post_process or pre_process must be defined."
        if self.skip_unrolled:
            assert (
                post_process is not None or pre_process is not None
            ), "If skip_unrolled is True, pre_process or post_process must be defined."

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
            from lensless.recon.utils import get_drunet_function, get_drunet_function_v2

            process_model = process
            if self._legacy_denoiser:
                process_function = get_drunet_function(process_model, mode="train")
            else:
                process_function = get_drunet_function_v2(process_model, mode="train")
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

    def set_pre_process(self, pre_process):
        (
            self.pre_process,
            self.pre_process_model,
            self.pre_process_param,
        ) = self._prepare_process_block(pre_process)

    def set_post_process(self, post_process):
        (
            self.post_process,
            self.post_process_model,
            self.post_process_param,
        ) = self._prepare_process_block(post_process)

    def freeze_pre_process(self):
        """
        Method for freezing the pre process block.
        """
        if self.pre_process_param is not None:
            self.pre_process_param.requires_grad = False
        if self.pre_process_model is not None:
            for param in self.pre_process_model.parameters():
                param.requires_grad = False

    def freeze_post_process(self):
        """
        Method for freezing the post process block.
        """
        if self.post_process_param is not None:
            self.post_process_param.requires_grad = False
        if self.post_process_model is not None:
            for param in self.post_process_model.parameters():
                param.requires_grad = False

    def unfreeze_pre_process(self):
        """
        Method for unfreezing the pre process block.
        """
        if self.pre_process_param is not None:
            self.pre_process_param.requires_grad = True
        if self.pre_process_model is not None:
            for param in self.pre_process_model.parameters():
                param.requires_grad = True

    def unfreeze_post_process(self):
        """
        Method for unfreezing the post process block.
        """
        if self.post_process_param is not None:
            self.post_process_param.requires_grad = True
        if self.post_process_model is not None:
            for param in self.post_process_model.parameters():
                param.requires_grad = True

    def forward(self, batch, psfs=None):
        """
        Method for performing iterative reconstruction on a batch of images.
        This implementation is a properly vectorized implementation of FISTA.

        Parameters
        ----------
        batch : :py:class:`~torch.Tensor` of shape (batch, depth, channels, height, width)
            The lensless images to reconstruct.
        psfs : :py:class:`~torch.Tensor` of shape (batch, depth, channels, height, width)
            The lensless images to reconstruct.

        Returns
        -------
        :py:class:`~torch.Tensor` of shape (batch, depth, channels, height, width)
            The reconstructed images.
        """
        self._data = batch
        assert len(self._data.shape) == 5, "batch must be of shape (N, D, C, H, W)"
        batch_size = batch.shape[0]

        if psfs is not None:
            # assert same shape
            assert psfs.shape == batch.shape, "psfs must have the same shape as batch"
            # -- update convolver
            self._convolver = RealFFTConvolve2D(psfs.to(self._data.device), **self._convolver_param)
        elif self._data.device != self._convolver._H.device:
            # need for multi-GPU... TODO better solution?
            self._convolver = RealFFTConvolve2D(
                self._psf.to(self._data.device), **self._convolver_param
            )

        # pre process data
        if self.pre_process is not None and not self.skip_pre:
            device_before = self._data.device
            self._data = self.pre_process(self._data, self.pre_process_param)
            self._data = self._data.to(device_before)
        pre_processed = self._data

        self.reset(batch_size=batch_size)

        # unrolled algorithm
        if not self.skip_unrolled:
            if self.compensation_branch is not None:
                compensation_branch_inputs = [self._data]

            for i in range(self._n_iter):
                self._update(i)
                if self.compensation_branch is not None and i < self._n_iter - 1:
                    compensation_branch_inputs.append(self._form_image())

            image_est = self._form_image()
        else:
            image_est = self._data

        # post process data
        if self.post_process is not None and not self.skip_post:
            compensation_output = None
            if self.compensation_branch is not None:
                compensation_output = self.compensation_branch(compensation_branch_inputs)

            final_est = self.post_process(image_est, self.post_process_param, compensation_output)
        else:
            final_est = image_est

        if self.return_intermediate:
            return final_est, image_est, pre_processed
        else:
            return final_est

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
        pre_processed_image = None
        if self.pre_process is not None and not self.skip_pre:
            self._data = self.pre_process(self._data, self.pre_process_param)
            if output_intermediate:
                pre_processed_image = self._data[0, ...].clone()

        if not self.skip_unrolled:
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
        else:
            im = self._data

        # remove plot if returned
        if plot:
            im, _ = im

        # post process data
        pre_post_process_image = None
        if self.post_process is not None and not self.skip_post:

            compensation_output = None
            if self.compensation_branch is not None:
                compensation_output = self.compensation_branch(self.compensation_branch_inputs)

            # apply post process
            if output_intermediate:
                pre_post_process_image = im.clone()
            im = self.post_process(im, self.post_process_param, compensation_output)

        if plot:
            ax = plot_image(self._get_numpy_data(im[0]), ax=ax, gamma=gamma)
            ax.set_title(
                "Final reconstruction after {} iterations and post process".format(self._n_iter)
            )
            if save:
                plt.savefig(plib.Path(save) / "final.png")

        if output_intermediate:
            return im, pre_post_process_image, pre_processed_image
        elif plot:
            return im, ax
        else:
            return im
