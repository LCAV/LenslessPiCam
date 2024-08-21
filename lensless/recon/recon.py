# #############################################################################
# recon.py
# =================
# Authors :
# Eric BEZZAM [ebezzam@gmail.com]
# #############################################################################

"""
Reconstruction
==============

Check out `this notebook <https://drive.google.com/file/d/1Wgt6ZMRZVuctLHaXxk7PEyPaBaUPvU33/view?usp=drive_link>`_
on Google Colab for an overview of the reconstruction algorithms available in LenslessPiCam (analytic and learned).

The core algorithmic component of ``LenslessPiCam`` is the abstract
class :py:class:`~lensless.ReconstructionAlgorithm`. The five reconstruction
strategies available in ``LenslessPiCam`` derive from this class:

-  :py:class:`~lensless.GradientDescent`: projected gradient descent with a
   non-negativity constraint. Two accelerated approaches are also
   available: :py:class:`~lensless.NesterovGradientDescent` and
   :py:class:`~lensless.FISTA`.
-  :py:class:`~lensless.ADMM`: alternating direction method of multipliers (ADMM)
   with a non-negativity constraint and a total variation (TV)
   regularizer [1]_.
-  :py:class:`~lensless.APGD`: accelerated proximal gradient descent with Pycsou
   as a backend. Any differentiable or proximal operator can be used as
   long as it is compatible with Pycsou, namely derives from one of
   `DiffFunc <https://github.com/matthieumeo/pycsou/blob/a74b714192821501371c89dbd44eac15a5456a0f/src/pycsou/abc/operator.py#L980>`_
   or `ProxFunc <https://github.com/matthieumeo/pycsou/blob/a74b714192821501371c89dbd44eac15a5456a0f/src/pycsou/abc/operator.py#L741>`_.
-  :py:class:`~lensless.UnrolledFISTA`: unrolled FISTA with a non-negativity constraint.
-  :py:class:`~lensless.UnrolledADMM`: unrolled ADMM with a non-negativity constraint and a total variation (TV) regularizer [1]_.

Note that the unrolled algorithms derive from the abstract class
:py:class:`~lensless.TrainableReconstructionAlgorithm`, which itself derives from
:py:class:`~lensless.ReconstructionAlgorithm` while adding functionality
for training on batches and adding trainable pre- and post-processing
blocks.

New reconstruction algorithms can be conveniently implemented by
deriving from the abstract class and defining the following abstract
methods:

-  the update step: :py:class:`~lensless.ReconstructionAlgorithm._update`.
-  a method to reset state variables: :py:class:`~lensless.ReconstructionAlgorithm.reset`.
-  an image formation method: :py:class:`~lensless.ReconstructionAlgorithm._form_image`.

One advantage of deriving from :py:class:`~lensless.ReconstructionAlgorithm` is
that functionality for iterating, saving, and visualization is already
implemented. Consequently, using a reconstruction algorithm that derives
from it boils down to three steps:

1. Creating an instance of the reconstruction algorithm.
2. Setting the data.
3. Applying the algorithm.


2D example (ADMM)
-----------------

For example, for ADMM:

.. code:: python

       recon = ADMM(psf)
       recon.set_data(data)
       res = recon.apply(n_iter=n_iter)

A full running example can found in ``scripts/recon/admm.py`` and run as:

.. code:: bash

    python scripts/recon/admm.py

Note that a YAML configuration script is defined in ``configs/admm_thumbs_up.yaml``,
which is used by default. Individual parameters can be configured as such:

.. code:: bash

    python scripts/recon/admm.py admm.n_iter=10 preprocess.gray=True

``--help`` can be used to view all available parameters.

.. code::

    >> python scripts/recon/admm.py --help

    ...

    == Config ==
    Override anything in the config (foo.bar=value)

    files:
        psf: data/psf/tape_rgb.png
        data: data/raw_data/thumbs_up_rgb.png
    preprocess:
        downsample: 4
        shape: null
        flip: false
        bayer: false
        blue_gain: null
        red_gain: null
        single_psf: false
        gray: false
    display:
        disp: 1
        plot: true
        gamma: null
    save: false
    admm:
        n_iter: 5
        mu1: 1.0e-06
        mu2: 1.0e-05
        mu3: 4.0e-05
        tau: 0.0001

    ...


Alternatively, a new configuration file can be defined in the ``configs`` folder and
passed to the script:

.. code:: bash

    python scripts/recon/admm.py -cn <CONFIG_FILENAME_WITHOUT_YAML_EXT>


3D example
----------

It is also possible to reconstruct 3D scenes using :py:class:`~lensless.GradientDescent` or :py:class:`~lensless.APGD`. :py:class:`~lensless.ADMM` does not support 3D reconstruction yet.
This requires to use a 3D PSF as an input in the form of an ``.npy`` or ``.npz`` file, which is a set of 2D PSFs corresponding to the same diffuser sampled with light sources at different depths.
The input data for 3D reconstructions is still a 2D image, as collected by the camera. The reconstruction will be able to separate which part of the lensless data corresponds to which 2D PSF,
and therefore to which depth, effectively generating a 3D reconstruction, which will be outputed in the form of an ``.npy`` file. A 2D projection on the depth axis is also displayed to the user.

The same scripts for 2D reconstruction can be used for 3D reconstruction, namely ``scripts/recon/gradient_descent.py`` and ``scripts/recon/apgd_pycsou.py``.

3D data is provided in LenslessPiCam, but it is simulated. Real example data can be obtained from `Waller Lab <https://github.com/Waller-Lab/DiffuserCam/tree/master/example_data>`_.
For both the simulated data and the data from Waller Lab, it is best to set ``downsample=1``:

.. code:: bash

    python scripts/recon/gradient_descent.py \\
    input.psf="path/to/3D/psf.npy" \\
    input.data="path/to/lensless/data.tiff" \\
    preprocess.downsample=1


Other approaches
----------------

Scripts for other reconstruction algorithms can be found in ``scripts/recon`` and their
corresponding configurations in ``configs``.


**References**

.. [1] Boyd, S., Parikh, N., & Chu, E. (2011). Distributed optimization and statistical learning via the alternating direction method of multipliers. Now Publishers Inc.

"""


import abc
import numpy as np
import pathlib as plib
import matplotlib.pyplot as plt
from lensless.utils.plot import plot_image
from lensless.utils.io import get_dtype
from lensless.recon.rfft_convolve import RealFFTConvolve2D

try:
    import torch

    torch_available = True
except ImportError:
    torch_available = False


class ReconstructionAlgorithm(abc.ABC):
    """
    Abstract class for defining lensless imaging reconstruction algorithms.

    The following abstract methods need to be defined:

    * :py:class:`~lensless.ReconstructionAlgorithm._update`: updating state variables at each iterations.
    * :py:class:`~lensless.ReconstructionAlgorithm.reset`: reset state variables.
    * :py:class:`~lensless.ReconstructionAlgorithm._form_image`: any pre-processing that needs to be done in order to view the image estimate, e.g. reshaping or clipping.

    One advantage of deriving from this abstract class is that functionality for
    iterating, saving, and visualization is already implemented, namely in the
    ``apply`` method.

    Consequently, using a reconstruction algorithm that derives from it boils down
    to three steps:

    #. Creating an instance of the reconstruction algorithm.
    #. Setting the data.
    #. Applying the algorithm.


    """

    def __init__(
        self,
        psf,
        dtype=None,
        pad=True,
        n_iter=100,
        initial_est=None,
        reset=True,
        denoiser=None,
        **kwargs,
    ):
        """
        Base constructor. Derived constructor may define new state variables
        here and also reset them in `reset`.

        Parameters
        ----------

            psf : :py:class:`~numpy.ndarray` or :py:class:`~torch.Tensor`
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
            n_iter : int, optional
                Number of iterations to run algorithm for. Can be overridden in `apply`.
            initial_est : :py:class:`~numpy.ndarray` or :py:class:`~torch.Tensor`, optional
                Initial estimate of the image. If not provided, the initial estimate is
                set to zero or to the mean of the data, depending on the algorithm.
            reset : bool, optional
                Whether to reset state variables in the base constructor. Defaults to True.
                If False, you should call reset() at one point to initialize state variables.
            denoiser : dict, optional
                Dictionary defining a denoiser for plug-and-play. Must contain the following keys:

                * ``"network"``: model to use as a denoiser.
                * ``"noise_level"``: noise level of the denoiser.

                If provided, the denoiser will be used as a projection function at each iteration.
                Defaults to None.
        """
        super().__init__()
        self.is_torch = False

        if torch_available:
            self.is_torch = isinstance(psf, torch.Tensor)

        assert len(psf.shape) == 4, "PSF must be 4D: (depth, height, width, channels)."
        assert psf.shape[3] == 3 or psf.shape[3] == 1, "PSF must either be rgb (3) or grayscale (1)"
        self._psf = psf
        self._npix = np.prod(self._psf.shape)
        self._n_iter = n_iter

        self._psf_shape = np.array(self._psf.shape)

        # set dtype
        dtype = get_dtype(dtype, self.is_torch)

        if self.is_torch:
            if dtype:
                self._psf = self._psf.type(dtype)
                self._dtype = dtype
            else:
                self._dtype = self._psf.dtype
            if self._dtype == torch.float32 or dtype == "float32":
                self._complex_dtype = torch.complex64
            elif self._dtype == torch.float64 or dtype == "float64":
                self._complex_dtype = torch.complex128
            else:
                raise ValueError(f"Unsupported dtype : {self._dtype}")

        else:
            if dtype:
                self._psf = self._psf.astype(dtype)
                self._dtype = dtype
            else:
                self._dtype = self._psf.dtype
            if self._dtype == np.float32 or dtype == "float32":
                self._complex_dtype = np.complex64
            elif self._dtype == np.float64 or dtype == "float64":
                self._complex_dtype = np.complex128
            else:
                raise ValueError(f"Unsupported dtype : {self._dtype}")

        self._convolver_param = {"dtype": dtype, "pad": pad, **kwargs}
        self._convolver = RealFFTConvolve2D(psf, dtype=dtype, pad=pad, **kwargs)
        self._padded_shape = self._convolver._padded_shape

        if pad:
            self._image_est_shape = self._psf_shape
        else:
            self._image_est_shape = self._convolver._padded_shape

        # pre-compute operators / outputs / set estimates
        if initial_est is not None:
            self._set_initial_estimate(initial_est)
        else:
            self._initial_est = None
        self._data = None

        self._denoiser = None
        if denoiser is not None:
            assert self.is_torch
            assert "network" in denoiser.keys()
            assert "noise_level" in denoiser.keys()

            from lensless.recon.utils import get_drunet_function_v2, load_drunet

            device = self._psf.device
            if denoiser["network"] == "DruNet":
                denoiser_model = load_drunet(requires_grad=False).to(device)
                self._denoiser = get_drunet_function_v2(denoiser_model, mode="inference")
            else:
                raise NotImplementedError(f"Unsupported denoiser: {denoiser['network']}")
            self._denoiser_noise_level = denoiser["noise_level"]

        # used inside trainable recon
        self.compensation_branch = None
        self.compensation_branch_inputs = None

        if reset:
            self.reset()

    @abc.abstractmethod
    def reset(self):
        """
        Reset state variables.
        """
        return

    @abc.abstractmethod
    def _update(self, iter):
        """
        Update state variables.
        """
        return

    @abc.abstractmethod
    def _form_image(self):
        """
        Any pre-processing to form a viewable image, e.g. reshaping or clipping.
        """
        return

    def set_data(self, data):
        """
        Set lensless data for recontruction.

        Parameters
        ----------
        data : :py:class:`~numpy.ndarray`
            Lensless data on which to iterate to recover an estimate of the
            scene. Should match provide PSF, i.e. shape and 2D (grayscale) or
            3D (RGB).
        """

        if self.is_torch:
            assert isinstance(data, torch.Tensor)
        else:
            assert isinstance(data, np.ndarray)

        assert len(data.shape) >= 3, "Data must be at least 3D: [..., width, height, channel]."

        # assert same shapes
        assert np.all(
            self._psf_shape[-3:-1] == np.array(data.shape)[-3:-1]
        ), "PSF and data shape mismatch"

        if len(data.shape) == 3:
            self._data = data[None, None, ...]
        elif len(data.shape) == 4:
            self._data = data[None, ...]
        else:
            self._data = data

    def _set_initial_estimate(self, image_est):
        """
        Set initial estimate of image, e.g. to warm-start algorithm.

        Note that reset() should be called after setting the initial estimate
        so that it is taken into account.

        Parameters
        ----------
        image_est : :py:class:`~numpy.ndarray` or :py:class:`~torch.Tensor`
            Initial estimate of the image. Should match provide PSF shape.
        """

        if self.is_torch:
            assert isinstance(image_est, torch.Tensor)
        else:
            assert isinstance(image_est, np.ndarray)

        assert (
            len(image_est.shape) >= 4
        ), "Image estimate must be at least 4D: [..., depth, width, height, channel]."

        # assert same shapes
        assert np.all(
            self._image_est_shape[-3:-1] == np.array(image_est.shape)[-3:-1]
        ), f"Image estimate must be of shape (..., width, height, channel): {self._image_est_shape[-3:-1]}"

        if len(image_est.shape) == 4:
            self._initial_est = image_est[None, ...]
        else:
            self._initial_est = image_est

    def set_image_estimate(self, image_est):
        """
        Set initial estimate of image, e.g. to warm-start algorithm.

        Parameters
        ----------
        image_est : :py:class:`~numpy.ndarray` or :py:class:`~torch.Tensor`
            Initial estimate of the image. Should match provide PSF shape.
        """

        if self.is_torch:
            assert isinstance(image_est, torch.Tensor)
        else:
            assert isinstance(image_est, np.ndarray)

        assert (
            len(image_est.shape) >= 4
        ), "Image estimate must be at least 4D: [..., depth, width, height, channel]."

        # assert same shapes
        assert np.all(
            self._image_est_shape[-3:-1] == np.array(image_est.shape)[-3:-1]
        ), f"Image estimate must be of shape (..., width, height, channel): {self._image_est_shape[-3:-1]}"

        if len(image_est.shape) == 4:
            self._image_est = image_est[None, ...]
        else:
            self._image_est = image_est

    def get_image_estimate(self):
        """Get current image estimate as [Batch, Depth, Height, Width, Channels]."""
        return self._form_image()

    def _set_psf(self, psf):
        """
        Set PSF.

        Parameters
        ----------
        psf : :py:class:`~numpy.ndarray` or :py:class:`~torch.Tensor`
            PSF to set.
        """
        assert (
            psf.shape[-1] == 3 or psf.shape[-1] == 1
        ), "PSF must either be rgb (3) or grayscale (1)"
        assert self._psf.shape == psf.shape, "new PSF must have same shape as old PSF"
        assert isinstance(psf, type(self._psf)), "new PSF must have same type as old PSF"

        self._psf = psf
        self._convolver = RealFFTConvolve2D(
            psf,
            dtype=self._convolver._psf.dtype,
            pad=self._convolver.pad,
            norm=self._convolver.norm,
        )
        self.reset()

    def _progress(self):
        """
        Optional method for printing progress update, e.g. relative improvement
        in reconstruction.
        """
        return

    def _get_numpy_data(self, data):
        """
        Extract data from torch or numpy array.

        Parameters
        ----------
        data : :py:class:`~numpy.ndarray` or :py:class:`~torch.Tensor`
            Data to extract.

        Returns
        -------
        :py:class:`~numpy.ndarray`
            Extracted data.
        """
        if self.is_torch:
            return data.detach().cpu().numpy()
        else:
            return data

    def apply(
        self,
        n_iter=None,
        disp_iter=-1,
        plot_pause=0.2,
        plot=False,
        save=False,
        gamma=None,
        ax=None,
        reset=True,
        **kwargs,
    ):
        """
        Method for performing iterative reconstruction. Note that `set_data`
        must be called beforehand.

        Parameters
        ----------
        n_iter : int, optional
            Number of iterations. If not provided, default to `self._n_iter`.
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
        reset : bool, optional
            Whether to reset state variables before applying reconstruction. Default to True.
            Set to false if continuing reconstruction from previous state.

        Returns
        -------
        final_im : :py:class:`~numpy.ndarray`
            Final reconstruction.
        ax : :py:class:`~matplotlib.axes.Axes`
            `Axes` object on which final reconstruction is displayed. Only
            returning if `plot` or `save` is True.

        """
        assert self._data is not None, "Must set data with `set_data()`"
        assert (
            self._data.shape[0] == 1
        ), "Apply doesn't supports processing multiple images at once."

        if reset:
            self.reset()

        if n_iter is None:
            n_iter = self._n_iter

        if (plot or save) and disp_iter is not None:
            if ax is None:
                img = self._form_image()
                ax = plot_image(self._get_numpy_data(img[0]), gamma=gamma)

        else:
            ax = None
            disp_iter = n_iter + 1

        if self.compensation_branch is not None:
            self.compensation_branch_inputs = [self._data]

        for i in range(n_iter):
            self._update(i)
            if self.compensation_branch is not None and i < self._n_iter - 1:
                self.compensation_branch_inputs.append(self._form_image())

            if (plot or save) and (i + 1) % disp_iter == 0:
                self._progress()
                img = self._form_image()
                ax = plot_image(self._get_numpy_data(img[0]), ax=ax, gamma=gamma)
                if hasattr(ax, "__len__"):
                    ax[0, 0].set_title("Reconstruction after iteration {}".format(i + 1))
                else:
                    ax.set_title("Reconstruction after iteration {}".format(i + 1))
                if save:
                    plt.savefig(plib.Path(save) / f"{i + 1}.png")
                if plot:
                    plt.draw()
                    plt.pause(plot_pause)

        final_im = self._form_image()[0]
        if plot:
            ax = plot_image(self._get_numpy_data(final_im), ax=ax, gamma=gamma)
            if hasattr(ax, "__len__"):
                ax[0, 0].set_title("Final reconstruction after {} iterations".format(n_iter))
            else:
                ax.set_title("Final reconstruction after {} iterations".format(n_iter))
            if save:
                plt.savefig(plib.Path(save) / f"{n_iter}.png")
            return final_im, ax
        else:
            return final_im

    def reconstruction_error(self, prediction=None, lensless=None):
        """
        Compute reconstruction error.

        Parameters
        ----------
        prediction :  :py:class:`~numpy.ndarray` or :py:class:`~torch.Tensor`, optional
            Reconstructed image. If None, use current estimate, default None.
        lensless : :py:class:`~numpy.ndarray` or :py:class:`~torch.Tensor`, optional
            Lensless image. If None, use data provided by `set_data()`, default None.

        Returns
        -------
        _type_
            _description_
        """
        # default to current estimate and data if not provided
        if prediction is None:
            prediction = self.get_image_estimate()
        if lensless is None:
            lensless = self._data

        # convolver = self._convolver
        convolver = RealFFTConvolve2D(self._psf.to(prediction.device), **self._convolver_param)
        if not convolver.pad:
            prediction = convolver._pad(prediction)
        Hx = convolver.convolve(prediction)

        if not convolver.pad:
            Hx = convolver._crop(Hx)

        # don't reduce batch dimension
        if self.is_torch:
            return torch.sum(torch.sqrt((Hx - lensless) ** 2), dim=(-1, -2, -3, -4)) / self._npix

        else:
            return np.sum(np.sqrt((Hx - lensless) ** 2), axis=(-1, -2, -3, -4)) / self._npix
