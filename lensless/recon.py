# #############################################################################
# reconstruction.py
# =================
# Authors :
# Eric BEZZAM [ebezzam@gmail.com]
# #############################################################################

"""
Reconstruction
==============

The core algorithmic component of ``LenslessPiCam`` is the abstract
class :py:class:`~lensless.ReconstructionAlgorithm`. The three reconstruction
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


ADMM example
------------

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

It is also possible to reconstruct 3D scenes using Gradient Descent or APGD. ADMM doesn't supports 3D reconstruction yet.
This requires to use a 3D PSF as an input in the form of a .npy file, which actually is a set of 2D PSFs corresponding to the same diffuser sampeled with light sources from different depths.
The input data  for 3D reconstructions is still a 2D image, as collected by the camera. The reconstruction will be able to separate which part of the lensless data corresponds to which 2D PSF,
and therefore to which depth, effectively generating a 3D reconstruction, which will be outputed in the form of a .npy file as well as a 2D projection on the depth axis to be displayed to the
user as an image.

As for the 2D ADMM reconstuction, scripts for 3D reconstruction can be found in ``scripts/recon/gradient_descent.py`` and ``scripts/recon/apgd_pycsou.py``.
Outside of the input data and PSF, no special argument has to be given to the script for it to operate a 3D reconstruction, as actually, the 2D reconstuction is internally
viewed as a 3D reconstruction which has only one depth level. It is also the case for ADMM although for now, the reconstructions are wrong when more than one depth level is used.

3D data is not directly provided in the LenslessPiCam, but some can be :doc:`imported <data>` from the Waller Lab dataset. For this data, it is best to set the downsample to 1 :

.. code:: bash

    python scripts/recon/gradient_descent.py input.psf="path/to/3D/psf.npy" input.data="path/to/lensless/data.tiff" preprocess.downsample=1


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
from lensless.plot import plot_image
from lensless.rfft_convolve import RealFFTConvolve2D

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

    def __init__(self, psf, dtype=None, pad=True, n_iter=100, **kwargs):
        """
        Base constructor. Derived constructor may define new state variables
        here and also reset them in `reset`.

        Parameters
        ----------

            psf : :py:class:`~numpy.ndarray` or :py:class:`~torch.Tensor`
                Point spread function (PSF) that models forward propagation.
                2D (grayscale) or 3D (RGB) data can be provided and the shape will
                be used to determine which reconstruction (and allocate the
                appropriate memory).
            dtype : float32 or float64
                Data type to use for optimization.
            pad : bool, optional
                Whether to pad the PSF to avoid spatial aliasing.
            n_iter : int, optional
                Number of iterations to run algorithm for. Can be overridden in
                `apply`.
        """
        self.is_torch = False

        if torch_available:
            self.is_torch = isinstance(psf, torch.Tensor)

        assert len(psf.shape) == 4  # depth, width, height, channel
        assert psf.shape[3] == 3 or psf.shape[3] == 1  # either rgb or grayscale
        self._psf = psf
        self._n_iter = n_iter

        self._psf_shape = np.array(self._psf.shape)

        # set dtype
        if dtype is None:
            if self.is_torch:
                dtype = torch.float32
            else:
                dtype = np.float32
        else:
            if self.is_torch:
                dtype = torch.float32 if dtype == "float32" else torch.float64
            else:
                dtype = np.float32 if dtype == "float32" else np.float64

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

        self._convolver = RealFFTConvolve2D(psf, dtype=dtype, pad=pad, **kwargs)
        self._padded_shape = self._convolver._padded_shape

        # pre-compute operators / outputs
        self._image_est = None
        self._data = None
        self.reset()

    @abc.abstractmethod
    def reset(self):
        """
        Reset state variables.
        """
        return

    @abc.abstractmethod
    def _update(self):
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

        assert len(data.shape) == 4
        assert len(self._psf_shape) == 4

        # assert same shapes
        assert np.all(
            self._psf_shape[1:3] == np.array(data.shape)[1:3]
        ), "PSF and data shape mismatch"

        self._data = data

    def get_image_est(self):
        """Get current image estimate."""
        return self._form_image()

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
        self, n_iter=None, disp_iter=10, plot_pause=0.2, plot=True, save=False, gamma=None, ax=None
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

        Returns
        -------
        final_im : :py:class:`~numpy.ndarray`
            Final reconstruction.
        ax : :py:class:`~matplotlib.axes.Axes`
            `Axes` object on which final reconstruction is displayed. Only
            returning if `plot` or `save` is True.

        """
        assert self._data is not None, "Must set data with `set_data()`"

        if n_iter is None:
            n_iter = self._n_iter

        if (plot or save) and disp_iter is not None:
            if ax is None:
                ax = plot_image(self._get_numpy_data(self._data), gamma=gamma)
        else:
            ax = None
            disp_iter = n_iter + 1

        for i in range(n_iter):
            self._update()

            if (plot or save) and (i + 1) % disp_iter == 0:
                self._progress()
                img = self._form_image()
                ax = plot_image(self._get_numpy_data(img), ax=ax, gamma=gamma)
                ax.set_title("Reconstruction after iteration {}".format(i + 1))
                if save:
                    plt.savefig(plib.Path(save) / f"{i + 1}.png")
                if plot:
                    plt.draw()
                    plt.pause(plot_pause)

        final_im = self._form_image()
        if plot or save:
            ax = plot_image(self._get_numpy_data(final_im), ax=ax, gamma=gamma)
            ax.set_title("Final reconstruction after {} iterations".format(n_iter))
            if save:
                plt.savefig(plib.Path(save) / f"{n_iter}.png")
            return final_im, ax
        else:
            return final_im
