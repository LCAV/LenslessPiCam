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
class ``lensless.ReconstructionAlgorithm``. The three reconstruction
strategies available in ``LenslessPiCam`` derive from this class:

-  ``lensless.GradientDescient``: projected gradient descent with a
   non-negativity constraint. Two accelerated approaches are also
   available: ``lensless.NesterovGradientDescent`` and
   ``lensless.FISTA``.
-  ``lensless.ADMM``: alternating direction method of multipliers (ADMM)
   with a non-negativity constraint and a total variation (TV)
   regularizer [1]_.
-  ``lensless.APGD``: accelerated proximal gradient descent with Pycsou
   as a backend. Any differentiable or proximal operator can be used as
   long as it is compatible with Pycsou, namely derives from one of
   :py:class:`~pycsou.core.functional.DifferentiableFunctional` or
   :py:class:`~pycsou.core.functional.ProximableFunctional`.

New reconstruction algorithms can be conveniently implemented by
deriving from the abstract class and defining the following abstract
methods:

-  the update step: ``_update``.
-  a method to reset state variables: ``reset``.
-  an image formation method: ``_form_image``.

One advantage of deriving from ``lensless.ReconstructionAlgorithm`` is
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
from scipy.fftpack import next_fast_len
from lensless.plot import plot_image

try:
    import torch

    torch_available = True
except ImportError:
    torch_available = False


class ReconstructionAlgorithm(abc.ABC):
    """
    Abstract class for defining lensless imaging reconstruction algorithms.

    The following abstract methods need to be defined:

    * ``_update``: updating state variables at each iterations.
    * ``reset``: reset state variables.
    * ``_form_image``: any pre-processing that needs to be done in order to view the image estimate, e.g. reshaping or clipping.

    One advantage of deriving from this abstract class is that functionality for
    iterating, saving, and visualization is already implemented, namely in the
    ``apply`` method.

    Consequently, using a reconstruction algorithm that derives from it boils down
    to three steps:

    #. Creating an instance of the reconstruction algorithm.
    #. Setting the data.
    #. Applying the algorithm.


    """

    def __init__(self, psf, dtype=None):
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
        """
        self.is_torch = False
        if torch_available:
            self.is_torch = isinstance(psf, torch.Tensor)

        # prepate shapes for reconstruction
        self._is_rgb = len(psf.shape) == 3
        if self._is_rgb:
            self._psf = psf
            self._n_channels = 3
        else:
            self._psf = psf[:, :, None]
            self._n_channels = 1
        self._psf_shape = np.array(self._psf.shape)

        # set dtype
        if dtype is None:
            if self.is_torch:
                dtype = torch.float32
            else:
                dtype = np.float32

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

        # cropping / padding indices
        self._padded_shape = 2 * self._psf_shape[:2] - 1
        self._padded_shape = np.array([next_fast_len(i) for i in self._padded_shape])
        self._padded_shape = list(np.r_[self._padded_shape, [self._n_channels]])
        self._start_idx = (self._padded_shape[:2] - self._psf_shape[:2]) // 2
        self._end_idx = self._start_idx + self._psf_shape[:2]

        # pre-compute operators / outputs
        self._image_est = None
        self._data = None
        self.reset()

    @abc.abstractmethod
    def reset(self):
        """Reset state variables."""
        return

    @abc.abstractmethod
    def _update(self):
        """Update state variables."""
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

        if not self._is_rgb:
            assert len(data.shape) == 2
            data = data[:, :, None]
        assert len(self._psf_shape) == len(data.shape)
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
        self, n_iter=100, disp_iter=10, plot_pause=0.2, plot=True, save=False, gamma=None, ax=None
    ):
        """
        Method for performing iterative reconstruction. Note that `set_data`
        must be called beforehand.

        Parameters
        ----------
        n_iter : int
            Number of iterations.
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
