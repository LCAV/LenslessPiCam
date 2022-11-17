import abc
import warnings

import numpy as np
import pathlib as plib
import matplotlib.pyplot as plt
from scipy.fftpack import next_fast_len
from lensless.plot import plot_image


class ReconstructionAlgorithm(abc.ABC):
    """
    Abstract class for defining lensless imaging reconstruction algorithms.

    The following abstract methods need to be defined:
    - `_update`: updating state variables at each iterations.
    - `reset`: reset state variables.
    - `_form_image`: any pre-processing that needs to be done in order to view
    the image estimate, e.g. reshaping or clipping.

    One advantage of deriving from this abstract class is that functionality for
    iterating, saving, and visualization is already implemented, namely in the
    `apply` method.

    Consequently, using a reconstruction algorithm that derives from it boils down
    to three steps:

    1. Creating an instance of the reconstruction algorithm.
    2. Setting the data.
    3. Applying the algorithm.

    For example, for ADMM (full example in `scripts/admm.py`):
    ```python
        recon = ADMM(psf)
        recon.set_data(data)
        res = recon.apply(n_iter=n_iter)
    ```

    A template for applying a reconstruction algorithm (including loading the
    data) can be found in `scripts/reconstruction_template.py`.

    """

    def __init__(self, psf, dtype=np.float32, apgd=False):
        """
        Base constructor. Derived constructor may define new state variables
        here and also reset them in `reset`.

        Parameters
        ----------
        psf : :py:class:`~numpy.ndarray`
            Point spread function (PSF) that models forward propagation.
            2D (grayscale) or 3D (RGB) data can be provided and the shape will
            be used to determine which reconstruction (and allocate the
            appropriate memory).
        dtype : float32 or float64
            Data type to use for optimization.
        """

        if apgd:
            assert len(psf.shape) == 2
            self._is_rgb = False
            self._use_3d = False
        else:
            assert len(psf.shape) == 4
            self._use_3d = psf.shape[0] != 1
            self._is_rgb = psf.shape[3] == 3
            assert self._is_rgb or psf.shape[3] == 1

        self._psf = psf
        self._psf_shape = np.array(psf.shape)

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
        if apgd:
            self._padded_shape = 2 * self._psf_shape - 1
            self._padded_shape = np.array([next_fast_len(i) for i in self._padded_shape])
            self._padded_shape = np.r_[self._padded_shape]
            self._start_idx = (self._padded_shape - self._psf_shape) // 2
            self._end_idx = self._start_idx + self._psf_shape
        else:
            self._padded_shape = 2 * self._psf_shape[1:3] - 1
            self._padded_shape = np.array([next_fast_len(i) for i in self._padded_shape])
            self._padded_shape = np.r_[self._psf_shape[0], self._padded_shape, self._psf_shape[3]]
            self._start_idx = (self._padded_shape[1:3] - self._psf_shape[1:3]) // 2
            self._end_idx = self._start_idx + self._psf_shape[1:3]

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
        assert len(self._psf_shape) == len(data.shape) == 4
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
                ax = plot_image(self._data, gamma=gamma)
        else:
            ax = None
            disp_iter = n_iter + 1

        for i in range(n_iter):
            self._update()

            if (plot or save) and (i + 1) % disp_iter == 0:
                self._progress()
                img = self._form_image()
                ax = plot_image(img, ax=ax, gamma=gamma)
                ax.set_title("Reconstruction after iteration {}".format(i + 1))
                if save:
                    plt.savefig(plib.Path(save) / f"{i + 1}.png")
                if plot:
                    plt.draw()
                    plt.pause(plot_pause)

        final_im = self._form_image()
        if plot or save:
            ax = plot_image(final_im, ax=ax, gamma=gamma)
            ax.set_title("Final reconstruction after {} iterations".format(n_iter))
            if save:
                plt.savefig(plib.Path(save) / f"{n_iter}.png")
            return final_im, ax
        else:
            return final_im
