import abc
import numpy as np
import pathlib as plib
import matplotlib.pyplot as plt
from scipy.fftpack import next_fast_len
from diffcam.plot import plot_image


class ReconstructionAlgorithm(abc.ABC):
    def __init__(self, psf, dtype=np.float32):

        self._is_rgb = True if len(psf.shape) == 3 else False
        if self._is_rgb:
            self._psf = psf
            self._n_channels = 3
        else:
            self._psf = psf[:, :, np.newaxis]
            self._n_channels = 1
        self._psf_shape = np.array(self._psf.shape)

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
        self._padded_shape = np.r_[self._padded_shape, [self._n_channels]]
        self._start_idx = (self._padded_shape[:2] - self._psf_shape[:2]) // 2
        self._end_idx = self._start_idx + self._psf_shape[:2]

        # pre-compute operators / outputs
        self._image_est = None
        self._data = None
        self.reset()

    @abc.abstractmethod
    def reset(self):
        return

    @abc.abstractmethod
    def _forward(self):
        return

    @abc.abstractmethod
    def _backward(self, x):
        return

    @abc.abstractmethod
    def _update(self):
        return

    @abc.abstractmethod
    def _form_image(self):
        return

    def set_data(self, data):
        if not self._is_rgb:
            assert len(data.shape) == 2
            data = data[:, :, np.newaxis]
        assert len(self._psf_shape) == len(data.shape)
        self._data = data

    def get_image_est(self):
        return self._form_image()

    def apply(
        self, n_iter=100, disp_iter=10, plot_pause=0.2, plot=True, save=False, gamma=None, ax=None
    ):
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
                ax = plot_image(self._form_image(), ax=ax, gamma=gamma)
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
