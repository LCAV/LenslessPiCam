import numpy as np
from pycsou.core.linop import LinearOperator
from typing import Union, Optional
from numbers import Number


class RealFFTConvolve2D(LinearOperator):
    def __init__(self, filter, dtype: Optional[type] = None):
        """
        Linear operator that performs convolution in Fourier domain, and assumes
        real-valued signals.

        Parameters
        ----------
        filter :py:class:`~numpy.ndarray`
            2D filter to use. Must be of shape (height, width, channels) even if
            only one channel.
        dtype : float32 or float64
            Data type to use for optimization.
        """

        assert len(filter.shape) == 3

        # precompute filter in frequency domain
        raise NotImplementedError

        # call parent class
        super(RealFFTConvolve2D, self).__init__(shape=shape, dtype=dtype)

    def __call__(self, x: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        # like here: https://github.com/PyLops/pylops/blob/3e7eb22a62ec60e868ccdd03bc4b54806851cb26/pylops/signalprocessing/ConvolveND.py#L103
        raise NotImplementedError

    def adjoint(self, y: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        raise NotImplementedError
