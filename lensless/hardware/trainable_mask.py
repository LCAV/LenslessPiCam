# #############################################################################
# trainable_mask.py
# ==================
# Authors :
# Yohann PERRON [yohann.perron@gmail.com]
# Eric BEZZAM [ebezzam@gmail.com]
# #############################################################################

import abc
import torch
from lensless.utils.image import is_grayscale


class TrainableMask(torch.nn.Module, metaclass=abc.ABCMeta):
    """
    Abstract class for defining trainable masks.

    The following abstract methods need to be defined:

    - :py:class:`~lensless.hardware.trainable_mask.TrainableMask.get_psf`: returning the PSF of the mask.
    - :py:class:`~lensless.hardware.trainable_mask.TrainableMask.project`: projecting the mask parameters to a valid space (should be a subspace of [0,1]).

    """

    def __init__(self, initial_mask, optimizer="Adam", lr=1e-3, **kwargs):
        """
        Base constructor. Derived constructor may define new state variables

        Parameters
        ----------
        initial_mask : :py:class:`~torch.Tensor`
            Initial mask parameters.
        optimizer : str, optional
            Optimizer to use for updating the mask parameters, by default "Adam"
        lr : float, optional
            Learning rate for the mask parameters, by default 1e-3
        """
        super().__init__()
        self._mask = torch.nn.Parameter(initial_mask)
        self._optimizer = getattr(torch.optim, optimizer)([self._mask], lr=lr, **kwargs)
        self._counter = 0

    @abc.abstractmethod
    def get_psf(self):
        """
        Abstract method for getting the PSF of the mask. Should be fully compatible with pytorch autograd.

        Returns
        -------
        :py:class:`~torch.Tensor`
            The PSF of the mask.
        """
        raise NotImplementedError

    def update_mask(self):
        """Update the mask parameters. Acoording to externaly updated gradiants."""
        self._optimizer.step()
        self._optimizer.zero_grad(set_to_none=True)
        self.project()
        self._counter += 1

    @abc.abstractmethod
    def project(self):
        """Abstract method for projecting the mask parameters to a valid space (should be a subspace of [0,1])."""
        raise NotImplementedError


class TrainablePSF(TrainableMask):
    """
    Class for defining an object that directly optimizes the PSF, without any constraints on what can be realized physically.

    Parameters
    ----------
    grayscale : bool, optional
        Whether mask should be returned as grayscale when calling :py:class:`~lensless.hardware.trainable_mask.TrainableMask.get_psf`.
        Otherwise PSF will be returned as RGB. By default False.
    """

    def __init__(self, initial_mask, optimizer="Adam", lr=1e-3, grayscale=False, **kwargs):
        super().__init__(initial_mask, optimizer, lr, **kwargs)
        assert (
            len(initial_mask.shape) == 4
        ), "Mask must be of shape (depth, height, width, channels)"
        self.grayscale = grayscale
        self._is_grayscale = is_grayscale(initial_mask)
        if grayscale:
            assert self._is_grayscale, "Mask must be grayscale"

    def get_psf(self):
        if self._is_grayscale:
            if self.grayscale:
                # simulation in grayscale
                return self._mask
            else:
                # replicate to 3 channels
                return self._mask.expand(-1, -1, -1, 3)
        else:
            # assume RGB
            return self._mask

    def project(self):
        self._mask.data = torch.clamp(self._mask, 0, 1)
