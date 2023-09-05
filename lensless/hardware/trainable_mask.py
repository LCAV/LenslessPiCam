# #############################################################################
# trainable_mask.py
# ==================
# Authors :
# Yohann PERRON [yohann.perron@gmail.com]
# #############################################################################

import abc
import torch


class TrainableMask(metaclass=abc.ABCMeta):
    """
    Abstract class for defining trainable masks.

    The following abstract methods need to be defined:
    1. get_psf: getting the PSF of the mask from the mask parameter.
    2. project: projecting the mask parameters to a valid space (should be a subspace of [0,1]).
    """

    def __init__(self, initial_mask, optimizer="Adam", lr=1e-3, **kwargs):
        """
        Base constructor. Derived constructor may define new state variables

        Parameters
        ----------
        initial_mask : ``torch.Tensor``
            Initial mask parameters.
        optimizer : str, optional
            Optimizer to use for updating the mask parameters, by default "Adam"
        lr : float, optional
            Learning rate for the mask parameters, by default 1e-3
        """
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
    """

    def __init__(self, initial_mask, is_rgb=True, optimizer="Adam", lr=1e-3, **kwargs):
        super().__init__(initial_mask, optimizer, lr, **kwargs)
        self._is_rgb = is_rgb

    def get_psf(self):
        if self._is_rgb:
            return self._mask.expand(-1, -1, -1, 3)
        else:
            return self._mask

    def project(self):
        self._mask.data = torch.clamp(self._mask, 0, 1)
