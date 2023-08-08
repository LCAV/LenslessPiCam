# #############################################################################
# trainable_recon.py
# ==================
# Authors :
# Yohann PERRON [yohann.perron@gmail.com]
# #############################################################################

import abc
import torch


class TrainableMask(metaclass=abc.ABCMeta):
    """
    Virtual class for defining trainable masks.
    The following abstract methods need to be defined:
    get_psf: getting the PSF of the mask from the mask parameter.
    project: projecting the mask parameters to a valid space (should be a subspace of [0,1]).
    """

    def __init__(self, initial_mask, optimizer="Adam", lr=1e-3, update_frequency=1, **kwargs):
        self._mask = torch.nn.Parameter(initial_mask)
        self._optimizer = getattr(torch.optim, optimizer)([self._mask], lr=lr, **kwargs)
        self._update_frequency = update_frequency
        self._counter = 0

    @abc.abstractmethod
    def get_psf(self):
        """Abstract method for getting the PSF of the mask. Should be fully compatible with pytorch autograd.

        Returns
        -------
        :py:class:`~torch.Tensor`
            The PSF of the mask.
        """
        raise NotImplementedError

    def update_mask(self):
        """Update the mask parameters. Acoording to externaly updated gradiants."""
        if self._counter % self._update_frequency == 0:
            self._optimizer.step()
            self._optimizer.zero_grad(set_to_none=True)
        self._counter += 1

    @abc.abstractmethod
    def project(self):
        """Abstract method for projecting the mask parameters to a valid space (should be a subspace of [0,1])."""
        raise NotImplementedError


class AmplitudeMask(TrainableMask):
    """
    Class for defining trainable amplitude masks.
    """

    def __init__(self, initial_mask, optimizer="Adam", lr=1e-3, update_frequency=1, **kwargs):
        print("Warning: AmplitudeMask is not fully implemented yet.")
        super().__init__(initial_mask, optimizer, lr, update_frequency, **kwargs)

    def get_psf(self):
        return self._mask

    def project(self):
        self._mask.data = torch.clamp(self._mask, 0, 1)
