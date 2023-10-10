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
from lensless.hardware.slm import get_programmable_mask, get_intensity_psf
from lensless.hardware.sensor import VirtualSensor
from waveprop.devices import slm_dict


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
        self._optimizer = getattr(torch.optim, optimizer)([self._mask], lr=lr)
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
        """Update the mask parameters. According to externaly updated gradiants."""
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


class AdafruitLCD(TrainableMask):
    def __init__(
        self,
        initial_vals,
        sensor,
        slm,
        rotate=None,
        flipud=False,
        use_waveprop=None,
        vertical_shift=None,
        horizontal_shift=None,
        scene2mask=None,
        mask2sensor=None,
        downsample=None,
        **kwargs
    ):
        """
        Parameters
        ----------
        initial_vals : :py:class:`~torch.Tensor`
            Initial mask parameters.
        sensor : :py:class:`~lensless.hardware.sensor.VirtualSensor`
            Sensor object.
        slm_param : :py:class:`~lensless.hardware.slm.SLMParam`
            SLM parameters.
        rotate : float, optional
            Rotation angle in degrees, by default None
        flipud : bool, optional
            Whether to flip the mask vertically, by default False
        """
        super().__init__(initial_vals, **kwargs)

        self.slm_param = slm_dict[slm]
        self.sensor = VirtualSensor.from_name(sensor, downsample=downsample)
        self.rotate = rotate
        self.flipud = flipud
        self.use_waveprop = use_waveprop
        self.scene2mask = scene2mask
        self.mask2sensor = mask2sensor
        self.vertical_shift = vertical_shift
        self.horizontal_shift = horizontal_shift
        if downsample is not None and vertical_shift is not None:
            self.vertical_shift = vertical_shift // downsample
        if downsample is not None and horizontal_shift is not None:
            self.horizontal_shift = horizontal_shift // downsample
        if self.use_waveprop:
            assert self.scene2mask is not None
            assert self.mask2sensor is not None

    def get_psf(self):

        mask = get_programmable_mask(
            vals=self._mask,
            sensor=self.sensor,
            slm_param=self.slm_param,
            rotate=self.rotate,
            flipud=self.flipud,
        )

        if self.vertical_shift is not None:
            mask = torch.roll(mask, self.vertical_shift, dims=1)

        if self.horizontal_shift is not None:
            mask = torch.roll(mask, self.horizontal_shift, dims=2)

        psf_in = get_intensity_psf(
            mask=mask,
            sensor=self.sensor,
            waveprop=self.use_waveprop,
            scene2mask=self.scene2mask,
            mask2sensor=self.mask2sensor,
        )

        # add first dimension (depth)
        psf_in = psf_in.unsqueeze(0)

        # move channels to last dimension
        psf_in = psf_in.permute(0, 2, 3, 1)

        # flip mask
        psf_in = torch.flip(psf_in, dims=[-3, -2])

        # normalize
        psf_in = psf_in / psf_in.norm()

        return psf_in

    def project(self):
        self._mask.data = torch.clamp(self._mask, 0, 1)
