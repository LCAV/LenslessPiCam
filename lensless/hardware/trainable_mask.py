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
        self.train_mask_vals = True
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

    def get_vals(self):
        """Get the mask parameters."""
        return self._mask

    @abc.abstractmethod
    def project(self):
        """Abstract method for projecting the mask parameters to a valid space (should be a subspace of [0,1])."""
        raise NotImplementedError
    
    @classmethod
    def from_mask(cls, mask, **kwargs):
        return cls(initial_mask=mask, **kwargs)


class TrainableMultiLensArray(TrainableMask):

    def __init__(self, initial_mask, optimizer="Adam", lr=1e-3, **kwargs):
        super().__init__(initial_mask, optimizer, lr, **kwargs)
        self._loc = torch.nn.Parameter(self._mask.loc)
        self._radius = torch.nn.Parameter(self._mask.radius)

    
    def get_psf(self):
        self._mask.compute_psf()
        return self._mask.psf
    
    def project(self):
        # clamp back the radiuses
        torch.clamp(self._radius, 0, self._mask.size[0] / 2)

        # sort in descending order
        self._radius, idx = torch.sort(self._radius, descending=True)
        self._loc = self._loc[idx]

        circles = torch.cat((self._loc, self._radius.unsqueeze(-1)), dim=-1)
        for idx, r in enumerate(self._radius):
            # clamp back the locations
            torch.clamp(self._loc[idx, 0], r, self._mask.size[0] - r)
            torch.clamp(self._loc[idx, 1], r, self._mask.size[1] - r)

            # check for overlapping
            for (cx, cy, cr) in circles[idx+1:]:
                dist = torch.sqrt((self._loc[idx, 0] - cx)**2 + (self._loc[idx, 1] - cy)**2)
                if dist <= r + cr:
                    self._radius[idx] = dist - cr
                if self._radius[idx] < 0:
                    self._radius[idx] = 0
                    break
        
                     

            

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
        optimizer="Adam",
        lr=1e-3,
        train_mask_vals=True,
        color_filter=None,
        rotate=None,
        flipud=False,
        use_waveprop=None,
        vertical_shift=None,
        horizontal_shift=None,
        scene2mask=None,
        mask2sensor=None,
        downsample=None,
        min_val=0,
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
        self.train_mask_vals = train_mask_vals
        if color_filter is not None:
            self.color_filter = torch.nn.Parameter(color_filter)
            if train_mask_vals:
                param = [self._mask, self.color_filter]
            else:
                del self._mask
                self._mask = initial_vals
                param = [self.color_filter]
            self._optimizer = getattr(torch.optim, optimizer)(param, lr=lr)
        else:
            self.color_filter = None
            assert (
                train_mask_vals
            ), "If color filter is not trainable, mask values must be trainable"

        self.slm_param = slm_dict[slm]
        self.device = slm
        self.sensor = VirtualSensor.from_name(sensor, downsample=downsample)
        self.rotate = rotate
        self.flipud = flipud
        self.use_waveprop = use_waveprop
        self.scene2mask = scene2mask
        self.mask2sensor = mask2sensor
        self.vertical_shift = vertical_shift
        self.horizontal_shift = horizontal_shift
        self.min_val = min_val
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
            color_filter=self.color_filter,
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
        if self.train_mask_vals:
            self._mask.data = torch.clamp(self._mask, self.min_val, 1)
        if self.color_filter is not None:
            self.color_filter.data = torch.clamp(self.color_filter, 0, 1)
            # normalize each row to 1
            self.color_filter.data = self.color_filter / self.color_filter.sum(
                dim=[1, 2]
            ).unsqueeze(-1).unsqueeze(-1)
