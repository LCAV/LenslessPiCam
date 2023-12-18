# #############################################################################
# trainable_mask.py
# ==================
# Authors :
# Yohann PERRON [yohann.perron@gmail.com]
# Eric BEZZAM [ebezzam@gmail.com]
# #############################################################################

import abc
import omegaconf
import os
import numpy as np
from hydra.utils import get_original_cwd
import torch
from lensless.utils.image import is_grayscale, rgb2gray
from lensless.hardware.slm import full2subpattern, get_programmable_mask, get_intensity_psf
from lensless.hardware.sensor import VirtualSensor
from waveprop.devices import slm_dict
from lensless.hardware.mask import CodedAperture, MultiLensArray, HeightVarying


class TrainableMask(torch.nn.Module, metaclass=abc.ABCMeta):
    """
    Abstract class for defining trainable masks.

    The following abstract methods need to be defined:

    - :py:class:`~lensless.hardware.trainable_mask.TrainableMask.get_psf`: returning the PSF of the mask.
    - :py:class:`~lensless.hardware.trainable_mask.TrainableMask.project`: projecting the mask parameters to a valid space (should be a subspace of [0,1]).

    """

    def __init__(self, optimizer="Adam", lr=1e-3, **kwargs):
        """
        Base constructor. Derived constructor may define new state variables

        Parameters
        ----------
        optimizer : str, optional
            Optimizer to use for updating the mask parameters, by default "Adam"
        lr : float, optional
            Learning rate for the mask parameters, by default 1e-3
        """
        super().__init__()
        # self._param = [torch.nn.Parameter(p, requires_grad=True) for p in initial_param]
        # # self._param = initial_param
        # self._optimizer = getattr(torch.optim, optimizer)(self._param, lr=lr)
        # self._counter = 0
        self._optimizer = optimizer
        self._lr = lr
        self._counter = 0

    def _set_optimizer(self, param):
        """Set the optimizer for the mask parameters."""
        self._optimizer = getattr(torch.optim, self._optimizer)(param, lr=self._lr)

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
    

class TrainableMultiLensArray(TrainableMask):

    def __init__(
        self, sensor_name, downsample=None, optimizer="Adam", lr=1e-3, torch_device="cuda", **kwargs
    ):

        # 1) call base constructor so parameters can be set
        super().__init__(optimizer, lr, **kwargs)

        # 2) initialize mask
        assert "distance_sensor" in kwargs, "Distance to sensor must be specified"
        assert "N" in kwargs, "Number of Lenses must be specified"
        self._mask_obj = MultiLensArray.from_sensor(sensor_name, downsample, is_torch=True, torch_device=torch_device, **kwargs)
        self._mask = self._mask_obj.mask

        # 3) set learnable parameters (should be immediate attributes of the class)
        self._radius = torch.nn.Parameter(self._mask_obj.radius)
        initial_param = [self._radius]

        # 4) set optimizer
        self._set_optimizer(initial_param)
        
        # 5) compute PSF
        self._psf = None
        self.project()

    def get_psf(self):
        return self._psf

    
    def project(self):
        with torch.no_grad():
            # clamp back the radiuses
            rad = self._radius.data
            rad = torch.clamp(rad, self._mask_obj.radius_range[0], self._mask_obj.radius_range[1])
            # sort in descending order
            rad, idx = torch.sort(rad, descending=True)
            loca = self._mask_obj.loc[idx]
            self._mask_obj.loc = loca

            circles = torch.cat((loca, rad.unsqueeze(-1)), dim=-1)
            for idx, r in enumerate(rad):
                min_loc = torch.min(loca[idx, 0], loca[idx, 1])
                rad[idx] = torch.clamp(r, 0, min_loc)
                # check for overlapping
                for (cx, cy, cr) in circles[idx+1:]:
                    dist = torch.sqrt((loca[idx, 0] - cx)**2 + (loca[idx, 1] - cy)**2)
                    if dist <= r + cr:
                        rad[idx] = dist - cr
                        circles[idx, 2] = rad[idx]
                    if rad[idx] < 0:
                        rad[idx] = 0
                        circles[idx, 2] = rad[idx]
                        break
            # update the parameters
            self._radius.data = rad

        # recompute PSF
        self._mask_obj.create_mask(self._radius)
        self._mask_obj.compute_psf()
        self._psf = self._mask_obj.psf.unsqueeze(0)
        self._psf = self._psf / self._psf.norm()

        
                     
class TrainableHeightVarying(TrainableMask):

    def __init__(
            self, sensor_name, downsample = None, optimizer="Adam", lr=1e-3, torch_device="cuda", **kwargs
    ):
        #1)
        super().__init__(optimizer, lr, **kwargs)

        #2)
        assert "distance_sensor" in kwargs, "Distance to sensor must be specified"
        self._mask_obj = HeightVarying.from_sensor(sensor_name, downsample, is_torch=True, torch_device=torch_device, **kwargs)
        self._mask = self._mask_obj.mask

        #3)
        self._height_map = torch.nn.Parameter(self._mask_obj.height_map)
        initial_param = [self._height_map]
        
        #4)
        self._set_optimizer(initial_param)
        
         # 5) compute PSF
        self._psf = None
        self.project()
        
    def get_psf(self):
        return self._psf

    def project(self):
        with torch.no_grad():
            # clamp back the heights between min_height, and max_height
            self._height_map.data = torch.clamp(self._height_map.data, self._mask_obj.height_range[0], self._mask_obj.height_range[1])
        self._mask_obj.create_mask(self._height_map)
        self._mask_obj.compute_psf()

        psf = self._mask_obj.psf.unsqueeze(0)
        self._psf = psf / psf.norm()
        

            

class TrainablePSF(TrainableMask):
    # class TrainablePSF(torch.nn.Module, TrainableMask):
    """
    Class for defining an object that directly optimizes the PSF, without any constraints on what can be realized physically.

    Parameters
    ----------
    grayscale : bool, optional
        Whether mask should be returned as grayscale when calling :py:class:`~lensless.hardware.trainable_mask.TrainableMask.get_psf`.
        Otherwise PSF will be returned as RGB. By default False.
    """

    def __init__(self, initial_psf, optimizer="Adam", lr=1e-3, grayscale=False, **kwargs):

        # BEFORE
        super().__init__(optimizer, lr, **kwargs)
        self._psf = torch.nn.Parameter(initial_psf)
        initial_param = [self._psf]
        self._set_optimizer(initial_param)

        # # cast as learnable parameters
        # super().__init__()
        # self._psf = torch.nn.Parameter(initial_psf)
        # self._optimizer = getattr(torch.optim, optimizer)([self._psf], lr=lr)
        # self._counter = 0

        # checks
        assert len(initial_psf.shape) == 4, "Mask must be of shape (depth, height, width, channels)"
        self.grayscale = grayscale
        self._is_grayscale = is_grayscale(initial_psf)
        if grayscale:
            assert self._is_grayscale, "PSF must be grayscale"

    def get_psf(self):
        if self._is_grayscale:
            if self.grayscale:
                # simulation in grayscale
                return self._psf
            else:
                # replicate to 3 channels
                return self._psf.expand(-1, -1, -1, 3)
        else:
            # assume RGB
            return self._psf

    def project(self):
        self._psf.data = torch.clamp(self._psf, 0, 1)


class AdafruitLCD(TrainableMask):
    # class AdafruitLCD(torch.nn.Module, TrainableMask):
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
        **kwargs,
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

        super().__init__(optimizer, lr, **kwargs)  # when using TrainableMask init
        # super().__init__()  # when using torch.nn.Module

        self.train_mask_vals = train_mask_vals
        if train_mask_vals:
            self._vals = torch.nn.Parameter(initial_vals)
        else:
            self._vals = initial_vals

        if color_filter is not None:
            self._color_filter = torch.nn.Parameter(color_filter)
            if train_mask_vals:
                initial_param = [self._vals, self._color_filter]
            else:
                initial_param = [self._color_filter]
        else:
            assert (
                train_mask_vals
            ), "If color filter is not trainable, mask values must be trainable"

        # set optimizer
        # self._optimizer = getattr(torch.optim, optimizer)(initial_param, lr=lr)
        # self._counter = 0
        self._set_optimizer(initial_param)

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
            vals=self._vals,
            sensor=self.sensor,
            slm_param=self.slm_param,
            rotate=self.rotate,
            flipud=self.flipud,
            color_filter=self._color_filter,
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
            self._vals.data = torch.clamp(self._vals, self.min_val, 1)
        if self._color_filter is not None:
            self._color_filter.data = torch.clamp(self._color_filter, 0, 1)
            # normalize each row to 1
            self._color_filter.data = self._color_filter / self._color_filter.sum(
                dim=[1, 2]
            ).unsqueeze(-1).unsqueeze(-1)


class TrainableCodedAperture(TrainableMask):
    def __init__(
        self,
        sensor_name,
        downsample=None,
        binary=True,
        torch_device="cuda",
        optimizer="Adam",
        lr=1e-3,
        **kwargs,
    ):
        """
        TODO: Distinguish between separable and non-separable.
        """

        # 1) call base constructor so parameters can be set
        super().__init__(optimizer, lr, **kwargs)

        # 2) initialize mask
        assert "distance_sensor" in kwargs, "Distance to sensor must be specified"
        assert "method" in kwargs, "Method must be specified."
        assert "n_bits" in kwargs, "Number of bits must be specified."
        # self._mask_obj = CodedAperture.from_sensor(sensor_name, downsample, is_torch=True, **kwargs)
        self._mask_obj = CodedAperture.from_sensor(
            sensor_name,
            downsample,
            is_torch=True,
            torch_device=torch_device,
            **kwargs,
        )
        self._mask = self._mask_obj.mask

        # 3) set learnable parameters (should be immediate attributes of the class)
        self._row = None
        self._col = None
        self._vals = None
        if self._mask_obj.row is not None:
            # seperable
            self.separable = True
            self._row = torch.nn.Parameter(self._mask_obj.row)
            self._col = torch.nn.Parameter(self._mask_obj.col)
            initial_param = [self._row, self._col]
        else:
            # non-seperable
            self.separable = False
            self._vals = torch.nn.Parameter(self._mask_obj.mask)
            initial_param = [self._vals]
        self.binary = binary

        # 4) set optimizer
        self._set_optimizer(initial_param)

        # 5) compute PSF
        self._psf = None
        self.project()

    def get_psf(self):
        # self._mask_obj.create_mask(self._row, self._col)
        # self._mask_obj.compute_psf()
        # psf = self._mask_obj.psf.unsqueeze(0)

        # # # need normalize the PSF? would think so but NAN comes up if included
        # # psf = psf / psf.norm()

        # return psf
        return self._psf

    def project(self):
        with torch.no_grad():
            if self.separable:
                self._row.data = torch.clamp(self._row, 0, 1)
                self._col.data = torch.clamp(self._col, 0, 1)
                if self.binary:
                    self._row.data = torch.round(self._row)
                    self._col.data = torch.round(self._col)
            else:
                self._vals.data = torch.clamp(self._vals, 0, 1)
                if self.binary:
                    self._vals.data = torch.round(self._vals)

        # recompute PSF
        self._mask_obj.create_mask(self._row, self._col, mask=self._vals)
        self._mask_obj.compute_psf()
        self._psf = self._mask_obj.psf.unsqueeze(0)
        self._psf = self._psf / self._psf.norm()


"""
Utilities to prepare trainable masks.
"""

trainable_mask_dict = {
    "AdafruitLCD": AdafruitLCD,
    "TrainablePSF": TrainablePSF,
    "TrainableCodedAperture": TrainableCodedAperture,
    "TrainableHeightVarying": TrainableHeightVarying,
    "TrainableMultiLensArray": TrainableMultiLensArray,
}


def prep_trainable_mask(config, psf=None, downsample=None):
    mask = None
    color_filter = None
    downsample = config.files.downsample if downsample is None else downsample
    if config.trainable_mask.mask_type is not None:

        assert config.trainable_mask.mask_type in trainable_mask_dict.keys(), (
            f"Trainable mask type {config.trainable_mask.mask_type} not supported. "
            f"Supported types are {trainable_mask_dict.keys()}"
        )
        mask_class = trainable_mask_dict[config.trainable_mask.mask_type]

        if isinstance(config.trainable_mask.initial_value, omegaconf.dictconfig.DictConfig):

            # from mask config
            mask = mask_class(
                # mask = TrainableCodedAperture(
                sensor_name=config.simulation.sensor,
                downsample=downsample,
                distance_sensor=config.simulation.mask2sensor,
                optimizer=config.trainable_mask.optimizer,
                lr=config.trainable_mask.mask_lr,
                binary=config.trainable_mask.binary,
                torch_device=config.torch_device,
                **config.trainable_mask.initial_value,
            )

        else:

            if config.trainable_mask.initial_value == "random":
                if psf is not None:
                    initial_mask = torch.rand_like(psf)
                else:
                    sensor = VirtualSensor.from_name(
                        config.simulation.sensor, downsample=downsample
                    )
                    resolution = sensor.resolution
                    initial_mask = torch.rand((1, *resolution, 3))
            elif config.trainable_mask.initial_value == "psf":
                initial_mask = psf.clone()
            # if file ending with "npy"
            elif config.trainable_mask.initial_value.endswith("npy"):
                pattern = np.load(
                    os.path.join(config.trainable_mask.initial_value) #TODO: get_original_cwd(), 
                )

                initial_mask = full2subpattern(
                    pattern=pattern,
                    shape=config.trainable_mask.ap_shape,
                    center=config.trainable_mask.ap_center,
                    slm=config.trainable_mask.slm,
                )
                initial_mask = torch.from_numpy(initial_mask.astype(np.float32))

                # prepare color filter if needed
                from waveprop.devices import slm_dict
                from waveprop.devices import SLMParam as SLMParam_wp

                slm_param = slm_dict[config.trainable_mask.slm]
                if (
                    config.trainable_mask.train_color_filter
                    and SLMParam_wp.COLOR_FILTER in slm_param.keys()
                ):
                    color_filter = slm_param[SLMParam_wp.COLOR_FILTER]
                    color_filter = torch.from_numpy(color_filter.copy()).to(dtype=torch.float32)

                    # add small random values
                    color_filter = color_filter + 0.1 * torch.rand_like(color_filter)

            else:
                raise ValueError(
                    f"Initial PSF value {config.trainable_mask.initial_value} not supported"
                )

            if config.trainable_mask.grayscale and not is_grayscale(initial_mask):
                initial_mask = rgb2gray(initial_mask)

            mask = mask_class(
                initial_mask,
                downsample=downsample,
                color_filter=color_filter,
                **config.trainable_mask,
            )

    return mask
