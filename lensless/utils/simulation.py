# #############################################################################
# simulation.py
# =================
# Authors :
# Yohann PERRON [yohann.perron@gmail.com]
# #############################################################################

import numpy as np
from waveprop.simulation import FarFieldSimulator as FarFieldSimulator_wp


class FarFieldSimulator(FarFieldSimulator_wp):
    """
    LenslessPiCam-compatible wrapper for :py:class:`~waveprop.simulation.FarFieldSimulator` (source code on `GitHub <https://github.com/ebezzam/waveprop/blob/82dfb08b4db11c0c07ef00bdb59b5a769a49f0b3/waveprop/simulation.py#L11C11-L11C11>`__).
    """

    def __init__(
        self,
        object_height,
        scene2mask,
        mask2sensor,
        sensor,
        psf=None,
        output_dim=None,
        snr_db=None,
        max_val=255,
        device_conv="cpu",
        random_shift=False,
        is_torch=False,
        quantize=True,
        **kwargs
    ):
        """
        Parameters
        ----------
        psf : np.ndarray, optional.
            Point spread function. If not provided, return image at object plane.
        object_height : float or (float, float)
            Height of object in meters. Or range of values to randomly sample from.
        scene2mask : float
            Distance from scene to mask in meters.
        mask2sensor : float
            Distance from mask to sensor in meters.
        sensor : str
            Sensor name.
        snr_db : float, optional
            Signal-to-noise ratio in dB, by default None.
        max_val : int, optional
            Maximum value of image, by default 255.
        device_conv : str, optional
            Device to use for convolution (when using pytorch), by default "cpu".
        random_shift : bool, optional
            Whether to randomly shift the image, by default False.
        is_torch : bool, optional
            Whether to use pytorch, by default False.
        quantize : bool, optional
            Whether to quantize image, by default True.
        """

        if psf is not None:
            # convert HWC to CHW
            psf = psf.squeeze().movedim(-1, 0)

        super().__init__(
            object_height,
            scene2mask,
            mask2sensor,
            sensor,
            psf,
            output_dim,
            snr_db,
            max_val,
            device_conv,
            random_shift,
            is_torch,
            quantize,
            **kwargs
        )

        # save all the parameters in a dict
        self.params = {
            "object_height": object_height,
            "scene2mask": scene2mask,
            "mask2sensor": mask2sensor,
            "sensor": sensor,
            "output_dim": output_dim,
            "snr_db": snr_db,
            "max_val": max_val,
            "device_conv": device_conv,
            "random_shift": random_shift,
            "is_torch": is_torch,
            "quantize": quantize,
        }
        self.params.update(kwargs)

    def propagate(self, obj, return_object_plane=False):
        """
        Parameters
        ----------
        obj : np.ndarray or torch.Tensor
            Single image to propagate at format HWC.
        return_object_plane : bool, optional
            Whether to return object plane, by default False.
        """
        if self.is_torch:
            obj = obj.moveaxis(-1, 0)
            res = super().propagate(obj, return_object_plane)
            if isinstance(res, tuple):
                res = res[0].moveaxis(-3, -1), res[1].moveaxis(-3, -1)
            else:
                res = res.moveaxis(-3, -1)
            return res
        else:
            obj = np.moveaxis(obj, -1, 0)
            res = super().propagate(obj, return_object_plane)
            if isinstance(res, tuple):
                res = np.moveaxis(res[0], -3, -1), np.moveaxis(res[1], -3, -1)
            else:
                res = np.moveaxis(res, -3, -1)
            return res
