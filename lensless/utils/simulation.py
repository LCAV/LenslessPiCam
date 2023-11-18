# #############################################################################
# simulation.py
# =================
# Authors :
# Yohann PERRON [yohann.perron@gmail.com]
# Eric BEZZAM [ebezzam@gmail.com]
# #############################################################################

from waveprop.simulation import FarFieldSimulator as FarFieldSimulator_wp
import torch


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
        psf : np.ndarray or torch.Tensor, optional.
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
            assert len(psf.shape) == 4, "PSF must be of shape (depth, height, width, channels)"

            if torch.is_tensor(psf):
                # drop depth dimension, and convert HWC to CHW
                psf = psf[0].movedim(-1, 0)
                assert psf.shape[0] == 1 or psf.shape[0] == 3, "PSF must have 1 or 3 channels"
            else:
                psf = psf[0]
                assert psf.shape[-1] == 1 or psf.shape[-1] == 3, "PSF must have 1 or 3 channels"

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

        if psf is not None:
            if self.is_torch:
                assert (
                    self.psf.shape[0] == 1 or self.psf.shape[0] == 3
                ), "PSF must have 1 or 3 channels"
            else:
                assert (
                    self.psf.shape[-1] == 1 or self.psf.shape[-1] == 3
                ), "PSF must have 1 or 3 channels"

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

    def get_psf(self):
        if self.is_torch:
            # convert CHW to HWC
            return self.psf.movedim(0, -1).unsqueeze(0)
        else:
            return self.psf[None, ...]

    # needs different name from parent class
    def set_point_spread_function(self, psf):
        """
        Set point spread function.

        Parameters
        ----------
        psf : np.ndarray or torch.Tensor
            Point spread function.
        """
        assert len(psf.shape) == 4, "PSF must be of shape (depth, height, width, channels)"

        if torch.is_tensor(psf):
            # convert HWC to CHW
            psf = psf[0].movedim(-1, 0)
            assert psf.shape[0] == 1 or psf.shape[0] == 3, "PSF must have 1 or 3 channels"
        else:
            psf = psf[0]
            assert psf.shape[-1] == 1 or psf.shape[-1] == 3, "PSF must have 1 or 3 channels"

        return super().set_psf(psf)

    def propagate_image(self, obj, return_object_plane=False):
        """
        Parameters
        ----------
        obj : np.ndarray or torch.Tensor
            Single image to propagate of format HWC.
        return_object_plane : bool, optional
            Whether to return object plane, by default False.
        """

        assert obj.shape[-1] == 1 or obj.shape[-1] == 3, "Image must have 1 or 3 channels"

        if self.is_torch:
            # channel in first dimension as expected by waveprop for pytorch
            obj = obj.moveaxis(-1, 0)
            res = super().propagate(obj, return_object_plane)
            if isinstance(res, tuple):
                res = res[0].moveaxis(-3, -1), res[1].moveaxis(-3, -1)
            else:
                res = res.moveaxis(-3, -1)
            return res
        else:
            # TODO: not tested, but normally don't need to move dimensions for numpy
            res = super().propagate(obj, return_object_plane)
            return res
