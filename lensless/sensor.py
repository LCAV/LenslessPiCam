# #############################################################################
# sensor.py
# =================
# Authors :
# Eric BEZZAM [ebezzam@gmail.com]
# #############################################################################


"""
Sensor
======

TODO
- couple with lens / mask / aperture to get virtual image

"""


import numpy as np
from enum import Enum
from cv2 import resize

from lensless.util import rgb2gray
from lensless.io import load_image


class SensorOptions(Enum):
    RPI_HQ = "rpi_hq"
    RPI_GS = "rpi_gs"
    RPI_V2 = "rpi_v2"
    BASLER_287 = "basler_287"
    BASLER_548 = "basler_548"

    @staticmethod
    def values():
        return [dev.value for dev in SensorOptions]


class SensorParam:
    PIXEL_SIZE = "pixel_size"
    RESOLUTION = "resolution"
    DIAGONAL = "diagonal"
    COLOR = "color"
    BIT_DEPTH = "bit_depth"


"""
Note sensors are in landscape orientation.
"""
sensor_dict = {
    # Raspberry Pi HQ Camera Sensor
    # datasheet: https://www.arducam.com/sony/imx477/#imx477-datasheet
    # IMX 477
    SensorOptions.RPI_HQ.value: {
        SensorParam.PIXEL_SIZE: np.array([1.55e-6, 1.55e-6]),
        SensorParam.RESOLUTION: np.array([3040, 4056]),
        SensorParam.DIAGONAL: 7.857e-3,
        SensorParam.COLOR: True,
        SensorParam.BIT_DEPTH: [8, 12],
    },
    # Raspberry Pi Global Shutter Camera
    # https://www.raspberrypi.com/products/raspberry-pi-global-shutter-camera/
    # IMX 296
    SensorOptions.RPI_GS.value: {
        SensorParam.PIXEL_SIZE: np.array([3.45e-6, 3.45e-6]),
        SensorParam.RESOLUTION: np.array([1088, 1456]),
        SensorParam.DIAGONAL: 6.3e-3,
        SensorParam.COLOR: True,
        SensorParam.BIT_DEPTH: [8, 12],
    },
    # Raspberry Pi Camera Module V2
    # https://www.raspberrypi.com/documentation/accessories/camera.html#hardware-specification
    # IMX 219
    SensorOptions.RPI_V2.value: {
        SensorParam.PIXEL_SIZE: np.array([1.12e-6, 1.12e-6]),
        SensorParam.RESOLUTION: np.array([2464, 3280]),
        SensorParam.DIAGONAL: 4.6e-3,
        SensorParam.COLOR: True,
        SensorParam.BIT_DEPTH: [8],
    },
    # Basler daA720-520um
    # https://www.baslerweb.com/en/products/cameras/area-scan-cameras/dart/daa720-520um-cs-mount/
    # IMX 287
    SensorOptions.BASLER_287.value: {
        SensorParam.PIXEL_SIZE: np.array([6.9e-6, 6.9e-6]),
        SensorParam.RESOLUTION: np.array([540, 720]),
        # SensorParam.DIAGONAL: ,
        SensorParam.COLOR: False,
        SensorParam.BIT_DEPTH: [8, 12],
    },
    # Basler daA2448-70uc
    # https://www.baslerweb.com/en/products/cameras/area-scan-cameras/dart/daa2448-70uc-cs-mount/
    # IMX 548
    SensorOptions.BASLER_548.value: {
        SensorParam.PIXEL_SIZE: np.array([2.74e-6, 2.74e-6]),
        SensorParam.RESOLUTION: np.array([2048, 2448]),
        # 8.8 in other sources: https://www.gophotonics.com/products/cmos-image-sensors/sony-corporation/21-209-imx548?utm_source=gophotonics&utm_medium=similar
        # 8.7 in Basler docs but leads to error in shape...
        SensorParam.DIAGONAL: 8.8e-3,
        SensorParam.COLOR: True,
        SensorParam.BIT_DEPTH: [8, 10, 12],
    },
}


class VirtualSensor(object):
    """
    Generic sensor class.

    """

    def __init__(self, pixel_size, resolution, diagonal=None, color=True, bit_depth=None):

        self.pixel_size = pixel_size
        self.resolution = resolution
        self.diagonal = diagonal
        self.color = color
        if bit_depth is None:
            self.bit_depth = [8]
        else:
            self.bit_depth = bit_depth

        if diagonal is not None:
            # account for possible deadspace
            self.size = self.diagonal / np.linalg.norm(self.resolution) * self.resolution
        else:
            self.size = self.pixel_size * self.resolution

        self.image_shape = self.resolution
        if self.color:
            self.image_shape = np.append(self.image_shape, 3)

    # contructor from sensor name
    @classmethod
    def from_name(cls, name):
        """
        Create a sensor from a name.

        Parameters
        ----------
        name : str
            Name of the sensor.

        Returns
        -------
        sensor : :py:class:`~lensless.sensor.Sensor`
            Sensor.

        """
        if name not in SensorOptions.values():
            raise ValueError(f"Sensor {name} not supported.")
        return cls(**sensor_dict[name])

    def capture(self, scene=None, bit_depth=None, bayer=False):
        """
        Virtual capture of a scene (assuming perfectly focused lens).

        TODO: for simulated rescale appropriately; for real, display on screen

        Parameters
        ----------
        scene : :py:class:`~numpy.ndarray`, str, optional
            Scene to capture.
        bit_depth : int, optional
            Bit depth of the image. By default, use first available.
        bayer : bool, optional
            Whether to return a Bayer image or not. By default, return RGB (if color).

        Returns
        -------
        img : :py:class:`~numpy.ndarray`
            Captured image.

        """

        if bayer:
            raise NotImplementedError("Bayer capture not implemented yet.")

        if scene is None:
            scene = np.random.rand(*self.image_shape)
        else:

            if isinstance(scene, str):
                scene = load_image(scene)
            else:
                # check provided data has good shape
                if len(scene.shape) == 3:
                    assert scene.shape[2] == 3
                else:
                    assert len(scene.shape) == 2

            # rescale and keep aspect ratio
            scale = np.min(np.array(self.resolution) / np.array(scene.shape[:2]))
            dsize = tuple((np.array(scene.shape[:2]) * scale).astype(int))
            scene = resize(scene, dsize=dsize[::-1])
            diff = np.array(self.resolution) - np.array(scene.shape[:2])

            # pad if necessary
            if np.any(diff):

                # center padding
                pad_width = (
                    (diff[0] // 2, diff[0] - diff[0] // 2),
                    (diff[1] // 2, diff[1] - diff[1] // 2),
                )
                if len(scene.shape) == 3:
                    pad_width = pad_width + ((0, 0),)
                scene = np.pad(scene, pad_width, mode="constant", constant_values=0)

        assert scene is not None

        # convert to grayscale if necessary
        if not self.color:
            if len(scene.shape) == 3:
                scene = rgb2gray(scene, keepchanneldim=False)

        else:
            if len(scene.shape) == 2:
                # repeat channels
                scene = np.repeat(scene[:, :, np.newaxis], 3, axis=2)

        # normalize
        scene = scene.astype(np.float32)
        scene /= scene.max()

        # cast to appropriate bit depth
        if bit_depth is None:
            bit_depth = self.bit_depth[0]
        else:
            if bit_depth not in self.bit_depth:
                raise ValueError(f"Bit depth {bit_depth} not supported.")

        scene = (2**bit_depth - 1) * scene
        if bit_depth == 8:
            scene = scene.astype(np.uint8)
        elif bit_depth > 8:
            scene = scene.astype(np.uint16)

        return scene
