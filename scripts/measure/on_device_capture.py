"""
Capture raw Bayer data or post-processed RGB data.

```
python scripts/measure/on_device_capture.py legacy=True \
exp=0.02 bayer=True
```

With the Global Shutter sensor, legacy RPi software is not supported.
```
python scripts/measure/on_device_capture.py sensor=rpi_gs \
legacy=False exp=0.02 bayer=True
```

To capture PNG data (bayer=False) and downsample (by factor 2):
```
python scripts/measure/on_device_capture.py sensor=rpi_gs \
legacy=False exp=0.02 bayer=False down=2
```

See these code snippets for setting camera settings and post-processing
- https://github.com/scivision/pibayer/blob/1bb258c0f3f8571d6ded5491c0635686b59a5c4f/pibayer/base.py#L56
- https://picamera.readthedocs.io/en/release-1.13/recipes1.html#capturing-consistent-images
- https://www.strollswithmydog.com/open-raspberry-pi-high-quality-camera-raw

"""

import hydra
import os
import cv2
import numpy as np
from time import sleep
from PIL import Image
from lensless.hardware.utils import get_distro
from lensless.utils.image import bayer2rgb_cc, rgb2gray, resize
from lensless.hardware.constants import RPI_HQ_CAMERA_CCM_MATRIX, RPI_HQ_CAMERA_BLACK_LEVEL
from lensless.hardware.sensor import SensorOptions, sensor_dict, SensorParam
from fractions import Fraction
import time


SENSOR_MODES = [
    "off",
    "auto",
    "sunlight",
    "cloudy",
    "shade",
    "tungsten",
    "fluorescent",
    "incandescent",
    "flash",
    "horizon",
]


@hydra.main(version_base=None, config_path="../../configs", config_name="capture")
def capture(config):

    sensor = config.sensor
    assert sensor in SensorOptions.values(), f"Sensor must be one of {SensorOptions.values()}"

    bayer = config.bayer
    fn = config.fn
    exp = config.exp
    config_pause = config.config_pause
    sensor_mode = config.sensor_mode
    rgb = config.rgb
    gray = config.gray
    iso = config.iso
    sixteen = config.sixteen
    legacy = config.legacy
    down = config.down
    res = config.res
    nbits_out = config.nbits_out

    assert (
        nbits_out in sensor_dict[sensor][SensorParam.BIT_DEPTH]
    ), f"nbits_out must be one of {sensor_dict[sensor][SensorParam.BIT_DEPTH]} for sensor {sensor}"

    # https://www.raspberrypi.com/documentation/accessories/camera.html#hardware-specification
    sensor_param = sensor_dict[sensor]
    assert exp <= sensor_param[SensorParam.MAX_EXPOSURE]
    assert exp >= sensor_param[SensorParam.MIN_EXPOSURE]
    sensor_mode = int(sensor_mode)

    distro = get_distro()
    print("RPi distribution : {}".format(distro))

    if sensor == SensorOptions.RPI_GS.value:
        assert not legacy

    if "bullseye" in distro and not legacy:
        # TODO : grayscale and downsample
        assert not rgb
        assert not gray

        import subprocess

        if bayer:

            assert down is None

            # https://www.raspberrypi.com/documentation/computers/camera_software.html#raw-image-capture
            jpg_fn = fn + ".jpg"
            fn += ".dng"
            pic_command = [
                "libcamera-still",
                "-r",
                "--gain",
                f"{iso / 100}",
                "--shutter",
                f"{int(exp * 1e6)}",
                "-o",
                f"{jpg_fn}",
                # long exposure: https://www.raspberrypi.com/documentation/computers/camera_software.html#very-long-exposures
                # -- setting awbgains caused issues
                # "--awbgains 1,1",
                # "--immediate"
            ]

            cmd = subprocess.Popen(
                pic_command,
                shell=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            cmd.stdout.readlines()
            cmd.stderr.readlines()
            # os.remove(jpg_fn)
            os.system(f"exiftool {fn}")
            print("\nJPG saved to : {}".format(jpg_fn))
            # print("\nDNG saved to : {}".format(fn))

        else:

            from picamera2 import Picamera2, Preview

            picam2 = Picamera2()
            picam2.start_preview(Preview.NULL)

            fn += ".png"

            max_res = picam2.camera_properties["PixelArraySize"]
            if res:
                assert len(res) == 2
            else:
                res = np.array(max_res)
                if down is not None:
                    res = (np.array(res) / down).astype(int)

            res = tuple(res)
            print("Capturing at resolution: ", res)

            # capture low-dim PNG
            picam2.preview_configuration.main.size = res
            picam2.still_configuration.size = res
            picam2.still_configuration.enable_raw()
            picam2.still_configuration.raw.size = res

            # setting camera parameters
            picam2.configure(picam2.create_preview_configuration())
            new_controls = {
                "ExposureTime": int(exp * 1e6),
                "AnalogueGain": 1.0,
            }
            if config.awb_gains is not None:
                assert len(config.awb_gains) == 2
                new_controls["ColourGains"] = tuple(config.awb_gains)
            picam2.set_controls(new_controls)

            # take picture
            picam2.start("preview", show_preview=False)
            time.sleep(config.config_pause)

            picam2.switch_mode_and_capture_file("still", fn)

    # legacy camera software
    else:
        import picamerax.array

        fn += ".png"

        if bayer:

            camera = picamerax.PiCamera(framerate=1 / exp, sensor_mode=sensor_mode, resolution=res)

            # camera settings, as little processing as possible
            camera.iso = iso
            camera.shutter_speed = int(exp * 1e6)
            camera.exposure_mode = "off"
            camera.drc_strength = "off"
            camera.image_denoise = False
            camera.image_effect = "none"
            camera.still_stats = False

            sleep(config_pause)
            awb_gains = camera.awb_gains
            camera.awb_mode = "off"
            camera.awb_gains = awb_gains

            print("Resolution : {}".format(camera.resolution))
            print("Shutter speed : {}".format(camera.shutter_speed))
            print("ISO : {}".format(camera.iso))
            print("Frame rate : {}".format(camera.framerate))
            print("Sensor mode : {}".format(SENSOR_MODES[sensor_mode]))
            # keep this as it needs to be parsed from remote script!
            red_gain = float(awb_gains[0])
            blue_gain = float(awb_gains[1])
            print("Red gain : {}".format(red_gain))
            print("Blue gain : {}".format(blue_gain))

            # capture data
            stream = picamerax.array.PiBayerArray(camera)
            camera.capture(stream, "jpeg", bayer=True)

            # get bayer data
            if sixteen:
                output = np.sum(stream.array, axis=2).astype(np.uint16)
            else:
                output = (np.sum(stream.array, axis=2) >> 2).astype(np.uint8)

            # returning non-bayer data
            if rgb or gray:
                if sixteen:
                    n_bits = 12  # assuming Raspberry Pi HQ
                else:
                    n_bits = 8

                if config.awb_gains is not None:
                    red_gain = config.awb_gains[0]
                    blue_gain = config.awb_gains[1]

                output_rgb = bayer2rgb_cc(
                    output,
                    nbits=n_bits,
                    blue_gain=blue_gain,
                    red_gain=red_gain,
                    black_level=RPI_HQ_CAMERA_BLACK_LEVEL,
                    ccm=RPI_HQ_CAMERA_CCM_MATRIX,
                    nbits_out=nbits_out,
                )

                if down:
                    output_rgb = resize(
                        output_rgb[None, ...], 1 / down, interpolation=cv2.INTER_CUBIC
                    )[0]

                # need OpenCV to save 16-bit RGB image
                if gray:
                    output_gray = rgb2gray(output_rgb[None, ...])
                    output_gray = output_gray.astype(output_rgb.dtype).squeeze()
                    cv2.imwrite(fn, output_gray)
                else:
                    cv2.imwrite(fn, cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR))
            else:
                img = Image.fromarray(output)
                img.save(fn)

        else:

            # capturing and returning non-bayer data
            from picamerax import PiCamera

            camera = PiCamera()
            if res:
                assert len(res) == 2
            else:
                res = np.array(camera.MAX_RESOLUTION)
                if down is not None:
                    res = (np.array(res) / down).astype(int)

            # -- now set up camera with desired settings
            camera = PiCamera(framerate=1 / exp, sensor_mode=sensor_mode, resolution=tuple(res))

            # Wait for the automatic gain control to settle
            time.sleep(config.config_pause)

            if config.awb_gains is not None:
                assert len(config.awb_gains) == 2
                g = (Fraction(config.awb_gains[0]), Fraction(config.awb_gains[1]))
                g = tuple(g)
                camera.awb_mode = "off"
                camera.awb_gains = g
                time.sleep(0.1)

            print("Capturing at resolution: ", res)
            print("AWB gains: ", float(camera.awb_gains[0]), float(camera.awb_gains[1]))

            try:
                camera.resolution = tuple(res)
                camera.capture(fn)
            except ValueError:
                raise ValueError(
                    "Out of resources! Use bayer for higher resolution, or increase `gpu_mem` in `/boot/config.txt`."
                )

    print("Image saved to : {}".format(fn))


if __name__ == "__main__":
    capture()
