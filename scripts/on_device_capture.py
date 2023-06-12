"""
Capture raw Bayer data or post-processed RGB data.

```
python scripts/on_device_capture.py --legacy --exp 0.02 --sensor_mode 0
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
from lensless.util import bayer2rgb, get_distro, rgb2gray, resize
from lensless.constants import RPI_HQ_CAMERA_CCM_MATRIX, RPI_HQ_CAMERA_BLACK_LEVEL
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


@hydra.main(version_base=None, config_path="../configs", config_name="capture")
def capture(config):

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

    # https://www.raspberrypi.com/documentation/accessories/camera.html#maximum-exposure-times
    # TODO : check which camera
    assert exp <= 230
    assert exp >= 0.02
    sensor_mode = int(sensor_mode)

    distro = get_distro()
    print("RPi distribution : {}".format(distro))
    if "bullseye" in distro and not legacy:
        # TODO : grayscale and downsample
        assert not rgb
        assert not gray
        assert down is None

        import subprocess

        jpg_fn = fn + ".jpg"
        dng_fn = fn + ".dng"
        pic_command = [
            "libcamera-still",
            "-r",
            "--gain",
            f"{iso / 100}",
            "--shutter",
            f"{int(exp * 1e6)}",
            "-o",
            f"{jpg_fn}",
        ]

        cmd = subprocess.Popen(
            pic_command,
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        cmd.stdout.readlines()
        cmd.stderr.readlines()
        os.remove(jpg_fn)
        os.system(f"exiftool {dng_fn}")
        print("\nJPG saved to : {}".format(jpg_fn))
        print("\nDNG saved to : {}".format(dng_fn))
    else:
        import picamerax.array

        fn += ".png"

        if bayer:

            # if rgb:

            #     camera = picamerax.PiCamera(framerate=1 / exp, sensor_mode=sensor_mode, resolution=res)
            #     camera.iso = iso
            #     # Wait for the automatic gain control to settle
            #     sleep(config_pause)
            #     # Now fix the values
            #     camera.shutter_speed = camera.exposure_speed
            #     camera.exposure_mode = "off"
            #     g = camera.awb_gains
            #     camera.awb_mode = "off"
            #     camera.awb_gains = g

            #     print("Resolution : {}".format(camera.resolution))
            #     print("Shutter speed : {}".format(camera.shutter_speed))
            #     print("ISO : {}".format(camera.iso))
            #     print("Frame rate : {}".format(camera.framerate))
            #     print("Sensor mode : {}".format(SENSOR_MODES[sensor_mode]))
            #     # keep this as it needs to be parsed from remote script!
            #     red_gain = float(g[0])
            #     blue_gain = float(g[1])
            #     print("Red gain : {}".format(red_gain))
            #     print("Blue gain : {}".format(blue_gain))

            #     # take picture
            #     fn += ".png"
            #     camera.capture(str(fn), bayer=False, resize=None)

            # else:

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

            if rgb or gray:
                if sixteen:
                    n_bits = 12  # assuming Raspberry Pi HQ
                else:
                    n_bits = 8
                output_rgb = bayer2rgb(
                    output,
                    nbits=n_bits,
                    blue_gain=blue_gain,
                    red_gain=red_gain,
                    black_level=RPI_HQ_CAMERA_BLACK_LEVEL,
                    ccm=RPI_HQ_CAMERA_CCM_MATRIX,
                    nbits_out=nbits_out,
                )

                if down:
                    output_rgb = resize(output_rgb, 1 / down, interpolation=cv2.INTER_CUBIC)

                # need OpenCV to save 16-bit RGB image
                if gray:
                    output_gray = rgb2gray(output_rgb)
                    output_gray = output_gray.astype(output.dtype)
                    cv2.imwrite(fn, output_gray)
                else:
                    cv2.imwrite(fn, cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR))
            else:
                img = Image.fromarray(output)
                img.save(fn)

        else:

            from picamerax import PiCamera

            camera = PiCamera()
            if res:
                assert len(res) == 2
            else:
                res = np.array(camera.MAX_RESOLUTION)
                if down is not None:
                    res = (np.array(res) / down).astype(int)

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

        print("\nImage saved to : {}".format(fn))


if __name__ == "__main__":
    capture()
