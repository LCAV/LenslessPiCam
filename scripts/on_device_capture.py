"""
Capture raw Bayer data or post-processed RGB data.

See these code snippets for setting camera settings and post-processing
- https://github.com/scivision/pibayer/blob/1bb258c0f3f8571d6ded5491c0635686b59a5c4f/pibayer/base.py#L56
- https://picamera.readthedocs.io/en/release-1.13/recipes1.html#capturing-consistent-images
- https://www.strollswithmydog.com/open-raspberry-pi-high-quality-camera-raw

"""

import os
import cv2
import click
import numpy as np
from time import sleep
from PIL import Image
from diffcam.util import bayer2rgb, get_distro
from diffcam.constants import RPI_HQ_CAMERA_CCM_MATRIX, RPI_HQ_CAMERA_BLACK_LEVEL


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


@click.command()
@click.option(
    "--fn",
    default="test",
    type=str,
    help="File name for recorded image.",
)
@click.option(
    "--exp",
    default=0.5,
    type=float,
    help="Exposure time in seconds.",
)
@click.option(
    "--config_pause",
    default=2,
    type=float,
    help="Pause time for loading / setting camera configuration.",
)
@click.option(
    "--sensor_mode",
    default="0",
    type=click.Choice(np.arange(len(SENSOR_MODES)).astype(str)),
    help="{'off': 0, 'auto': 1, 'sunlight': 2, 'cloudy': 3, 'shade': 4, 'tungsten': 5, "
    "'fluorescent': 6, 'incandescent': 7, 'flash': 8, 'horizon': 9}",
)
@click.option(
    "--rgb", is_flag=True, help="Whether to reconstruct RGB data or return raw bayer data."
)
@click.option("--iso", default=100, type=int)
@click.option(
    "--sixteen",
    is_flag=True,
)
@click.option(
    "--nbits_out",
    default=8,
    type=int,
    help="Number of bits for output. Only used if saving RGB data.",
)
def capture(fn, exp, config_pause, sensor_mode, iso, sixteen, rgb, nbits_out):
    # https://www.raspberrypi.com/documentation/accessories/camera.html#maximum-exposure-times
    # TODO : check which camera
    assert exp <= 230
    assert exp > 0
    sensor_mode = int(sensor_mode)

    distro = get_distro()
    print("RPi distribution : {}".format(distro))
    if "bullseye" in distro:

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
        result = cmd.stdout.readlines()
        error = cmd.stderr.readlines()
        os.remove(jpg_fn)
        os.system(f"exiftool {dng_fn}")
        print("\nJPG saved to : {}".format(jpg_fn))
        print("\nDNG saved to : {}".format(dng_fn))
    else:
        import picamerax.array

        camera = picamerax.PiCamera(framerate=1 / exp, sensor_mode=sensor_mode)

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

        print("Resolution : {}".format(camera.MAX_RESOLUTION))
        print("Shutter speed : {}".format(camera.shutter_speed))
        print("ISO : {}".format(camera.iso))
        print("Frame rate : {}".format(camera.framerate))
        print("Sensor mode : {}".format(SENSOR_MODES[sensor_mode]))
        print("Config load time : {} seconds".format(config_pause))
        # keep this as it needs to be parsed from remote script!
        red_gain = float(awb_gains[0])
        blue_gain = float(awb_gains[1])
        print("Red gain : {}".format(red_gain))
        print("Blue gain : {}".format(blue_gain))

        # capture data
        stream = picamerax.array.PiBayerArray(camera)
        camera.capture(stream, "jpeg", bayer=True)
        fn += ".png"

        # get bayer data
        if sixteen:
            output = np.sum(stream.array, axis=2).astype(np.uint16)
        else:
            output = (np.sum(stream.array, axis=2) >> 2).astype(np.uint8)

        if rgb:
            if sixteen:
                n_bits = 12  # assuming Raspberry Pi HQ
            else:
                n_bits = 8
            output = bayer2rgb(
                output,
                nbits=n_bits,
                bg=blue_gain,
                rg=red_gain,
                black_level=RPI_HQ_CAMERA_BLACK_LEVEL,
                ccm=RPI_HQ_CAMERA_CCM_MATRIX,
                nbits_out=nbits_out,
            )

            # need OpenCV to save 16-bit RGB image
            cv2.imwrite(fn, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
        else:
            img = Image.fromarray(output)
            img.save(fn)

        print("\nImage saved to : {}".format(fn))


if __name__ == "__main__":
    capture()
