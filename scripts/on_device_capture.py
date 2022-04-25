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
from lensless.util import bayer2rgb, get_distro, rgb2gray, resize
from lensless.constants import RPI_HQ_CAMERA_CCM_MATRIX, RPI_HQ_CAMERA_BLACK_LEVEL


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
@click.option(
    "--gray",
    is_flag=True,
    help="Get grayscale data from the Raspberry Pi.",
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
@click.option(
    "--legacy",
    is_flag=True,
    help="Whether to use legacy camera software, despite being on Bullseye.",
)
@click.option("--down", type=float, help="Factor by which to downsample output.", default=None)
def capture(fn, exp, config_pause, sensor_mode, iso, sixteen, rgb, nbits_out, legacy, gray, down):
    # https://www.raspberrypi.com/documentation/accessories/camera.html#maximum-exposure-times
    # TODO : check which camera
    assert exp <= 230
    assert exp > 0
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

        if rgb or gray:
            if sixteen:
                n_bits = 12  # assuming Raspberry Pi HQ
            else:
                n_bits = 8
            output = bayer2rgb(
                output,
                nbits=n_bits,
                blue_gain=blue_gain,
                red_gain=red_gain,
                black_level=RPI_HQ_CAMERA_BLACK_LEVEL,
                ccm=RPI_HQ_CAMERA_CCM_MATRIX,
                nbits_out=nbits_out,
            )

            if down:
                output = resize(output, 1 / down, interpolation=cv2.INTER_CUBIC)

            # need OpenCV to save 16-bit RGB image
            if gray:
                output_gray = rgb2gray(output)
                output_gray = output_gray.astype(output.dtype)
                cv2.imwrite(fn, output_gray)
            else:
                cv2.imwrite(fn, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
        else:
            img = Image.fromarray(output)
            img.save(fn)

        print("\nImage saved to : {}".format(fn))


if __name__ == "__main__":
    capture()
