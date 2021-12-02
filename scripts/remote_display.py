"""

Prior to launching, launch `feh` image viewer from Raspberry Pi:
```
feh DiffuserCam_display --scale-down --auto-zoom -R 0.1 -x -F -Y
```
Refresh shouldn't be too fast, otherwise copying may have an issue.

Consequently, after running this script, the displayed image will be reloaded
on the display of the Raspberry Pi.
```
python scripts/remote_display.py --fp data/original_images/rect.jpg --pad 50
```

Procedure is as follows:
- Image is copied to Raspberry Pi
- On the Raspberry Pi it is padded accordingly and saved to the path being viewed
by `feh`

"""

import cv2
import os
import click
import numpy as np
from PIL import Image
import subprocess
import sys


REMOTE_PYTHON = "~/DiffuserCam/diffcam_env/bin/python"
REMOTE_IMAGE_PREP_SCRIPT = "~/DiffuserCam/scripts/prep_display_image.py"
REMOTE_DISPLAY_PATH = "~/DiffuserCam_display/test.jpg"
REMOTE_TMP_PATH = "~/tmp_display.jpg"


@click.command()
@click.option(
    "--fp",
    type=str,
    help="Image to display on RPi display.",
)
@click.option(
    "--hostname",
    type=str,
    help="Hostname or IP address.",
)
@click.option(
    "--pad",
    default=0,
    type=float,
    help="Padding percentage.",
)
@click.option(
    "--psf",
    default=0,
    type=int,
    help="Number of pixels for PSF if positive",
)
@click.option(
    "--black",
    is_flag=True,
    help="All black background",
)
@click.option(
    "--vshift",
    default=0,
    type=float,
    help="Vertical shift percentage.",
)
@click.option(
    "--brightness",
    default=0,
    type=float,
    help="Brightness percentage.",
)
def remote_display(fp, hostname, pad, vshift, brightness, psf, black):

    if psf:
        assert fp is None
        # create image, size of https://www.dell.com/en-us/work/shop/dell-ultrasharp-usb-c-hub-monitor-u2421e/apd/210-axmg/monitors-monitor-accessories#techspecs_section
        shape = np.array((1200, 1920))
        point_source = np.zeros(tuple(shape) + (3,))
        mid_point = shape // 2
        start_point = mid_point - psf // 2
        end_point = start_point + psf
        point_source[start_point[0] : end_point[0], start_point[1] : end_point[1]] = 255
        fp = "tmp_display.png"
        im = Image.fromarray(point_source.astype("uint8"), "RGB")
        im.save(fp)
    elif black:
        assert fp is None
        shape = np.array((1200, 1920))
        point_source = np.zeros(tuple(shape) + (3,))
        fp = "tmp_display.png"
        im = Image.fromarray(point_source.astype("uint8"), "RGB")
        im.save(fp)

    """ processing on remote machine, less issues with copying """
    # copy picture to Raspberry Pi
    print(f"\nCopying over picture...")
    os.system('scp %s "pi@%s:%s" ' % (fp, hostname, REMOTE_TMP_PATH))

    prep_command = f"{REMOTE_PYTHON} {REMOTE_IMAGE_PREP_SCRIPT} --fp {REMOTE_TMP_PATH} \
        --pad {pad} --vshift {vshift} --brightness {brightness} --output_path {REMOTE_DISPLAY_PATH} "
    print(f"COMMAND : {prep_command}")
    subprocess.Popen(
        ["ssh", "pi@%s" % hostname, prep_command],
        shell=False,
    )

    if psf:
        os.remove(fp)


if __name__ == "__main__":
    remote_display()
