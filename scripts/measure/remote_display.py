"""

Prior to launching, launch `feh` image viewer from Raspberry Pi:
```
feh LenslessPiCam_display --scale-down --auto-zoom -R 0.1 -x -F -Y
```
Refresh shouldn't be too fast, otherwise copying may have an issue.

Consequently, after running this script, the displayed image will be reloaded
on the display of the Raspberry Pi.
```
python scripts/measure/remote_display.py
```

Check out the `configs/demo.yaml` file for parameters, specifically:

- `fp`: path to image to display
- `display`: parameters for displaying image

Procedure is as follows:
- Image is copied to Raspberry Pi
- On the Raspberry Pi it is padded accordingly and saved to the path being viewed
by `feh`

"""

import os
import numpy as np
from PIL import Image
import hydra
from lensless.hardware.utils import display
from lensless.hardware.utils import check_username_hostname


@hydra.main(version_base=None, config_path="../../configs", config_name="demo")
def remote_display(config):

    check_username_hostname(config.rpi.username, config.rpi.hostname)
    username = config.rpi.username
    hostname = config.rpi.hostname

    fp = config.fp
    shape = np.array(config.display.screen_res)
    psf = config.display.psf
    black = config.display.black
    white = config.display.white

    if psf:
        point_source = np.zeros(tuple(shape) + (3,))
        mid_point = shape // 2
        start_point = mid_point - psf // 2
        end_point = start_point + psf
        point_source[start_point[0] : end_point[0], start_point[1] : end_point[1]] = 255
        fp = "tmp_display.png"
        im = Image.fromarray(point_source.astype("uint8"), "RGB")
        im.save(fp)

    elif black:
        point_source = np.zeros(tuple(shape) + (3,))
        fp = "tmp_display.png"
        im = Image.fromarray(point_source.astype("uint8"), "RGB")
        im.save(fp)

    elif white:
        point_source = np.ones(tuple(shape) + (3,)) * 255
        fp = "tmp_display.png"
        im = Image.fromarray(point_source.astype("uint8"), "RGB")
        im.save(fp)

    """ processing on remote machine, less issues with copying """
    # copy picture to Raspberry Pi
    print("\nCopying over picture...")
    display(fp=fp, rpi_username=username, rpi_hostname=hostname, **config.display)

    if psf or black or white:
        os.remove(fp)


if __name__ == "__main__":
    remote_display()
