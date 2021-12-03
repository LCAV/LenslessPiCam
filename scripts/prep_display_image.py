"""
Prep image for displaying in full screen for DiffuserCam capture.

```
python scripts/prep_display_image.py --fp data/original_images/rect.jpg \
--pad 50 --output_path test.jpg
```

"""

import cv2
import numpy as np
from PIL import Image
import click


@click.command()
@click.option(
    "--fp",
    type=str,
    help="Path of file to display.",
)
@click.option(
    "--pad",
    default=0,
    type=float,
    help="Padding percentage.",
)
@click.option(
    "--output_path",
    default=None,
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
def display(fp, pad, output_path, vshift, brightness):

    # load image
    img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # pad image
    if pad:
        padding_amount = np.array(img.shape[:2]) * pad / 100
        pad_width = (
            (int(padding_amount[0] // 2), int(padding_amount[0] // 2)),
            (int(padding_amount[1] // 2), int(padding_amount[1] // 2)),
            (0, 0),
        )
        img = np.pad(img, pad_width=pad_width)

    if vshift:
        nx, _, _ = img.shape
        img = np.roll(img, shift=int(vshift * nx / 100), axis=0)

    if brightness:
        img = (img * brightness / 100).astype(np.uint8)

    # save to jpg
    im = Image.fromarray(img)
    im.save(output_path)


if __name__ == "__main__":
    display()
