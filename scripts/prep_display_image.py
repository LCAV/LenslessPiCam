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
    help="Padding percentage along each dimension.",
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
    "--hshift",
    default=0,
    type=float,
    help="Horizontal shift percentage.",
)
@click.option(
    "--brightness",
    default=100,
    type=float,
    help="Brightness percentage.",
)
@click.option(
    "--screen_res",
    default=None,
    nargs=2,
    type=int,
    help="Screen resolution in pixels (width, height).",
)
def display(fp, pad, output_path, vshift, brightness, screen_res, hshift):

    interpolation = cv2.INTER_NEAREST

    # load image
    img_og = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
    img_og = cv2.cvtColor(img_og, cv2.COLOR_BGR2RGB)

    if screen_res:
        img_og_dim = img_og.shape
        img = np.zeros((screen_res[1], screen_res[0], 3), dtype=img_og.dtype)

        if screen_res[0] < screen_res[1]:
            new_width = int(screen_res[0] / (1 + 2 * pad / 100))
            ratio = new_width / float(img_og_dim[0])
            new_height = int(ratio * img_og_dim[1])
        else:
            new_height = int(screen_res[1] / (1 + 2 * pad / 100))
            ratio = new_height / float(img_og_dim[1])
            new_width = int(ratio * img_og_dim[0])
        image_res = (new_width, new_height)
        img_og = cv2.resize(img_og, image_res, interpolation=interpolation)
        img[: image_res[1], : image_res[0]] = img_og

        # center
        img = np.roll(img, shift=int((screen_res[1] - image_res[1]) / 2), axis=0)
        img = np.roll(img, shift=int((screen_res[0] - image_res[0]) / 2), axis=1)

    else:

        # pad image
        if pad:
            padding_amount = np.array(img_og.shape[:2]) * pad / 100
            pad_width = (
                (int(padding_amount[0] // 2), int(padding_amount[0] // 2)),
                (int(padding_amount[1] // 2), int(padding_amount[1] // 2)),
                (0, 0),
            )
            img = np.pad(img_og, pad_width=pad_width)
        else:
            img = img_og

    if vshift:
        nx, _, _ = img.shape
        img = np.roll(img, shift=int(vshift * nx / 100), axis=0)

    if hshift:
        _, ny, _ = img.shape
        img = np.roll(img, shift=int(hshift * ny / 100), axis=1)

    if brightness:
        img = (img * brightness / 100).astype(np.uint8)

    # save to file
    im = Image.fromarray(img)
    im.save(output_path)


if __name__ == "__main__":
    display()
