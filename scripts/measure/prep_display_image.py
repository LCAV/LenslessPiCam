"""
Prep image for displaying in full screen for DiffuserCam capture.

```
python scripts/measure/prep_display_image.py --fp data/original_images/rect.jpg \
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
@click.option(
    "--rot90",
    default=0,
    type=int,
    help="How many times to rotate provided image by 90 degrees.",
)
@click.option(
    "--landscape",
    is_flag=True,
    help="Force landscape.",
)
@click.option(
    "--image_res",
    default=None,
    nargs=2,
    type=int,
    help="Image resolution in pixels (width, height).",
)
def display(
    fp, pad, output_path, vshift, brightness, screen_res, hshift, rot90, landscape, image_res
):
    interpolation = cv2.INTER_NEAREST

    # load image
    img_og = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
    img_og = cv2.cvtColor(img_og, cv2.COLOR_BGR2RGB)
    if landscape:
        if img_og.shape[0] > img_og.shape[1]:
            img_og = np.rot90(img_og)

    # rotate image
    if rot90:
        img_og = np.rot90(img_og, k=rot90)

        # if odd, swap hshift and vshift
        if rot90 % 2:
            vshift, hshift = hshift, vshift

    if screen_res:
        image_height, image_width = img_og.shape[:2]
        img = np.zeros((screen_res[1], screen_res[0], 3), dtype=img_og.dtype)

        if image_res is None:

            # set image with padding and correct aspect ratio
            if screen_res[0] < screen_res[1]:

                max_ratio = screen_res[1] / float(image_height)

                new_width = int(screen_res[0] / (1 + 2 * pad / 100))
                ratio = new_width / float(image_width)
                # new_height = int(ratio * image_height)

            else:

                max_ratio = screen_res[0] / float(image_width)

                new_height = int(screen_res[1] / (1 + 2 * pad / 100))
                ratio = new_height / float(image_height)
                # new_width = int(ratio * image_width)

            ratio = min(ratio, max_ratio)
            new_width = int(ratio * image_width)
            new_height = int(ratio * image_height)
            image_res = (new_width, new_height)

        # if negative value in image res
        elif image_res[0] < 0 or image_res[1] < 0:
            assert image_res[0] > 0 or image_res[1] > 0, "Both dimensions cannot be negative."
            # rescale according to non-negative value
            if image_res[0] < 0:
                new_height = image_res[1]
                ratio = new_height / float(image_height)
                image_res = (int(ratio * image_width), new_height)

            elif image_res[1] < 0:
                new_width = image_res[0]
                ratio = new_width / float(image_width)
                image_res = (new_width, int(ratio * image_height))

        # set image within screen
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
