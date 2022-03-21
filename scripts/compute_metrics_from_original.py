"""

Example usage:
```
python scripts/compute_metrics_from_original.py \
--recon data/reconstruction/admm_thumbs_up_rgb.npy \
--original data/original/thumbs_up.png \
--vertical_crop 262 371 \
--horizontal_crop 438 527 \
--rotation -0.5
```
you can get examples files from SWITCHDrive: https://drive.switch.ch/index.php/s/NdgHlcDeHVDH5ww

"""

import numpy as np
import click
import matplotlib.pyplot as plt
from lensless.plot import plot_image
from lensless.io import load_image
from lensless.metric import mse, psnr, ssim, lpips, extract
import matplotlib

font = {"family": "DejaVu Sans", "size": 18}
matplotlib.rc("font", **font)


@click.command()
@click.option(
    "--recon",
    type=str,
    help="File path to reconstruction.",
)
@click.option(
    "--original",
    type=str,
    help="File path to original file.",
)
@click.option(
    "--vertical_crop", type=(int, int), help="Cropping for vertical dimension.", default=(0, -1)
)
@click.option(
    "--horizontal_crop", type=(int, int), help="Cropping for horizontal dimension.", default=(0, -1)
)
@click.option("--rotation", type=float, help="Degrees to rotate reconstruction.", default=0)
@click.option("-v", "--verbose", count=True)
def compute_metrics(recon, original, vertical_crop, horizontal_crop, rotation, verbose):

    # load estimate
    est = np.load(recon)
    if verbose:
        print("estimate shape", est.shape)

    # load original image
    img = load_image(original)
    img = img / img.max()

    # extract matching parts from estimate and original
    est, img_resize = extract(est, img, vertical_crop, horizontal_crop, rotation, verbose=verbose)

    _, ax = plt.subplots(ncols=2, nrows=1, figsize=(10, 5))
    plot_image(est, ax=ax[0])
    ax[0].set_title("Reconstruction")
    plot_image(img_resize, ax=ax[1])
    ax[1].set_title("Original")

    print("\nMSE", mse(img_resize, est))
    print("PSNR", psnr(img_resize, est))
    print("SSIM", ssim(img_resize, est))
    print("LPIPS", lpips(img_resize, est))

    plt.show()


if __name__ == "__main__":
    compute_metrics()
