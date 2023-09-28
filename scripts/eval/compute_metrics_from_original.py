"""
After downloading the example files:
- Reconstrution: https://drive.switch.ch/index.php/s/NdgHlcDeHVDH5ww?path=%2Freconstruction
- Original: https://drive.switch.ch/index.php/s/NdgHlcDeHVDH5ww?path=%2Foriginal

And placing in (respectively):
- data/reconstruction
- data/original

The script can be run with:
```
python scripts/eval/compute_metrics_from_original.py
```

"""

import hydra
import os
from hydra.utils import to_absolute_path
import numpy as np
import matplotlib.pyplot as plt
from lensless.utils.plot import plot_image
from lensless.utils.io import load_image
from lensless.eval.metric import mse, psnr, ssim, lpips, extract
import matplotlib

font = {"family": "DejaVu Sans", "size": 18}
matplotlib.rc("font", **font)


@hydra.main(
    version_base=None, config_path="../../configs", config_name="compute_metrics_from_original"
)
def compute_metrics(config):
    recon = to_absolute_path(config.files.recon)
    original = to_absolute_path(config.files.original)
    vertical_crop = config.alignment.vertical_crop
    horizontal_crop = config.alignment.horizontal_crop
    rotation = config.alignment.rotation
    verbose = config.verbose

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

    plt.savefig("comparison.png")
    save = os.getcwd() + "/comparison.png"
    print(f"Save comparison to {save}")
    plt.show()


if __name__ == "__main__":
    compute_metrics()
