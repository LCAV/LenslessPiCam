"""
Check maximum pixel value of images and check for saturation / underexposure.

```
python scripts/measure/analyze_measured_dataset.py folder=PATH
```
"""

import hydra
from hydra.utils import to_absolute_path
import glob
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time


@hydra.main(version_base=None, config_path="../../configs", config_name="analyze_dataset")
def analyze_dataset(config):

    folder = config.dataset_path
    desired_range = config.desired_range
    delete_saturate = config.delete_saturated

    assert (
        folder is not None
    ), "Must specify folder to analyze in config or through command line (folder=PATH)."

    # get all PNG files in folder
    files = sorted(glob.glob(os.path.join(folder, "*.png")))
    print("Found {} files".format(len(files)))
    if config.n_files is not None:
        files = files[: config.n_files]
        print("Analyzing first {} files".format(len(files)))

    # loop over files for maximum value
    max_vals = []
    n_bad_files = 0
    start_time = time.time()
    for fn in files:
        im = np.array(Image.open(fn))
        max_val = im.max()
        max_vals.append(max_val)

        # if out of desired range, print filename
        if max_val < desired_range[0] or max_val > desired_range[1]:
            # print("File {} has max value {}".format(fn, max_val))
            n_bad_files += 1

        if delete_saturate and max_val == 255:
            os.remove(fn)
            print("REMOVED file {}".format(fn))

    proc_time = time.time() - start_time
    print("Went through {} files in {:.2f} seconds".format(len(files), proc_time))
    print(
        "Found {} / {} bad files  ({}%)".format(
            n_bad_files, len(files), n_bad_files / len(files) * 100
        )
    )

    # plot histogram
    output_folder = os.getcwd()
    output_fp = os.path.join(output_folder, "max_vals.png")
    plt.hist(max_vals, bins=100)
    plt.savefig(output_fp)

    print("Saved histogram to {}".format(output_fp))


if __name__ == "__main__":
    analyze_dataset()
