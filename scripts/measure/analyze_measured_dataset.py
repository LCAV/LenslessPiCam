"""
Check maximum pixel value of images and check for saturation / underexposure.

```
python scripts/measure/analyze_measured_dataset.py dataset_path=PATH
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
import tqdm
import re


def convert(text):
    return int(text) if text.isdigit() else text.lower()


def alphanum_key(key):
    return [convert(c) for c in re.split("([0-9]+)", key)]


def natural_sort(arr):
    return sorted(arr, key=alphanum_key)


@hydra.main(version_base=None, config_path="../../configs", config_name="analyze_dataset")
def analyze_dataset(config):

    folder = config.dataset_path
    desired_range = config.desired_range
    delete_bad = config.delete_bad
    start_idx = config.start_idx
    saturation_percent = config.saturation_percent

    assert (
        folder is not None
    ), "Must specify folder to analyze in config or through command line (folder=PATH)."

    # get all PNG files in folder
    files = natural_sort(glob.glob(os.path.join(folder, "*.png")))
    print("Found {} files".format(len(files)))
    if start_idx is not None:
        files = files[start_idx:]
        print("Starting at file {}".format(files[0]))
    if config.n_files is not None:
        files = files[: config.n_files]
        print("Analyzing first {} files".format(len(files)))

    # loop over files for maximum value
    max_vals = []
    n_bad_files = 0
    bad_files = []
    start_time = time.time()
    for fn in tqdm.tqdm(files):
        im = np.array(Image.open(fn))
        max_val = im.max()
        max_vals.append(max_val)
        saturation_ratio = np.sum(im >= desired_range[1]) / im.size

        if max_val < desired_range[0]:
            n_bad_files += 1
            bad_files.append(fn)

            if delete_bad:
                os.remove(fn)
                print("REMOVED file {}".format(fn))
            else:
                print("File {} has max value {}".format(fn, max_val))

        elif saturation_ratio > saturation_percent:
            n_bad_files += 1
            bad_files.append(fn)

            if delete_bad:
                os.remove(fn)
                print("REMOVED file {}".format(fn))
            else:
                print("File {} has saturation ratio {}".format(fn, saturation_ratio))

        # # if out of desired range, print filename
        # if max_val < desired_range[0] or saturation_ratio > saturation_percent:
        #     # print("File {} has max value {}".format(fn, max_val))
        #     n_bad_files += 1
        #     bad_files.append(fn)

        #     if delete_bad:
        #         os.remove(fn)
        #         print("REMOVED file {}".format(fn))
        #     else:
        #         print("File {} has max value {}".format(fn, max_val))

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

    # command line input on whether to delete bad files
    if not delete_bad:
        response = None
        while response not in ["yes", "no"]:
            response = input("Delete bad files: [yes|no] : ")
        if response == "yes":
            for _fn in bad_files:
                os.remove(_fn)
        else:
            print("Not deleting bad files")

    # check for matching background file
    files_bg = natural_sort(glob.glob(os.path.join(folder, "black_background*.png")))
    # -- remove files_bg from files
    files = [fn for fn in files if fn not in files_bg]

    if len(files_bg) > 0:
        print("Found {} background files".format(len(files_bg)))
        # detect files that don't have background
        files_no_bg = []
        for fn in files:
            bn = os.path.basename(fn).split(".")[0]
            _bg_file = os.path.join(folder, "black_background{}.png".format(bn))
            if _bg_file not in files_bg:
                files_no_bg.append(fn)

        print("Found {} files without background".format(len(files_no_bg)))
        # ask to delete files without background
        response = None
        while response not in ["yes", "no"]:
            response = input("Delete files without background: [yes|no] : ")
        if response == "yes":
            for _fn in files_no_bg:
                if os.path.exists(_fn):  # maybe already deleted before
                    os.remove(_fn)


if __name__ == "__main__":
    analyze_dataset()
