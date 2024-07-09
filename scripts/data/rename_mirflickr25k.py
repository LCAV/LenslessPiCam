"""
Utility file to rename files in MirFlickr25k dataset (https://press.liacs.nl/mirflickr/)
so that they match the names in the larger dataset, i.e. removing "im" from the filename.
"""


import os
import glob
from lensless.utils.dataset import natural_sort


dir_path = "/dev/shm/mirflickr"

# get all jpg files
files = natural_sort(glob.glob(os.path.join(dir_path, "*.jpg")))

# Rename all files in the directory
for filename in files:
    # remove "im" from the filename
    new_filename = filename.replace("im", "")

    # decrement by 1
    bn = os.path.basename(new_filename)
    file_number = int(bn.split(".")[0])
    new_bn = f"{file_number - 1}.jpg"
    new_filename = new_filename.replace(bn, new_bn)

    os.rename(filename, new_filename)

print(f"Number of files: {len(files)}")
print("Done")
