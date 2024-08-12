"""
Utility file to rename files in MirFlickr25k dataset (https://press.liacs.nl/mirflickr/)
so that they match the names in the larger dataset, i.e. removing "im" from the filename.

First download MIRFLICKR-25K dataset from the above link and extract it to a directory.
```bash
wget http://press.liacs.nl/mirflickr/mirflickr25k.v3b/mirflickr25k.zip -P data/mirflickr
unzip data/mirflickr/mirflickr25k.zip -d data/mirflickr
```
Then run this script to rename the files (updating `dir_path`).
```bash
python scripts/data/rename_mirflickr25k.py
```
"""

import os
import glob
from lensless.utils.dataset import natural_sort


dir_path = "data/mirflickr/mirflickr"

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
