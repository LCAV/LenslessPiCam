"""
This script is used to export the .mat paf from https://github.com/Waller-Lab/DiffuserCam/tree/master/example_data
"""

import sys
import numpy as np
import scipy.io as sp


if len(sys.argv) < 2:
    print("Error : no filename provided. Aborting.")
    sys.exit()

filename = sys.argv[1]
if not filename.endswith(".mat"):
    print("Error : file is not a .mat file. Aborting")
    sys.exit()

img = np.array(list(sp.loadmat(filename).values()), dtype=object)[3]

img = img - np.min(img)
img = img / np.max(img)

img = np.swapaxes(img, 0, -1)
img = np.swapaxes(img, 1, 2)

np.save("psf.npy", img)
