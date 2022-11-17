#!/usr/bin/python3
import numpy as np
import scipy.io as sp
from PIL import Image

img = np.array(list(sp.loadmat("psf.mat").values()),dtype=object)[3]

img = img - np.min(img)
img = img / np.max(img)

img = np.swapaxes(img,0,-1)
img = np.swapaxes(img,1,2)

np.save("psf.npy", img)

for i in range(np.shape(img)[0]):
	result = Image.fromarray(img[i,:,:])
	result.save("psf/psf" + ("0" if i<10 else "") + str(i) + ".tiff")
