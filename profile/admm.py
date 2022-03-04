from diffusercam_admm import apply_admm
from lensless.admm import ADMM
from lensless.io import load_data
import time
import pathlib as plib
import numpy as np


psf_fp = "data/psf/tape_rgb.png"
data_fp = "data/raw_data/thumbs_up_rgb.png"

n_iter = 5
downsample = 4
gray = True
dtype = np.float32


psf, data = load_data(
    psf_fp=psf_fp,
    data_fp=data_fp,
    downsample=downsample,
    plot=False,
    gray=gray,
    dtype=dtype,
)

save = "profile_admm"
save = plib.Path(__file__).parent / save
save.mkdir(exist_ok=True)

""" LenslessPiCam """
recon = ADMM(psf, dtype=dtype)
recon.set_data(data)
start_time = time.time()
res = recon.apply(n_iter=n_iter, save=save, disp_iter=None, plot=False)
print(f"LenslessPiCam : {time.time() - start_time} s")

""" DiffuserCam"""
sensor_size = np.array(psf.shape[:2])
full_size = 2 * sensor_size
param = {
    "mu1": 1e-6,
    "mu2": 1e-5,
    "mu3": 4e-5,
    "tau": 0.0001,
    "sensor_size": sensor_size,
    "full_size": full_size,
}
start_time = time.time()
apply_admm(psf, data, n_iter=n_iter, param=param, disp_iter=None, save=save)
print(f"DiffuserCam   : {time.time() - start_time} s")
