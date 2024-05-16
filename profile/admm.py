from diffusercam_admm import apply_admm
from lensless.recon.admm import ADMM
from lensless.utils.io import load_data
import time
import pathlib as plib
import numpy as np


psf_fp = "data/psf/tape_rgb.png"
data_fp = "data/raw_data/thumbs_up_rgb.png"

n_iter = 5
downsample = 4
gray = True
dtype = "float32"
n_trials = 10


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
res = recon.apply(n_iter=n_iter, save=save, disp_iter=None, plot=False)
recon.reset()
total_time = 0
for _ in range(n_trials):
    start_time = time.time()
    res = recon.apply(n_iter=n_iter, disp_iter=None, plot=False)
    total_time += time.time() - start_time
    recon.reset()
print(f"LenslessPiCam (avg) : {total_time / n_trials} s")

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
apply_admm(psf, data, n_iter=n_iter, param=param, disp_iter=None, save=save)
start_time = time.time()
for _ in range(n_trials):
    apply_admm(psf, data, n_iter=n_iter, param=param, disp_iter=None)
print(f"DiffuserCam (avg)  : {(time.time() - start_time) / n_trials} s")


""" PyTorch CPU """
psf, data = load_data(
    psf_fp=psf_fp,
    data_fp=data_fp,
    downsample=downsample,
    plot=False,
    gray=gray,
    dtype=dtype,
    use_torch=True,
)

recon = ADMM(psf, dtype=dtype)
recon.set_data(data)
res = recon.apply(n_iter=n_iter, save=save, disp_iter=None, plot=False)
recon.reset()
total_time = 0
for _ in range(n_trials):
    start_time = time.time()
    res = recon.apply(n_iter=n_iter, disp_iter=None, plot=False)
    total_time += time.time() - start_time
    recon.reset()
print(f"LenslessPiCam, PyTorch CPU (avg) : {total_time / n_trials} s")

""" PyTorch GPU """
psf, data = load_data(
    psf_fp=psf_fp,
    data_fp=data_fp,
    downsample=downsample,
    plot=False,
    gray=gray,
    dtype=dtype,
    use_torch=True,
    torch_device="cuda",
)

recon = ADMM(psf, dtype=dtype)
recon.set_data(data)
res = recon.apply(n_iter=n_iter, save=save, disp_iter=None, plot=False)
recon.reset()
total_time = 0
for _ in range(n_trials):
    start_time = time.time()
    res = recon.apply(n_iter=n_iter, disp_iter=None, plot=False)
    total_time += time.time() - start_time
    recon.reset()
print(f"LenslessPiCam, PyTorch GPU (avg) : {total_time / n_trials} s")
