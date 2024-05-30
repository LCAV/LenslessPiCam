from diffusercam_gd import grad_descent, GradientDescentUpdate
from lensless.recon.gd import FISTA
from lensless.recon.apgd import APGD, APGDPriors
from lensless.utils.io import load_data
import time
import pathlib as plib


psf_fp = "data/psf/tape_rgb.png"
data_fp = "data/raw_data/thumbs_up_rgb.png"

n_iter = 300
downsample = 4
gray = True
dtype = "float32"
n_trials = 3

psf, data = load_data(
    psf_fp=psf_fp,
    data_fp=data_fp,
    downsample=downsample,
    plot=False,
    gray=gray,
    dtype=dtype,
)

save = "profile_gradient_descent"
save = plib.Path(__file__).parent / save
save.mkdir(exist_ok=True)

""" LenslessPiCam """
recon = FISTA(psf, dtype=dtype)
recon.set_data(data)
# res = recon.apply(n_iter=n_iter, save=save, disp_iter=None)
# recon.reset()
total_time = 0
for _ in range(n_trials):
    start_time = time.time()
    res = recon.apply(n_iter=n_iter, disp_iter=None)
    total_time += time.time() - start_time
    recon.reset()
print(f"LenslessPiCam (avg) : {total_time / n_trials} s")

# -- using Pycsou, complex conv
recon = APGD(
    psf,
    max_iter=n_iter,
    acceleration="BT",
    diff_penalty=None,
    prox_penalty=APGDPriors.NONNEG,
    realconv=False,
)
recon.set_data(data)
res = recon.apply(n_iter=n_iter, save=save, disp_iter=None)
recon.reset()
total_time = 0
for _ in range(n_trials):
    start_time = time.time()
    res = recon.apply(n_iter=n_iter, disp_iter=None)
    total_time += time.time() - start_time
    recon.reset()
print(f"LenslessPiCam (Pycsou, complex) : {total_time / n_trials} s")

# -- using Pycsou, real conv
recon = APGD(
    psf,
    max_iter=n_iter,
    acceleration="BT",
    diff_penalty=None,
    prox_penalty=APGDPriors.NONNEG,
    realconv=True,
)
recon.set_data(data)
res = recon.apply(n_iter=n_iter, save=save, disp_iter=None)
recon.reset()
total_time = 0
for _ in range(n_trials):
    start_time = time.time()
    res = recon.apply(n_iter=n_iter, disp_iter=None)
    total_time += time.time() - start_time
    recon.reset()
print(f"LenslessPiCam (Pycsou, real) : {total_time / n_trials} s")

""" DiffuserCam"""
method = GradientDescentUpdate.FISTA
grad_descent(psf, data, n_iter=n_iter, update_method=method, disp_iter=None, save=save)
start_time = time.time()
for _ in range(n_trials):
    grad_descent(psf, data, n_iter=n_iter, update_method=method, disp_iter=None)
print(f"DiffuserCam : {(time.time() - start_time) / n_trials} s")

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

recon = FISTA(psf, dtype=dtype)
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

recon = FISTA(psf, dtype=dtype)
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
