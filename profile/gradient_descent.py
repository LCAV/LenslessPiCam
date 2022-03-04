from diffusercam_gd import grad_descent, GradientDescentUpdate
from lensless.gradient_descent import FISTA
from lensless.io import load_data
import time
import pathlib as plib


psf_fp = "data/psf/tape_rgb.png"
data_fp = "data/raw_data/thumbs_up_rgb.png"

n_iter = 300
downsample = 4
gray = True


psf, data = load_data(
    psf_fp=psf_fp,
    data_fp=data_fp,
    downsample=downsample,
    plot=False,
    gray=gray,
)

save = "profile_gradient_descent"
save = plib.Path(__file__).parent / save
save.mkdir(exist_ok=True)

""" LenslessPiCam """
recon = FISTA(psf)
recon.set_data(data)
start_time = time.time()
res = recon.apply(n_iter=n_iter, save=save, disp_iter=None, plot=False)
print(f"LenslessPiCam : {time.time() - start_time} s")

""" DiffuserCam"""
method = GradientDescentUpdate.FISTA
start_time = time.time()
grad_descent(psf, data, n_iter=n_iter, update_method=method, disp_iter=None, save=save)
print(f"DiffuserCam   : {time.time() - start_time} s")
