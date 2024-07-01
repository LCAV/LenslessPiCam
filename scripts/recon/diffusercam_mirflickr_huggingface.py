from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt
from lensless.utils.io import load_image, load_psf
from lensless.recon.admm import apply_admm
from lensless.recon.gd import apply_gradient_descent
import numpy as np


repo_id = "bezzam/DiffuserCam-Lensless-Mirflickr-Dataset-NORM"
psf = "psf.tiff"
lensless = "lensless_example.png"
lensed = "lensed_example.png"
single_psf = True

n_iter_admm = 10
n_iter_gd = 300
downsample = 4
use_torch = True
flip_ud = True

# download individual files
psf_fp = hf_hub_download(repo_id=repo_id, filename=psf, repo_type="dataset")
lensless_fp = hf_hub_download(repo_id=repo_id, filename=lensless, repo_type="dataset")
lensed_fp = hf_hub_download(repo_id=repo_id, filename=lensed, repo_type="dataset")

# apply ADMM
print("\n-- ADMM")
res = apply_admm(
    psf_fp,
    lensless_fp,
    n_iter_admm,
    downsample=downsample,
    use_torch=use_torch,
    flip_ud=flip_ud,
    verbose=True,
    single_psf=single_psf,
)
if use_torch:
    res = res.cpu().numpy()
res_admm = res[0] / res.max()

# apply GD
print("\n-- Gradient descent")
res_gd = apply_gradient_descent(
    psf_fp,
    lensless_fp,
    n_iter=n_iter_gd,
    downsample=downsample,
    use_torch=use_torch,
    flip_ud=flip_ud,
    verbose=True,
    single_psf=single_psf,
)
if use_torch:
    res_gd = res_gd.cpu().numpy()
res_gd = res_gd[0] / res_gd.max()

# Wiener filtering
psf = load_psf(psf_fp, downsample=downsample, flip_ud=flip_ud, single_psf=True)
lensless_img = load_image(lensless_fp, downsample=downsample / 4, flip_ud=flip_ud, as_4d=True)

# plot lensless, reconstruction, and ground truth
# -- measurements already 4x downsampled wrt to PSF
lensed_img = load_image(lensed_fp, downsample=downsample / 4, flip_ud=flip_ud)

fig, ax = plt.subplots(1, 5, figsize=(15, 5))
ax[0].imshow(lensless_img[0])
ax[0].set_title("Raw")
ax[1].imshow(psf[0], cmap="gray")
ax[1].set_title("PSF")
ax[2].imshow(res_gd)
ax[2].set_title("GD")
ax[3].imshow(res_admm)
ax[3].set_title("ADMM")
ax[4].imshow(lensed_img)
ax[4].set_title("Lensed")

plt.show()
