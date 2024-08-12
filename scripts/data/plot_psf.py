from huggingface_hub import hf_hub_download
from lensless.utils.io import load_psf, save_image
from lensless.utils.image import gamma_correction
import os
import numpy as np
import torch
from lensless.hardware.trainable_mask import AdafruitLCD

gamma = 1.8

## TapeCam
repo_id = "bezzam/TapeCam-Mirflickr-25K"
psf = "psf.png"
downsample = 8
flip_ud = False

# DigiCam-CelebA
repo_id = "bezzam/DigiCam-CelebA-26K"
psf = "psf_measured.png"
downsample = 8
gamma = 1.5
flip_ud = False

# # DigiCam-MirFlickr-25K
# repo_id = "bezzam/DigiCam-Mirflickr-SingleMask-25K"
# psf = "mask_pattern.npy"
# downsample = 8
# flip_ud = True

# # Multi Mask
# repo_id = "bezzam/DigiCam-Mirflickr-MultiMask-25K"
# psf = "masks/mask_4.npy"
# downsample = 8
# flip_ud = True


psf_fp = hf_hub_download(repo_id=repo_id, filename=psf, repo_type="dataset")

# load PSF
if psf.endswith(".npy"):
    mask_vals = np.load(psf_fp)
    mask = AdafruitLCD(
        initial_vals=torch.from_numpy(mask_vals.astype(np.float32)),
        sensor="rpi_hq",
        slm="adafruit",
        downsample=downsample,
        flipud=flip_ud,
        use_waveprop=True,
        scene2mask=0.3,
        mask2sensor=0.002,
        deadspace=True,
    )
    psf = mask.get_psf().detach().numpy()
else:
    psf = load_psf(psf_fp, downsample=downsample, flip_ud=flip_ud)

psf = psf / psf.max()
if gamma > 1:
    psf = gamma_correction(psf, gamma=gamma)

# save as viewable PNG
fn = os.path.basename(repo_id) + "_psf.png"
save_image(psf, fn)
print(f"Saved PSF as {fn}")
