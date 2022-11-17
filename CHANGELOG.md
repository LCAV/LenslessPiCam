# CHANGELOG.md

## Unreleased

#### Added

- Support of 3D psfs, which have to be provided as .npy files.
- 3D reconstriction works for gradient descent and ADMM but the regularizer term will probably need to be adapted
- The code will automatically perform 3D reconstruction when the provided psf is a .npy file, and perform 2D construction otherwise
- Uploaded 3D data from `https://github.com/Waller-Lab/DiffuserCam`
- Try it : `/scripts/recon/gradient_descent.py --psf_fp data/diffuser3D/psf.npy --data_fp data/diffuser3D/image.tiff --downsample 1 --disp 5 --n_iter 200`

#### Changed

- The data of images and psfs are now always stored as (depth, width, height, color) arrays in memory.
- Each reconstruction algorith was adapted accordingly.

#### Bugfix

- Loading grayscale PSFs would cause an dimension error when removing the background pixels

## 1.0.2 - (2022-05-31)

#### Added

- Example of RGB reconstruction with complex-valued FFT: `scripts/recon/apgd_pycsou.py`

#### Bugfix

- Possible shape mismatch when using the real-valued FFT: forward and backward.

## 1.0.1 - (2022-04-26)

#### Added

- Scripts for collecting MNIST.
- Option to collect grayscale data.

#### Changed

- Restructure example scripts, i.e. subfolder `recon` for reconstructions.
- Remove heavy installs from setup (e.g. pycsou, lpips, skikit-image).


## 1.0.0 - (2022-03-21)

First version!
