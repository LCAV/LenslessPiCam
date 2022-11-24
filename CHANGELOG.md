# CHANGELOG.md

## Unreleased

#### Added

- Support of 3D psfs, which have to be provided as .npy files.
- 3D reconstriction for gradient descent and ADMM
- Script to export mat data for `https://github.com/Waller-Lab/DiffuserCam` to usable .npy

#### Changed

- The data of images and psfs are now always stored as (depth, width, height, color) arrays in memory. Each reconstruction algorithm was adapted accordingly.

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
