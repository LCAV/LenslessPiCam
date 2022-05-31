# CHANGELOG.md

## Unreleased

#### Added

- Example of RGB reconstruction with complex-valued FFT: `scripts/recon/apgd_pycsou.py`

#### Changed

- 

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