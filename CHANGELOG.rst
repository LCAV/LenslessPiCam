Changelog
=========

All notable changes to `LenslessPiCam
<https://github.com/LCAV/LenslessPiCam>`_ will be documented in this file.

The format is based on `Keep a Changelog <http://keepachangelog.com/en/1.0.0/>`__.

Unreleased
----------

Added
~~~~~

-  Documentation files and configuration, using Sphinx.
-  Implementations for ``autocorr2d`` and ``RealFFTConvolve2D``.
-  Benchmarking tool for ReconstructionAlgorithm
-  ``n_iter`` parameter for ReconstructionAlgorithm constructor, so don't need to pass to ``apply``.

- Recon comparaison with/without Pytorch in 2D and in 3D
	Gradient Descent, 2D

	Default : ~140 s
	python scripts/recon/gradient_descent.py

	Torch, CPU : ~70 s
	python scripts/recon/gradient_descent.py torch=True

	Torch, cuda:0 : ~22 s
	python scripts/recon/gradient_descent.py -cn pytorch
	Gradient Descent, 3D

	Default : ~135 s
	python scripts/recon/gradient_descent.py input.psf="data/psf/diffuser_cam.npy" input.data="data/raw_data/diffuser_cam.tiff" preprocess.downsample=1

	Torch, CPU : ~105 s
	python scripts/recon/gradient_descent.py torch=True input.psf="data/psf/diffuser_cam.npy" input.data="data/raw_data/diffuser_cam.tiff" preprocess.downsample=1

	Torch, cuda:0 : ~27 s
	python scripts/recon/gradient_descent.py -cn pytorch input.psf="data/psf/diffuser_cam.npy" input.data="data/raw_data/diffuser_cam.tiff" preprocess.downsample=1

Changed
~~~~~~~

-  README.md to READ.rst for documentation.
-  CONTRIBUTING and CHANGELOG, to .rst for documentation.
-  Shorten README to separate contents in different pages of docs.
-  Fix typo in GradientDescent class name.
-  Updated to Pycsou V2, as ``pip install pycsou`` (Pycsou V1) may not work on some machines.

Bugfix
~~~~~~

-  Loading grayscale PSFs would cause an dimension error when removing the background pixels.

1.0.2 - (2022-05-31)
--------------------

Added
~~~~~

-  Example of RGB reconstruction with complex-valued FFT: ``scripts/recon/apgd_pycsou.py``


Bugfix
~~~~~~

-  Possible shape mismatch when using the real-valued FFT: forward and
   backward.

1.0.1 - (2022-04-26)
--------------------


Added
~~~~~

-  Scripts for collecting MNIST.
-  Option to collect grayscale data.


Changed
~~~~~~~

-  Restructure example scripts, i.e. subfolder ``recon`` for reconstructions.
-  Remove heavy installs from setup (e.g. pycsou, lpips, skikit-image).



1.0.0 - (2022-03-21)
--------------------

First version!



