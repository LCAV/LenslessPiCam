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
-  Support of 3D reconstruction for Gradient Descent and APGD, with and without Pytorch.
-  Option to warm-start reconstruction algorithm with ``initial_est``.
-  TrainableReconstructionAlgorithm class inherited from ReconstructionAlgorithm and torch.module for use with pytorch autograd and optimizers.
-  Unrolled version of FISTA and ADMM as TrainableReconstructionAlgorithm with learnable parameters.
-  Unrolled_recon.py script for training unrolled algorithms.

Changed
~~~~~~~

-  README.md to READ.rst for documentation.
-  CONTRIBUTING and CHANGELOG, to .rst for documentation.
-  Shorten README to separate contents in different pages of docs.
-  Fix typo in GradientDescent class name.
-  Updated to Pycsou V2, as ``pip install pycsou`` (Pycsou V1) may not work on some machines.
-  Data and PSF are now always stored as 4D Data [depth, width, height, color]. Grayscale data has a color axis of length 1 and 2D data has a depth axis of length 1.
-  Added batch support to RealFFTConvolve2D.

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



