Changelog
=========

All notable changes to `LenslessPiCam
<https://github.com/LCAV/LenslessPiCam>`_ will be documented in this file.

The format is based on `Keep a Changelog <http://keepachangelog.com/en/1.0.0/>`__.


Unreleased
----------

Added
~~~~~

- Sensor module.
- Single-script and Telegram demo.
- Link and citation for JOSS.
- Authors at top of source code files.
- Add paramiko as dependency for remote capture and display.
- Mask module, for CodedAperture (FlatCam), PhaseContour (PhlatCam), and FresnelZoneAperture.
- Script for measuring arbitrary dataset (from Raspberry Pi).
- Support for preprocessing and postprocessing, such as denoising, in ``TrainableReconstructionAlgorithm``. Both trainable and fix postprocessing can be used.
- Utilities to load a trained DruNet model for use as postprocessing in ``TrainableReconstructionAlgorithm``.
- Unified interface for dataset. See ``utils.dataset.DualDataset``.
- New simulated dataset compatible with new data format ([(batch_size), depth, width, height, color]). See ``utils.dataset.SimulatedDataset``.
- New dataset for pair of original image and thair measurement from a screen. See ``utils.dataset.LenslessDataset``.
- Support for unrolled loading and inference in the script ``admm.py``.
- Tikhonov reconstruction for coded aperture measurements (MLS / MURA).


Changed
~~~~~~~

- Simpler remote capture and display scripts with Hydra.
- Group source code into four modules: ``hardware``, ``recon``, ``utils``, ``eval``.
- Split scripts into subfolders.
- Displaying 3D reconstructions now shows projections on all three axis.


Bugfix
~~~~~~

- Fix overwriting of sensor parameters when downsampling.
- Displaying 3D reconstructions by summing values along axis would produce un-normalized values.

1.0.4 - (2023-06-14)
--------------------

Bugfix
~~~~~~

- Fix rendering of README on PyPI.


1.0.3 - (2023-06-14)
--------------------

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
- ``train_unrolled.py`` script for training unrolled algorithms.
- ``benchmark_recon.py`` script for benchmarking and comparing reconstruction algorithms.
- Added ``reconstruction_error`` to ``ReconstructionAlgorithm`` .
- Added support for npy/npz image in load_image.

Changed
~~~~~~~

-  README.md to READ.rst for documentation.
-  CONTRIBUTING and CHANGELOG, to .rst for documentation.
-  Shorten README to separate contents in different pages of docs.
-  Fix typo in GradientDescent class name.
-  Updated to Pycsou V2, as ``pip install pycsou`` (Pycsou V1) may not work on some machines.
-  PSF are now always stored as 4D Data [depth, width, height, color], Data are stored as [(batch_size), depth, width, height, color] batch_size being optional. Grayscale data has a color axis of length 1 and 2D data has a depth axis of length 1.
-  Added batch support to RealFFTConvolve2D.
-  ``ReconstructionAlgorithm.update`` now take the number of the current iteration to allow for unrolled algorithms.
-  ``ReconstructionAlgorithm.apply`` now takes a reset parameter (default true) to automaticaly call reset.
-  Call to reset in ``ReconstructionAlgorithm.__init__`` is now optional (see reset parameter).
-  Make sure image estimate is reset when reset() is called, either to zeros/mean data or to self._initial_est if set.

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



