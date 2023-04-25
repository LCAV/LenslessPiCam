Simulating raw data
===================

Check out `this Medium post <https://medium.com/@bezzam/simulating-camera-measurements-through-wave-optics-with-pytorch-support-faf3fa620789>`__
for a detailed explanation of how to simulate raw data of lensless cameras, given a digital image of a scene and point spread function (PSF).

In short, there are several scripts inside the `scripts/sim <https://github.com/LCAV/LenslessPiCam/tree/main/scripts/sim>`__
folder that can be used to simulate raw data. Behind the scenes, code from the `waveprop <https://pypi.org/project/waveprop/>`__
library is used with the following simulation steps:

#. **Prepare object plane**: resize and pad the original image according to the physical dimensions of the setup and camera sensor.
#. **Convolve with PSF**.
#. (Optionally) **downsample**: perhaps you use a higher resolution PSF than your actual camera sensor.
#. **Add noise**: e.g. shot noise to replicate noise at the sensor or Gaussian noise.
#. **Quantize** according to the bit depth of the sensor.

PyTorch support is available to speed up simulation on GPU, and to create Dataset and DataLoader objects for training and testing!

Simulating 3D data
------------------

Check out `this other Medium post <https://medium.com/@julien.sahli/3d-imaging-with-lensless-camera-822983618455>`__.

The corresponding code will likely be added soon in waveprop.