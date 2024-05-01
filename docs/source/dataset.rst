Dataset objects (for training and testing)
==========================================

The software below provides functionality (with PyTorch) to load
datasets for training and testing.

.. automodule:: lensless.utils.dataset

Abstract base class
-------------------

All dataset objects derive from this abstract base class, which
lays out the notion of a dataset with pairs of images: one image
is lensed (simulated or measured), and the other is lensless (simulated
or measured).

.. autoclass:: lensless.utils.dataset.DualDataset
    :members: _get_images_pair
    :special-members: __init__, __len__


Measured dataset objects
------------------------

.. autoclass:: lensless.utils.dataset.HFDataset
    :members:
    :special-members: __init__

.. autoclass:: lensless.utils.dataset.MeasuredDataset
    :members:
    :special-members: __init__

.. autoclass:: lensless.utils.dataset.MeasuredDatasetSimulatedOriginal
    :members:
    :special-members: __init__

.. autoclass:: lensless.utils.dataset.DiffuserCamTestDataset
    :members:
    :special-members: __init__


Simulated dataset objects
-------------------------

These dataset objects can be used for training and testing with 
simulated data. The main assumption is that the imaging system 
is linear shift-invariant (LSI), and that the lensless image is
the result of a convolution of the lensed image with a point-spread
function (PSF). Check out `this Medium post <https://medium.com/@bezzam/simulating-camera-measurements-through-wave-optics-with-pytorch-support-faf3fa620789>`__
for more details on the simulation procedure.

With simulated data, we can avoid the hassle of collecting a large
amount of data. However, it's important to note that the LSI assumption
can sometimes be too idealistic, in particular for large angles.

Nevertheless, simulating data is the only option of learning the 
mask / PSF.

.. autoclass:: lensless.utils.dataset.SimulatedFarFieldDataset
    :members:
    :special-members: __init__

.. autoclass:: lensless.utils.dataset.SimulatedDatasetTrainableMask
    :members:
    :special-members: __init__
