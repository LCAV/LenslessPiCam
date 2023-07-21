Utilities
=========


Data loading
------------

.. automodule:: lensless.utils.io
    :members:
    :undoc-members:
    :show-inheritance:

Plotting
--------

.. automodule:: lensless.utils.plot
    :members:
    :undoc-members:
    :show-inheritance:

Image processing
----------------

.. autofunction:: lensless.utils.image.resize

.. autofunction:: lensless.utils.image.bayer2rgb

.. autofunction:: lensless.utils.image.rgb2gray

.. autofunction:: lensless.utils.image.gamma_correction


Image analysis
--------------

.. autofunction:: lensless.utils.image.autocorr2d

.. autofunction:: lensless.utils.image.print_image_info

Dataset
------------

.. automodule:: lensless.utils.dataset

.. autoclass:: lensless.utils.dataset.DualDataset
    :members: _get_images_pair
    :special-members: __init__, __len__

.. autoclass:: lensless.utils.dataset.LenslessDataset
    :members:
    :special-members: __init__

.. autoclass:: lensless.utils.dataset.SimulatedDataset
    :members:
    :special-members: __init__

.. autoclass:: lensless.utils.dataset.DiffuserCamTestDataset
    :members:
    :special-members: __init__