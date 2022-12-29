Already available data
======================

You can download example PSFs and raw data that we've measured
`here <https://drive.switch.ch/index.php/s/NdgHlcDeHVDH5ww>`__. We
recommend placing this content in the ``data`` folder.

You can download a subset for the `DiffuserCam Lensless Mirflickr
Dataset <https://waller-lab.github.io/LenslessLearning/dataset.html>`__
that we've prepared
`here <https://drive.switch.ch/index.php/s/vmAZzryGI8U8rcE>`__ with
``scripts/prepare_mirflickr_subset.py``. The original dataset is quite 
large (25000 files, 100 GB). So we've prepared a more manageable
dataset (200 files, 725 MB). It was prepared with the following script:

.. code:: bash

    python scripts/prepare_mirflickr_subset.py \
    --data ~/Documents/DiffuserCam/DiffuserCam_Mirflickr_Dataset
