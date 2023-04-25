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


3D data
-------

You can download example 3D PSF and raw data from the Waller lab
`here  <https://github.com/Waller-Lab/DiffuserCam/tree/master/example_data>`__.
The PSF has to be converted from .mat to .npy in order to be usable :

.. code:: bash

    python scripts/data/3d/mat_to_npy.py ~/path/to/example_psfs.mat
	

Once you have run a reconstruction, you may want to convert the
resulting .npy files in separate tiff images for each depth.
This can be done with the following script :

.. code:: bash

	python scripts/data/3d/npy_to_tiff.py ~path/to/output.npy


You may also want to export it into a wavefront .obj file
for it to be displayed in 3D rendering softwares with the following
scrpit. It mostly exists to allow the user to preview it and is not
totally accurate as the problem of converting discrete pixels in a
"continuous" wavefront object is subject to interpreatation :

.. code:: bash

	python scripts/data/3d/npy_to_obj.py ~/path/to/output.npy
	