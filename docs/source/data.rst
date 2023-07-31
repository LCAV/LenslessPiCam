Measured data
=============

You can download PSFs and raw data that we've measured
`here <https://drive.switch.ch/index.php/s/NdgHlcDeHVDH5ww>`__. We
recommend placing this content in the ``data`` folder.

The commands below to download the data and place it in the ``data`` 
folder.

.. code:: bash

    wget https://drive.switch.ch/index.php/s/NdgHlcDeHVDH5ww/download -O data.zip
    unzip data.zip -d data
    cp -r data/*/* data/
    rm -rf data/LenslessPiCam_GitHub_data
    rm data.zip


The commands below perform a reconstruction with the above data. Be sure to 
use the correct PSF file for the data you're using!

.. code:: bash

    # field of view
    python scripts/recon/gradient_descent.py -cn in_the_wild

    # mug
    python scripts/recon/gradient_descent.py -cn in_the_wild \
    input.data=data/raw_data/mug_rgb_31032023.png

    # plant
    python scripts/recon/gradient_descent.py -cn in_the_wild \
    input.data=data/raw_data/plant_rgb_31032023.png

    # thumbs up
    python scripts/recon/gradient_descent.py -cn in_the_wild \
    input.data=data/raw_data/thumbs_up_rgb.png \
    input.psf=data/psf/tape_rgb.png


DiffuserCam Lensless Mirflickr Dataset (DLMD)
---------------------------------------------

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

You can download example 3D PSF and raw data from Prof. Laura Waller's lab
`here  <https://github.com/Waller-Lab/DiffuserCam/tree/master/example_data>`__,
or by running the commands at the beginning of this page to download all
the example data.

Their PSF has to be converted from ``.mat`` to ``.npy`` in order to be usable:

.. code:: bash

    # replace path to .mat file if different
    python scripts/data/3d/mat_to_npy.py data/psf/waller_3d_psfs.mat


The following command can be used to run a reconstruction on the 3D data:

.. code:: bash

    python scripts/recon/gradient_descent.py \
    input.data=data/raw_data/waller_3d_raw.png \
    input.psf=psf.npy preprocess.downsample=1 \
    -cn pytorch   # if pytorch is available with GPU


You can also perform a 3D reconstruction on data we have simulated:

.. code:: bash

    # 3D LCAV logo
    python scripts/recon/gradient_descent.py \
    input.data=data/raw_data/3d_sim.png \
    input.psf=data/psf/3d_sim.npz \
    -cn pytorch   # if pytorch is available with GPU

Once you have run a reconstruction, you may want to convert the
resulting ``.npy`` files in separate ``.tiff`` images for each depth.
This can be done with the following script:

.. code:: bash

	python scripts/data/3d/npy_to_tiff.py ~path/to/output.npy


You may also want to export it into a wavefront ``.obj`` file
for it to be displayed in 3D rendering softwares with the following
script. It mostly exists to allow the user to preview it and is not
100% accurate, as there are multiple approach to interpolate discrete 
pixels into a "continuous" wavefront:

.. code:: bash

	python scripts/data/3d/npy_to_obj.py ~/path/to/output.npy
	
