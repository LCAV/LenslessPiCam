Already available data
======================

You can download example PSFs and raw data that we've measured
`here <https://drive.switch.ch/index.php/s/NdgHlcDeHVDH5ww>`__. We
recommend placing this content in the ``data`` folder.

The commands below to download the data and place it in the``data`` 
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


Dataset collected by other people
---------------------------------

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
