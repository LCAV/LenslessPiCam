=============
LenslessPiCam
=============
Ahmed Elalamy (324610), Seif Hamed (312081), Ghita Tagemouati (330383)

Our work is mainly done in the trainable_mask.py and mask.py files.

Setup 
-----

First, install the lensless package

.. code:: bash

   pip install lensless


*Local machine setup*
=====================

Below are commands that worked for our configuration (Ubuntu
21.04), but there are certainly other ways to download a repository and
install the library locally.

.. code:: bash

   # download from GitHub
   git clone git@github.com:LCAV/LenslessPiCam.git
   cd LenslessPiCam

   # create virtual environment (as of Oct 4 2023, rawpy is not compatible with Python 3.12)
   # -- using conda
   conda create -n lensless python=3.11
   conda activate lensless

   # -- OR venv
   python3.11 -m venv lensless_env
   source lensless_env/bin/activate

   # install package
   pip install -e .

   # extra dependencies for local machine for plotting/reconstruction
   pip install -r recon_requirements.txt
   pip install -r mask_requirements.txt

   # training with the height varying mask
   python scripts/recon/train_unrolled.py -cn train_heightvarying

