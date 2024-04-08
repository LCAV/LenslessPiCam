=============
LenslessPiCam
=============

.. image:: https://readthedocs.org/projects/lensless/badge/?version=latest
    :target: http://lensless.readthedocs.io/en/latest/
    :alt: Documentation Status


.. image:: https://joss.theoj.org/papers/10.21105/joss.04747/status.svg
      :target: https://doi.org/10.21105/joss.04747
      :alt: DOI

.. image:: https://static.pepy.tech/badge/lensless
      :target: https://www.pepy.tech/projects/lensless
      :alt: Downloads


.. image:: https://colab.research.google.com/assets/colab-badge.svg
      :target: https://drive.google.com/drive/folders/1nBDsg86RaZIqQM6qD-612k9v8gDrgdwB?usp=drive_link
      :alt: notebooks

.. image:: https://huggingface.co/datasets/huggingface/badges/resolve/main/powered-by-huggingface-dark.svg
      :target: https://huggingface.co/bezzam
      :alt: huggingface


*A Hardware and Software Toolkit for Lensless Computational Imaging with a Raspberry Pi*
-----------------------------------------------------------------------------------------

.. image:: https://github.com/LCAV/LenslessPiCam/raw/main/scripts/recon/example.png
    :alt: Lensless imaging example
    :align: center


This toolkit has everything you need to perform imaging with a lensless
camera. We make use of a low-cost implementation of DiffuserCam [1]_, 
where we use a piece of tape instead of the lens and the
`Raspberry Pi HQ camera sensor <https://www.raspberrypi.com/products/raspberry-pi-high-quality-camera>`__
(the `V2 sensor <https://www.raspberrypi.com/products/camera-module-v2/>`__
is also supported). Similar principles and methods can be used for a
different lensless encoder and a different sensor. 

*If you are interested in exploring reconstruction algorithms without building the camera, that is entirely possible!*
The provided reconstruction algorithms can be used with the provided data or simulated data.

We've also written a few Medium articles to guide users through the process
of building the camera, measuring data with it, and reconstruction.
They are all laid out in `this post <https://medium.com/@bezzam/a-complete-lensless-imaging-tutorial-hardware-software-and-algorithms-8873fa81a660>`__.

Setup 
-----

If you are just interested in using the reconstruction algorithms and 
plotting / evaluation tools you can install the package via ``pip``:

.. code:: bash

   pip install lensless


For plotting, you may also need to install
`Tk <https://stackoverflow.com/questions/5459444/tkinter-python-may-not-be-configured-for-tk>`__.


For performing measurements, the expected workflow is to have a local 
computer which interfaces remotely with a Raspberry Pi equipped with 
the HQ camera sensor (or V2 sensor). Instructions on building the camera
can be found `here <https://lensless.readthedocs.io/en/latest/building.html>`__.

The software from this repository has to be installed on **both** your
local machine and the Raspberry Pi. Note that we highly recommend using
Python 3.9, as some Python library versions may not be available with 
earlier versions of Python. Moreover, its `end-of-life <https://endoflife.date/python>`__ 
is Oct 2025.

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

   # (optional) try reconstruction on local machine
   python scripts/recon/admm.py

   # (optional) try reconstruction on local machine with GPU
   python scripts/recon/admm.py -cn pytorch


Note (25-04-2023): for using the :py:class:`~lensless.recon.apgd.APGD` reconstruction method based on Pycsou
(now `Pyxu <https://github.com/matthieumeo/pyxu>`__), a specific commit has 
to be installed (as there was no release at the time of implementation):

.. code:: bash

   pip install git+https://github.com/matthieumeo/pycsou.git@38e9929c29509d350a7ff12c514e2880fdc99d6e

If PyTorch is installed, you will need to be sure to have PyTorch 2.0 or higher, 
as Pycsou is not compatible with earlier versions of PyTorch. Moreover, 
Pycsou requires Python within 
`[3.9, 3.11) <https://github.com/matthieumeo/pycsou/blob/v2-dev/setup.cfg#L28>`__.

Moreover, ``numba`` (requirement for Pycsou V2) may require an older version of NumPy:

.. code:: bash

   pip install numpy==1.23.5

*Raspberry Pi setup*
====================

After `flashing your Raspberry Pi with SSH enabled <https://medium.com/@bezzam/setting-up-a-raspberry-pi-without-a-monitor-headless-9a3c2337f329>`__, 
you need to set it up for `passwordless access <https://medium.com/@bezzam/headless-and-passwordless-interfacing-with-a-raspberry-pi-ssh-453dd75154c3>`__. 
Do not set a password for your SSH key pair, as this will not work with the
provided scripts.

On the Raspberry Pi, you can then run the following commands (from the ``home`` 
directory):

.. code:: bash

   # dependencies
   sudo apt-get install -y libimage-exiftool-perl libatlas-base-dev \
   python3-numpy python3-scipy python3-opencv
   sudo pip3 install -U virtualenv

   # download from GitHub
   git clone git@github.com:LCAV/LenslessPiCam.git

   # install in virtual environment
   cd LenslessPiCam
   virtualenv --system-site-packages -p python3 lensless_env
   source lensless_env/bin/activate
   pip install --no-deps -e .
   pip install -r rpi_requirements.txt

   # test on-device camera capture (after setting up the camera)
   source lensless_env/bin/activate
   python scripts/measure/on_device_capture.py

You may still need to manually install ``numpy`` and/or ``scipy`` with ``pip`` in case libraries (e.g. ``libopenblas.so.0``) cannot be detected.
   

Acknowledgements
----------------

The idea of building a lensless camera from a Raspberry Pi and a piece of 
tape comes from Prof. Laura Waller's group at UC Berkeley. So a huge kudos 
to them for the idea and making tools/code/data available! Below is some of 
the work that has inspired this toolkit:

* `Build your own DiffuserCam tutorial <https://waller-lab.github.io/DiffuserCam/tutorial>`__.
* `DiffuserCam Lensless MIR Flickr dataset <https://waller-lab.github.io/LenslessLearning/dataset.html>`__ [2]_. 

A few students at EPFL have also contributed to this project:

* Julien Sahli: support and extension of algorithms for 3D.
* Yohann Perron: unrolled algorithms for reconstruction.

Citing this work
----------------

If you use these tools in your own research, please cite the following:

::

   @article{Bezzam2023,
      doi = {10.21105/joss.04747},
      url = {https://doi.org/10.21105/joss.04747},
      year = {2023},
      publisher = {The Open Journal},
      volume = {8},
      number = {86},
      pages = {4747},
      author = {Eric Bezzam and Sepand Kashani and Martin Vetterli and Matthieu Simeoni},
      title = {LenslessPiCam: A Hardware and Software Platform for Lensless Computational Imaging with a Raspberry Pi},
      journal = {Journal of Open Source Software}
   }

References
----------

.. [1] Antipa, N., Kuo, G., Heckel, R., Mildenhall, B., Bostan, E., Ng, R., & Waller, L. (2018). DiffuserCam: lensless single-exposure 3D imaging. Optica, 5(1), 1-9.

.. [2] Monakhova, K., Yurtsever, J., Kuo, G., Antipa, N., Yanny, K., & Waller, L. (2019). Learned reconstructions for practical mask-based lensless imaging. Optics express, 27(20), 28075-28090.
