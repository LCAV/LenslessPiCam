=============
LenslessPiCam
=============

*A Hardware and Software Platform for Lensless Computational Imaging with a Raspberry Pi*
-----------------------------------------------------------------------------------------

.. image:: https://github.com/LCAV/LenslessPiCam/raw/main/scripts/recon/example.png
    :alt: Lensless imaging example
    :align: center

This package provides functionalities to perform imaging with a lensless
camera. We make use of a low-cost implementation of DiffuserCam, [1]_
where we use a piece of tape instead of the lens and the
`Raspberry Pi HQ camera sensor <https://www.raspberrypi.com/products/raspberry-pi-high-quality-camera>`__
(the `V2 sensor <https://www.raspberrypi.com/products/camera-module-v2/>`__
is also supported). However, the same principles can be used for a
different diffuser/mask and a different sensor (although the capture
script would change). 

*If you are interested in exploring reconstruction algorithms without building the camera, that is entirely possible!*
The provided reconstruction algorithms can be used with :ref:`the provided data<Already available data>`
or :ref:`simulated data<Simulating raw data>`.

We've also written a few Medium articles to guide users through the process
of building the camera, measuring data with it, and reconstruction.
They are all laid out in `this post <https://medium.com/@bezzam/a-complete-lensless-imaging-tutorial-hardware-software-and-algorithms-8873fa81a660>`__.

Note that this material has been used for our graduate signal processing
course at EPFL, and therefore includes some exercises / code to
complete: 

* ``lensless.autocorr.autocorr2d``: to compute a 2D autocorrelation in the frequency domain;
* ``lensless.realfftconv.RealFFTConvolve2D``: to pre-compute the PSF's Fourier transform, perform a convolution in the frequency domain with the real-valued FFT, and vectorize operations for RGB.

If you are an instructor, you can request access to the solutions
`here <https://drive.google.com/drive/folders/1Y1scM8wVfjVAo5-8Nr2VfE4b6VHeDSia?usp=sharing>`__
or send an email to ``eric[dot]bezzam[at]epfl[dot]ch``.

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
can be found here :ref:`here<Building the camera>`.

The software from this repository has to be installed on **both** your
local machine and the Raspberry Pi. Note that we highly recommend using
Python 3.9, as some Python library versions may not be available with 
earlier versions of Python. Moreover, its `end-of-life <https://endoflife.date/python>`__ 
is Oct 2025.

**Local machine**

Below are commands that worked for our configuration (Ubuntu
21.04), but there are certainly other ways to download a repository and
install the library locally.

.. code:: bash

   # download from GitHub
   git clone git@github.com:LCAV/LenslessPiCam.git

   # install in virtual environment
   cd LenslessPiCam
   python3 -m venv lensless_env
   source lensless_env/bin/activate
   pip install -e .

   # -- extra dependencies for local machine for plotting/reconstruction
   pip install -r recon_requirements.txt

   # (optional) try reconstruction on local machine
   python scripts/recon/admm.py --psf_fp data/psf/tape_rgb.png \
   --data_fp data/raw_data/thumbs_up_rgb.png --n_iter 5


**Raspberry Pi**

After `flashing your Raspberry Pi with SSH enabled <https://medium.com/@bezzam/setting-up-a-raspberry-pi-without-a-monitor-headless-9a3c2337f329>`__, 
you need to set it up for `passwordless access <https://medium.com/@bezzam/headless-and-passwordless-interfacing-with-a-raspberry-pi-ssh-453dd75154c3>`__. 
Do not set a password for your SSH key pair, as this will not work with the
provided scripts.

On the Raspberry Pi, you can then run the following commands (from the ``home`` 
directory):

.. code:: bash

   # download from GitHub
   git clone git@github.com:LCAV/LenslessPiCam.git

   # install in virtual environment
   cd LenslessPiCam
   python3 -m venv lensless_env
   source lensless_env/bin/activate
   pip install -e .


You may also have to install the following:

.. code:: bash

   sudo apt-get install libimage-exiftool-perl
   sudo apt-get install libatlas-base-dev



Acknowledgements
----------------

The idea of building a lensless camera from a Raspberry Pi and a piece of 
tape comes from Prof. Laura Waller's group at UC Berkeley. So a huge kudos 
to them for the idea and making tools/code/data available! Below is some of 
the work that has inspired this toolkit:

* `Build your own DiffuserCam tutorial <https://waller-lab.github.io/DiffuserCam/tutorial>`__.
* `DiffuserCam Lensless MIR Flickr dataset <https://waller-lab.github.io/LenslessLearning/dataset.html>`__. [2]_



Citing this work 
----------------

If you use these tools in your own research, please cite the following:

::

   @misc{lenslesspicam,
       url = {https://infoscience.epfl.ch/record/294041?&ln=en},
       author = {Bezzam, Eric and Kashani, Sepand and Vetterli, Martin and Simeoni, Matthieu},
       title = {Lensless{P}i{C}am: A Hardware and Software Platform for Lensless Computational Imaging with a {R}aspberry {P}i},
       publisher = {Infoscience},
       year = {2022},
   }

References
----------

.. [1] Antipa, N., Kuo, G., Heckel, R., Mildenhall, B., Bostan, E., Ng, R., & Waller, L. (2018). DiffuserCam: lensless single-exposure 3D imaging. Optica, 5(1), 1-9.

.. [2] Monakhova, K., Yurtsever, J., Kuo, G., Antipa, N., Yanny, K., & Waller, L. (2019). Learned reconstructions for practical mask-based lensless imaging. Optics express, 27(20), 28075-28090.
