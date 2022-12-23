Measurement
===========

`This Medium article <https://medium.com/@bezzam/measuring-a-diffusercam-psf-and-raw-data-b01ee29eda4>`__
discusses the typical measurement process for a lensless camera:

#. Measure the point spread function (PSF), which involves some calibration. See `this article <https://medium.com/@bezzam/measuring-an-optical-psf-with-an-arduino-an-led-and-a-cardboard-box-2f3ddac660c1>`__ for how to make a low-cost point source generator.
#. Measure raw data.

With some data in hand, :ref:`reconstructions algorithms<Reconstruction>` can be explored!

The scripts used below assume that you have :ref:`setup<Setup>` the package on **both** your 
local machine and a remote Raspberry Pi with SSH and passwordless access.

Sometimes, we have noticed problems with locale when running the remote capture and
display scripts, for example:

.. code:: bash

   perl: warning: Setting locale failed.
   perl: warning: Please check that your locale settings:
   ...

This may arise due to incompatible locale settings between your local
machine and the Raspberry Pi. There are two possible solutions to this,
as proposed in `this
forum <https://forums.raspberrypi.com/viewtopic.php?t=11870>`__. 

#. Comment ``SendEnv LANG LC_*`` in ``/etc/ssh/ssh_config`` on your laptop.
#. Comment ``AcceptEnv LANG LC_*`` in ``/etc/ssh/sshd_config`` on the Raspberry Pi.

Remote capture 
--------------

You can remotely capture raw `Bayer data <https://medium.com/@bezzam/bayer-capture-and-processing-with-the-raspberry-pi-hq-camera-in-python-8496fed9dcb7>`__ 
with the following script.

.. code:: bash

   python scripts/remote_capture.py --exp 0.1 --iso 100 --bayer --fn <FN> --hostname <HOSTNAME>

where ``<HOSTNAME>`` is the hostname or IP address of your Raspberry Pi,
``<FN>`` is the name of the file to save the Bayer data, and the other
arguments can be used to adjust camera settings.

Note if using the *Legacy Camera* on Bullseye OS, you should include the
``--legacy`` flag as well!

Remote display 
--------------

For collecting images displayed on a screen, we have prepared some
software to remotely display images on a Raspberry Pi installed with
this software and connected to a monitor.

You first need to install the ``feh`` command line tool on your
Raspberry Pi.

.. code:: bash

   sudo apt-get install feh

Then make a folder where we will create and read prepared images.

.. code:: bash

   mkdir ~/LenslessPiCam_display
   cp ~/LenslessPiCam/data/original/mnist_3.png ~/LenslessPiCam_display/test.png

Then we can use ``feh`` to launch the image viewer.

.. code:: bash

   feh LenslessPiCam_display --scale-down --auto-zoom -R 0.1 -x -F -Y

Then from your laptop you can use the following script to display an
image on the Raspberry Pi:

.. code:: bash

   python scripts/remote_display.py --fp <FP> --hostname <HOSTNAME> \
   --pad 80 --vshift 10 --brightness 90

where ``<HOSTNAME>`` is the hostname or IP address of your Raspberry Pi,
``<FP>`` is the path on your local computer of the image you would like
to display, and the other arguments can be used to adjust the
positioning of the image and its brightness.

When collecting a dataset, you can disable screen blanking (the screen
from entering power saving mode) by following these `steps <https://pimylifeup.com/raspberry-pi-disable-screen-blanking/>`__.

Collecting MNIST 
----------------

We provide a couple scripts to collect MNIST with the proposed camera.

Script that can be launched from the Raspberry Pi:

.. code:: bash

   python scripts/collect_mnist_on_device.py --input_dir MNIST_original \
   --output_dir MNIST_meas

If the MNIST dataset is not available at ``MNIST_original`` it will be
downloaded from `here <http://yann.lecun.com/exdb/mnist/>`__. The above
command will measure the training set. The ``--test`` flag can be used
to measure the test set. It is recommended to run the script from a
`screen <https://linuxize.com/post/how-to-use-linux-screen/>`__
session as it takes a long time to go through all the files! The
``--n_files <N_FILES>`` option can be used to measure a user-specified
amount of files.

To remotely collect the MNIST dataset (although quite slow due to
copying files back and forth):

.. code:: bash

   python scripts/collect_mnist.py --hostname <IP_ADDRESS> --output_dir MNIST_meas