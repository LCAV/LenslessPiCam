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

.. note::

   Note if using the *Legacy Camera* on Bullseye OS, you should set ``legacy=True`` in your configuration file.


You can remotely capture data with the following script:

.. code:: bash

   python scripts/remote_capture.py \
      rpi.username=USERNAME \
      rpi.hostname=HOSTNAME \
      plot=True

where ``<HOSTNAME>`` is the hostname or IP address of your Raspberry Pi.
The script will save the captured data (``raw_data.png``) and plots
(including histogram to check for saturation) in ``demo_lensless``.

The default configuration (``config/demo.yaml``) converts the captured
Bayer data to RGB and downsamples the image by a factor of 4 on the 
Raspberry Pi, and then sends it back to your computer. 

Moreover, this configuration uses white balancing gains that were 
determined for our lensless camera (``capture.awb_gains``,
``camera.red_gain``, and ``camera.blue_gain``). You will very likely
need to determine these gains for your lensless camera by first capturing
the raw `Bayer data <https://medium.com/@bezzam/bayer-capture-and-processing-with-the-raspberry-pi-hq-camera-in-python-8496fed9dcb7>`__
of a known white object (e.g., a white sheet of paper).

To capture the raw 12-bit Bayer data for the Raspberry Pi HQ camera, 
you can run the above script with a different configation:

.. code:: bash

   python scripts/remote_capture.py -cn capture_bayer \
      rpi.username=USERNAME \
      rpi.hostname=HOSTNAME


You can then use ``scripts/analyze_image.py`` to play around with the red and
blue gains to find the best white balance for your camera.


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


.. code-block:: bash

    python scripts/remote_display.py \
        rpi.username=USERNAME \
        rpi.hostname=HOSTNAME \
        fp=FP

where ``USERNAME`` and ``HOSTNAME`` are the username and hostname of the RPi,
and ``FP`` is the path on your local computer of the image you would like
to display. The default parameters can be found in ``config/demo.yaml``,
specifically the ``display`` section, where you may be interested in
adjusting the screen resolution, positioning, brightness, padding, and
rotation.

.. note::

   It is recommended to disable screen blanking (the screen from entering
   power saving mode and turning off) by following these `steps <https://pimylifeup.com/raspberry-pi-disable-screen-blanking/>`__.

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


Collecting arbitrary dataset
----------------------------

We provide a script to collect an arbitrary dataset with the proposed
camera. The script can be launched **from the Raspberry Pi**:

.. code:: 

   python scripts/collect_dataset_on_device.py

By default this script will collect a subset (100 files) of the `CelebA <https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>`__
dataset.

The default configuration can be found in ``configs/collect_dataset.yaml``. You can
change the dataset by changing the ``input_dir`` and ``input_file_ext`` to set
the directory and file extension of the dataset you would like to collect. You
can schedule the dataset collection with ``runtime`` and ``start_delay``.

As raw Bayer data can quickly take up a lot of space, the script will save downloaded
RGB data.

.. note::

   To convert to RGB correctly, you need to determine your white balance gains as described in the :ref:`Remote capture section<Remote capture>`.

You may also need to `mount <https://thepihut.com/blogs/raspberry-pi-tutorials/how-to-mount-an-external-hard-drive-on-the-raspberry-pi-raspian>`__
an external hard-drive on the Raspberry Pi to save the dataset into.

.. code:: bash

   sudo mount /dev/sda1 /mnt

Change ``/dev/sda1`` to the correct device name of your external hard-drive.
