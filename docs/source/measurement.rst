Measurement
===========

The goal of this section is to provide a guide on how to measure data with *LenslessPiCam* and a Raspberry Pi-based lensless camera.

`This Medium article <https://medium.com/@bezzam/measuring-a-diffusercam-psf-and-raw-data-b01ee29eda4>`__
discusses the typical measurement process for a lensless camera:

#. Measure the point spread function (PSF), which involves some calibration. See `this article <https://medium.com/@bezzam/measuring-an-optical-psf-with-an-arduino-an-led-and-a-cardboard-box-2f3ddac660c1>`__ for how to make a low-cost point source generator.
#. Measure raw data.

All scripts related to measurement can be found in the `scripts/measure <https://github.com/LCAV/LenslessPiCam/tree/main/scripts/measure>`__ 
directory. Some scripts are meant to be run on your local machine, while others are meant to be run on the Raspberry Pi.

.. note::

   Before trying to collect a dataset, you should first go through each section below to ensure that the camera is working correctly and that you are able to capture and display images.


It is highly recommended to connect to your Raspberry Pi through an IDE like VS Code, as it will be much easier to edit configuration files.


Capturing from your Raspberry Pi (on-device capture)
----------------------------------------------------

The on-device scripts need to work before you can perform remote capture.

*LenslessPiCam* should be cloned and installed on your Raspberry Pi, see :ref:`here<Setup>` (scroll down to "Raspberry Pi setup").

The following script can be used to capture data with the Raspberry Pi camera:

.. code:: bash

   python scripts/on_device_capture.py

Default parameters can be found in `configs/capture.yaml <https://github.com/LCAV/LenslessPiCam/blob/main/configs/capture.yaml>`__.
A description of each parameters can be found in this `pull request <https://github.com/LCAV/LenslessPiCam/pull/104/files#diff-5c2872943948eced6af96fa06f035330c48e400bb63f9cb3add461714420cc11>`__.

.. note::

   Note if using the *Legacy Camera* on Bullseye OS, you should set ``legacy=True`` in your configuration file.


Capturing from your local computer (remote capture)
---------------------------------------------------

For remote capture, you need to have *LenslessPiCam* installed on your local machine (see :ref:`here<Setup>`),
and you need to have `passwordless access <https://medium.com/@bezzam/headless-and-passwordless-interfacing-with-a-raspberry-pi-ssh-453dd75154c3>`__
to your Raspberry Pi.

You can remotely capture data with the following script:

.. code:: bash

   python scripts/measure/remote_capture.py \
      rpi.username=USERNAME \
      rpi.hostname=HOSTNAME \
      plot=True

where ``<HOSTNAME>`` is the hostname or IP address of your Raspberry Pi.
The script will save the captured data (``raw_data.png``) and plots
(including histogram to check for saturation) in ``demo_lensless``.

The default configuration (`config/demo.yaml <https://github.com/LCAV/LenslessPiCam/blob/main/configs/demo.yaml>`__)
converts the captured Bayer data to RGB and downsamples the image by a factor of 4 on the 
Raspberry Pi, and then sends it back to your computer. 
The `capture <https://github.com/LCAV/LenslessPiCam/blob/b863d265fd69e6ded140ccfc36c9accbe562de87/configs/demo.yaml#L32>`_ section, 
essentially sets the parameters for the on-device capture configation.

The default configuration uses white balancing gains that were 
determined for our LCD-based lensless camera (``capture.awb_gains``,
``camera.red_gain``, and ``camera.blue_gain``). You will very likely
need to determine these gains for your lensless camera by first capturing
the raw `Bayer data <https://medium.com/@bezzam/bayer-capture-and-processing-with-the-raspberry-pi-hq-camera-in-python-8496fed9dcb7>`__
of a known white object (e.g., a white sheet of paper).

To capture the raw 12-bit Bayer data for the Raspberry Pi HQ camera, 
you can run the above script with a different configation:

.. code:: bash

   python scripts/measure/remote_capture.py -cn capture_bayer \
      rpi.username=USERNAME \
      rpi.hostname=HOSTNAME


You can then use ``scripts/measure/analyze_image.py`` to play around with the red and
blue gains to find the best white balance for your camera.

Preparing an external monitor for displaying images (remote display)
--------------------------------------------------------------------

Our approach for displaying images on a monitor connected to a Raspberry Pi is to
view an image in full-screen mode using the `feh <https://feh.finalrewind.org/>`__ command line tool,
which reads a folder of images and displays them in a slideshow.
Our folder only has one image, and we rewrite this image every time we want to display a new image.

You first need to install the ``feh`` command line tool on your Raspberry Pi.

.. code:: bash

   sudo apt-get install feh

Then make a folder where we will create and read prepared images.

.. code:: bash

   mkdir ~/LenslessPiCam_display
   cp ~/LenslessPiCam/data/original/mnist_3.png ~/LenslessPiCam_display/test.png

We will be overwriting ``~/LenslessPiCam_display/test.png`` whenever we want to display a new image.
Then **from a Terminal on the Raspberry Pi** we can use ``feh`` to launch the image viewer.

.. code:: bash

   feh LenslessPiCam_display --scale-down --auto-zoom -R 0.1 -x -F -Y

This command cannot be launched from an SSH session connected to the Raspberry Pi, even if there is a monitor connected to the Raspberry Pi.

Then *from your laptop with passwordless access*, you can use the following script to display an
image on the Raspberry Pi:

.. code-block:: bash

    python scripts/measure/remote_display.py \
        rpi.username=USERNAME \
        rpi.hostname=HOSTNAME \
        fp=FP

where ``USERNAME`` and ``HOSTNAME`` are the username and hostname of the RPi,
and ``FP`` is the path on your local computer of the image you would like
to display. The default parameters can be found in ``config/demo.yaml``,
specifically the `display <https://github.com/LCAV/LenslessPiCam/blob/b863d265fd69e6ded140ccfc36c9accbe562de87/configs/demo.yaml#L16>`_ section, 
to adjust the screen resolution, positioning, brightness, padding, and rotation.


Measuring an arbitrary dataset
------------------------------

.. note::

   Please go through the above sections to ensure that the camera is working correctly and that you are able to capture and display images.


The following script can be used to collect an arbitrary dataset with the proposed camera.
The script should be launched **from the Raspberry Pi**:

.. code:: 

   python scripts/measure/collect_dataset_on_device.py -cn CONFIG_NAME

The default configuration can be found in `configs/collect_dataset.yaml <https://github.com/LCAV/LenslessPiCam/blob/main/configs/collect_dataset.yaml>`__.


The following needs to be done before running the script:

#. Determine the white balance gains for your camera, e.g. with ``scripts/measure/analyze_image.py`` as described above.
#. As you probably won't have enough memory on the Raspberry Pi, `mount <https://thepihut.com/blogs/raspberry-pi-tutorials/how-to-mount-an-external-hard-drive-on-the-raspberry-pi-raspian>`__ an external hard-drive to save the dataset into.

   .. code:: bash

      sudo mount /dev/sda1 /mnt

   Change ``/dev/sda1`` to the correct device name of your external hard-drive.

#. Define your own configuration file inside the ``configs`` directory. You can "inherit" from the default configuration file and overwrite the parameters you want to change:

   .. code:: yaml

      defaults:
         - collect_dataset
         - _self_

      input_dir: /mnt/YOUR_INPUT_DATASET
      output_dir: /mnt/YOUR_OUTPUT_DATASET
      n_files: 10000
      min_level: 140
      max_tries: 4

      display:
         image_res: [900, 1200]

      capture:
         exposure: 0.5
         awb_gains: YOUR_AWB_GAINS


You can schedule the dataset collection with ``runtime`` and ``start_delay``.

By default, the script will save downloaded RGB data, as raw Bayer data can quickly take up a lot of space. 

See `configs/collect_mirflickr_multimask.yaml <https://github.com/LCAV/LenslessPiCam/blob/main/configs/collect_mirflickr_multimask.yaml>`__ 
for an example of measuring a multi-mask dataset with DigiCam.

.. note::

   Play around with the display/capture settings until you get the ones that align well for an ADMM reconstruction,
   and that require the least amount of re-tries.

.. note::

   As the measurement may run for a while, it is recommended to run the script in a `screen <https://linuxize.com/post/how-to-use-linux-screen/>`__ session (or similar) to avoid the script being interrupted if the SSH connection is lost.

.. note::

   From the Raspberry Pi, it is recommended to disable screen blanking (the screen from entering
   power saving mode and turning off) by following these `steps <https://pimylifeup.com/raspberry-pi-disable-screen-blanking/>`__.


Troubleshooting
---------------

The scripts used below assume that you have :ref:`setup<Setup>` the package on **both** your 
local machine and a remote Raspberry Pi with SSH and
`passwordless access <https://medium.com/@bezzam/headless-and-passwordless-interfacing-with-a-raspberry-pi-ssh-453dd75154c3>`__.

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

