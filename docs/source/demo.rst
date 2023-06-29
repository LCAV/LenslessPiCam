Demo
====

A full demo script can be found in ``scripts/demo.py``. Its configuration
file can be found in ``configs/demo.yaml``.

It assumes the following setup:

* You have a Raspberry Pi (RPi) with ``LenslessPiCam`` installed.
* You have a PC with ``LenslessPiCam`` installed.
* The RPi and the PC are connected to the same network.
* You can SSH into the RPi from the PC without a password.
* The RPi is connected to a lensless camera and a display.
* The PSF of the lensless camera is known and saved as an RGB file.

.. image:: demo_setup.png
    :alt: Example setup for the demo.
    :align: center

With the above setup, you can run the demo script on the PC to:

#. Display an image on the screen.
#. Capture an image from the lensless camera.
#. Send the captured image to the PC.
#. Perform the reconstruction on the PC.

with the following command:

.. code-block:: bash

    python scripts/demo.py \
        rpi.username=USERNAME \
        rpi.hostname=HOSTNAME \
        camera.psf=PSF_FP \
        fp=data/original/mnist_3.png \
        recon.algo=admm \
        recon.downsample=8 \
        recon.use_torch=True \
        plot=True

where ``USERNAME`` and ``HOSTNAME`` are the username and hostname of the RPi,
and ``PSF_FP`` is the path to the PSF file.

Note that there is a custom post-processing by default to extract a 
specific region. You may want to modify this (``postproc.crop_hor``
and ``postproc.crop_vert``).

For your setup, you may also need to change the display settings
(resolution, brightness, etc) and the camera settings (exposure, etc).

Below are example outputs from our setup and the above script.

.. image:: https://github.com/LCAV/LenslessPiCam/raw/main/scripts/recon/example.png
    :alt: Example demo output.
    :align: center


Telegram demo
-------------

You can also run the demo remotely using Telegram. To do so, you need to: