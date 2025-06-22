Lensless Imaging WebApp
========================

This web application provides an interactive interface for capturing, calibrating,
and reconstructing lensless images using the Lensless Imaging Development Kit.
It is designed to run on a Raspberry Pi and accessed through a browser on the same network.
No software installation is required on the host machine.

This module is located in the ``lenslesswebapp/`` directory of the main project.

Features
--------

- Full lensless imaging pipeline: PSF capture, autocorrelatio plotting, object capture, and image reconstruction
- Optional auto exposure and RGB gain correction 
- Remote capture and reconstruction
- Headless access via browser using preconfigured Raspberry Pi image
- No installation required on client machines


Project Structure
-----------------

::

    lensless-web-app/
    ├── client/               # Vite-based frontend (React + Tailwind)
    ├── server/               # Node.js backend (image processing, control)
    ├── public/               # Static assets
    ├── captures/             # PSFs and image captures
    ├── README.rst            # This file

Demo
----

Once the frontend server is running, the local IP address of the Raspberry Pi
and the full URL (e.g., ``http://192.168.1.42:5173``) will be printed in the terminal.

You must:

- SSH into the Raspberry Pi
- Run the frontend (`npm run dev`)
- Copy the printed IP address from the terminal
- Open the browser from another device on the same network

The IP address may change depending on your local network configuration.

Getting Started
---------------

1. Flash and Boot Raspberry Pi

- Use the provided `.img` file to flash your SD card.
- Insert the SD card into the Raspberry Pi and power it up.
- Ensure the Pi is connected to the same local network as your browser device.

2. Run the Application

Frontend (client)
~~~~~~~~~~~~~~~~~

Start the frontend development server:

.. code-block:: bash

    cd lenslesswebapp/
    npm install
    npm run dev

To keep the frontend running in the background using `screen`:

.. code-block:: bash

    screen -S client
    npm run dev

To detach: press `Ctrl + A`, then `D`  
To reattach later: `screen -r client`

Backend (server)
~~~~~~~~~~~~~~~~

Start the backend server:

.. code-block:: bash

    cd lenslesswebapp
    node server/index.cjs

To keep the backend running in the background using `screen`:

.. code-block:: bash

    screen -S server
    node server/index.cjs

To detach: press `Ctrl + A`, then `D`  
To reattach later: `screen -r server`

Module Features
---------------

PSF Capture Block
~~~~~~~~~~~~~~~~~

This block handles the acquisition and calibration of the Point Spread Function using a known white point light source.

Acquisition Options
^^^^^^^^^^^^^^^^^^^

- Capture a new PSF using the camera interface and plot its autocorrelation
- Load a previously saved PSF from disk with its autocorrelation
- Download a zip of the 2 pictures

Exposure and Color Correction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Automatic exposure tuning (`auto_exp_psf`)
- Automatic red and blue gain calibration 

Visualization Tools
^^^^^^^^^^^^^^^^^^^

- Autocorrelation plot generated to assess sharpness and quality
- Useful for visually validating PSF quality and compare with others

Image Capture & Reconstruction Block
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This block allows you to capture scene images and reconstruct them using a selected PSF.

Image Input Options
^^^^^^^^^^^^^^^^^^^

- Take a photo directly on your device
- Upload an image from your device
- Upload an image from your phone via the `/phone` page ( PS : to take selfies from your phone and retrieve them to reconstruct them on your computer)

Reconstruction Workflow
^^^^^^^^^^^^^^^^^^^^^^^

- Select a PSF (from all the captured ones)
- Reconstruct the captured image using that PSF using ADMM or Gradient descent with a selected number of iterations

Display and Feedback
^^^^^^^^^^^^^^^^^^^^

- View the raw lensless image and its reconstruction side-by-side
- Allows inspection of reconstruction quality in real time and can change number of iterations and the algorithm of reconstruction for comparison 


Author
------

Imane Raihane  
Semester Project, Spring 2025  
Supervisor: Eric Bezzam (LCAV – EPFL)
