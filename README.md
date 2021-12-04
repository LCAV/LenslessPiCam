# DiffuserCam

### _Lensless imaging with a Raspberry Pi and a piece of tape!_

![Example PSF, raw data, and reconstruction.](scripts/example_reconstruction.png)

This package provides functionalities to perform imaging and reconstruction
with a lensless camera known as DiffuserCam [[1]](#1). We use a more rudimentary
version of the DiffuserCam where we use a piece of tape instead of the lens and 
the [Raspberry Pi HQ camera sensor](https://www.raspberrypi.com/products/raspberry-pi-high-quality-camera)
([V2 sensor](https://www.raspberrypi.com/products/camera-module-v2/) is also
supported). However, the same principles can be used for a different diffuser 
and a different sensor (although the capture script would change). The content of 
this project is largely based off of the work from Prof. Laura Waller's group at
UC Berkeley:
- [Build your own DiffuserCam tutorial](https://waller-lab.github.io/DiffuserCam/tutorial).
- [DiffuserCam lensless MIR Flickr dataset](https://waller-lab.github.io/LenslessLearning/dataset.html) [[2]](#2).

So a huge kudos to them for the idea and making the tools/code/data available!

We've also made a few Medium articles to guide you through the process of
building the DiffuserCam, measuring data with it, and reconstruction:
1. [Raspberry Pi setup](https://medium.com/@bezzam/setting-up-a-raspberry-pi-without-a-monitor-headless-9a3c2337f329) and [SSH'ing without password](https://medium.com/@bezzam/headless-and-passwordless-interfacing-with-a-raspberry-pi-ssh-453dd75154c3) (needed for the remote capture/display scripts).
2. [Building DiffuserCam](https://medium.com/@bezzam/building-a-diffusercam-with-the-raspberry-hq-camera-cardboard-and-tape-896b6020aff6).
3. [Measuring DiffuserCam PSF and raw data](https://medium.com/@bezzam/measuring-a-diffusercam-psf-and-raw-data-b01ee29eda4).
4. [Imaging with DiffuserCam](https://medium.com/@bezzam/lensless-imaging-with-the-raspberry-pi-and-python-diffusercam-473e47662857).

Note that this material has been prepared for our graduate signal processing 
course at EPFL, and therefore includes some exercises / code to complete. If you
are an instructor or trying to replicate this tutorial, feel free to send an 
email to `eric[dot]bezzam[at]epfl[dot]ch`.

## Setup
The expected workflow is to have a local computer which interfaces remotely
with a Raspberry Pi equipped with the HQ camera sensor (or V2 sensor as in the 
original tutorial).

The software from this repository has to be installed on your both your local 
machine and the Raspberry Pi (from the `home` directory of the Pi):
```bash
# download from GitHub
git clone git@github.com:LCAV/DiffuserCam.git

# install in virtual environment
cd DiffuserCam
python3.9 -m venv diffcam_env
source diffcam_env/bin/activate
pip install -e .
```

On the Raspberry Pi, you may also have to install the following:
```bash
sudo apt-get install libimage-exiftool-perl
sudo apt-get install libatlas-base-dev
```

Note that we highly recommend using Python 3.9, as its [end-of-life](https://endoflife.date/python) is Oct 2025. Some Python library versions may not be available with earlier versions of Python.

For plotting on your local computer, you may also need to [install Tk](https://stackoverflow.com/questions/5459444/tkinter-python-may-not-be-configured-for-tk).

The scripts for remote capture and remote display assume that you can SSH to the
Raspberry Pi without a password. To see this up you can follow instruction from
[this page](https://medium.com/@bezzam/headless-and-passwordless-interfacing-with-a-raspberry-pi-ssh-453dd75154c3).

## Data for examples

You can download example PSFs and raw data that we've measured [here](https://drive.switch.ch/index.php/s/NdgHlcDeHVDH5ww).
We recommend placing this content in the `data` folder.

You can download a subset for the [DiffuserCam Lensless Mirflickr Dataset](https://waller-lab.github.io/LenslessLearning/dataset.html)
that we've prepared [here](https://drive.switch.ch/index.php/s/vmAZzryGI8U8rcE)
with `scripts/prepare_mirflickr_subset.py`.

## Reconstruction

There is one script / algorithm available for reconstruction - ADMM [[3]](#3).
```bash
python scripts/admm.py --psf_fp data/psf/diffcam_rgb.png \
--data_fp data/raw_data/thumbs_up_rgb.png --n_iter 5
```

A template - `scripts/reconstruction_template.py` - can be used to implement
other reconstruction approaches. Moreover, the abstract class 
[`diffcam.recon.ReconstructionAlgorithm`](https://github.com/LCAV/DiffuserCam/blob/70936c1a1d0797b50190d978f8ece3edc7413650/diffcam/recon.py#L9)
can be used to define new reconstruction algorithms by defining a few methods:
forward model and its adjoint (`_forward` and `_backward`), the update step
(`_update`), a method to reset state variables (`_reset`), and an image
formation method (`_form_method`). Functionality for iterating, saving, and 
visualization are implemented within the abstract class. [`diffcam.admm.ADMM`](https://github.com/LCAV/DiffuserCam/blob/70936c1a1d0797b50190d978f8ece3edc7413650/diffcam/admm.py#L6)
shows an example reconstruction implementation deriving from this abstract 
class.

## Evaluating on a dataset

You can run ADMM on the [DiffuserCam Lensless Mirflickr Dataset](https://waller-lab.github.io/LenslessLearning/dataset.html)
with the following script.
```bash
python scripts/evaluate_mirflickr_admm.py --data <FP>
```
where `<FP>` is the path to the dataset.

However, the original dataset is quite large (25000 files, 100 GB). So we've 
prepared [this subset](https://drive.switch.ch/index.php/s/vmAZzryGI8U8rcE) (200
files, 725 MB) which you can also pass to the script. It is also possible to 
set the number of files.
```bash
python scripts/evaluate_mirflickr_admm.py \
--data DiffuserCam_Mirflickr_200_3011302021_11h43_seed11 \
--n_files 10 --save
```
The `--save` flag will save a viewable image for each reconstruction.

You can also apply ADMM on a single image and visualize the iterative 
reconstruction.
```bash
python scripts/apply_admm_single_mirflickr.py \
--data DiffuserCam_Mirflickr_200_3011302021_11h43_seed11 \
--fid 172
```

## Remote capture

You can remotely capture raw Bayer data with the following script.
```bash
python scripts/remote_capture.py --exp 0.1 --iso 100 --bayer --fp <FN> --hostname <HOSTNAME>
```
where `<HOSTNAME>` is the hostname or IP address of your Raspberry Pi, `<FN>` is
the name of the file to save the Bayer data, and the other arguments can be used
to adjust camera settings.

## Remote display

For collecting images displayed on a screen, we have prepared some software to
remotely display images on a Raspberry Pi installed with this software and
connected to a monitor.

You first need to install the `feh` command line tool on your Raspberry Pi.
```bash
sudo apt-get install feh
```

Then make a folder where we will create and read prepared images.
```bash
mkdir DiffuserCam_display
mv ~/DiffuserCam/data/original_images/rect.jpg ~/DiffuserCam_display/test.jpg
```

Then we can use `feh` to launch the image viewer.
```bash
feh DiffuserCam_display --scale-down --auto-zoom -R 0.1 -x -F -Y
```

Then from your laptop you can use the following script to display an image on
the Raspberry Pi:
```bash
python scripts/remote_display.py --fp <FP> --hostname <HOSTNAME> \
--pad 80 --vshift 10 --brightness 90
```
where `<HOSTNAME>` is the hostname or IP address of your Raspberry Pi, `<FN>` is
the path on your local computer of the image you would like to display, and the 
other arguments can be used to adjust the positioning of the image and its
brightness.

## Formatting
You can use `black` to format your code.

First install the library.
```bash
pip install black
```
Then run the formatting script we've prepared.
```bash
./format_code.sh
```

## References
<a id="1">[1]</a> 
Antipa, N., Kuo, G., Heckel, R., Mildenhall, B., Bostan, E., Ng, R., & Waller, L. (2018). DiffuserCam: lensless single-exposure 3D imaging. Optica, 5(1), 1-9.

<a id="2">[2]</a> 
Monakhova, K., Yurtsever, J., Kuo, G., Antipa, N., Yanny, K., & Waller, L. (2019). Learned reconstructions for practical mask-based lensless imaging. Optics express, 27(20), 28075-28090.

<a id="3">[3]</a> 
Boyd, S., Parikh, N., & Chu, E. (2011). Distributed optimization and statistical learning via the alternating direction method of multipliers. Now Publishers Inc.