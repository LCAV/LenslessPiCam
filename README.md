# DiffuserCam

Setup on local computer and/or Raspberry Pi:
```bash
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

You can download example PSFs and raw data [here](https://drive.switch.ch/index.php/s/NdgHlcDeHVDH5ww).

You can download a subset for the [DiffuserCam Lensless Mirflickr Dataset](https://waller-lab.github.io/LenslessLearning/dataset.html)
that we've prepared [here](https://drive.switch.ch/index.php/s/vmAZzryGI8U8rcE)
with `scripts/prepare_mirflickr_subset.py`.

## Reconstruction

There is one script / algorithm available for reconstruction - ADMM.
```bash
python scripts/admm.py --psf_fp data/psf/diffcam_rgb.png \
--data_fp data/raw_data/thumbs_up_rgb.png --n_iter 5
```

A template - `scripts/reconstruction_template.py` - can be used to implement
other reconstruction approaches.

## Evaluating on a dataset

You can run ADMM on the [DiffuserCam Lensless Mirflickr Dataset](https://waller-lab.github.io/LenslessLearning/dataset.html)
with the following script.
```bash
python scripts/evaluate_mirflickr_admm.py --data <FP>
```
where `<FP>` is the path to the dataset.

You can also pass user [the subset](https://drive.switch.ch/index.php/s/vmAZzryGI8U8rcE)
we've prepared and apply it to a few files.
```bash
python scripts/evaluate_mirflickr_admm.py \
--data DiffuserCam_Mirflickr_200_3011302021_11h43_seed11 \
--n_files 10 --save
```
The `--save` flag will save a viewable image for each reconstruction.

You can also apply ADMM to a single image and visualize the iterative reconstruction.
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
the name of the file to saw the Bayer data, and the other arguments can be used
to adjust camera settings.

## Remote display

For collecting images displayed on a screen, we have prepared some software to
display images remotely on Raspberry Pi connected to a monitor.

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
python scripts/remote_display.py --fp data/original_images/rect.jpg \
--hostname <HOSTNAME> --pad 80 --vshift 10 --brightness 90
```
where `<HOSTNAME>` is the hostname or IP address of your Raspberry Pi and the 
other arguments can be used to adjust the positioning of the image and its
brightness.

## Formatting

```bash
pip install black
./format_code.sh
```