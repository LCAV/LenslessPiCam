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

## Diffuser MirFlickr dataset

Download subset that we've prepared [here](https://drive.switch.ch/index.php/s/vmAZzryGI8U8rcE).

## Formatting

```bash
pip install black
./format_code.sh
```