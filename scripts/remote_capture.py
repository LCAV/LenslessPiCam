"""

python scripts/remote_capture.py

"""


import os
import sys
import subprocess
import click
import cv2
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt

from diffcam.util import rgb2gray
from diffcam.plot import plot_image, pixel_histogram
from diffcam.io import load_image

REMOTEHOST = "pi@raspberrypi4.local"
REMOTE_PYTHON = "~/DiffuserCam/diffcam_env/bin/python"
REMOTE_CAPTURE_FP = "~/DiffuserCam/scripts/on_device_capture.py"
SENSOR_MODES = [
    "off",
    "auto",
    "sunlight",
    "cloudy",
    "shade",
    "tungsten",
    "fluorescent",
    "incandescent",
    "flash",
    "horizon",
]


@click.command()
@click.option(
    "--fn",
    default="test",
    type=str,
    help="File name for recorded image.",
)
@click.option(
    "--exp",
    default=0.5,
    type=float,
    help="Exposure time in seconds.",
)
@click.option(
    "--iso",
    default=100,
    type=int,
    help="ISO",
)
@click.option(
    "--source",
    type=click.Choice(["white", "red", "green", "blue"]),
    default="white",
    help="Light source.",
)
@click.option(
    "--config_pause",
    default=2,
    type=float,
    help="Pause time for loading / setting camera configuration.",
)
@click.option(
    "--sensor_mode",
    default="0",
    type=click.Choice(np.arange(len(SENSOR_MODES)).astype(str)),
    help="{'off': 0, 'auto': 1, 'sunlight': 2, 'cloudy': 3, 'shade': 4, 'tungsten': 5, "
    "'fluorescent': 6, 'incandescent': 7, 'flash': 8, 'horizon': 9}",
)
@click.option(
    "--nbits",
    default=12,
    type=int,
    help="Number of bits to set maximum value in histogram. Default is 12 for RPi HQ camera.",
)
@click.option(
    "--rgb",
    is_flag=True,
    help="Get RGB data from the Raspberry Pi or reconstruct it here (default). Takes longer to copy RGB data.",
)
@click.option(
    "--gamma",
    default=2.2,
    type=float,
    help="Gamma factor for plotting.",
)
def liveview(fn, exp, iso, config_pause, sensor_mode, nbits, source, rgb, gamma):
    assert exp <= 10
    assert exp > 0

    # take picture
    remote_fn = "remote_capture"
    print("\nTaking picture...")
    pic_command = (
        f"{REMOTE_PYTHON} {REMOTE_CAPTURE_FP} --fn {remote_fn} --exp {exp} --iso {iso} "
        f"--config_pause {config_pause} --sensor_mode {sensor_mode}"
    )
    if nbits > 8:
        pic_command += " --sixteen"
    if rgb:
        pic_command += " --rgb"
    print(f"COMMAND : {pic_command}")
    ssh = subprocess.Popen(
        ["ssh", "%s" % REMOTEHOST, pic_command],
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    result = ssh.stdout.readlines()
    error = ssh.stderr.readlines()
    if error != []:
        print("ERROR: %s" % error)
        return
    if result == []:
        error = ssh.stderr.readlines()
        print(sys.stderr, "ERROR: %s" % error)
        return
    else:
        result = [res.decode("UTF-8") for res in result]
        result = [res for res in result if len(res) > 3]
        result_dict = dict(map(lambda s: map(str.strip, s.split(":")), result))
        print(f"COMMAND OUTPUT : ")
        pprint(result_dict)

    # copy over file
    # more pythonic? https://stackoverflow.com/questions/250283/how-to-scp-in-python
    remotefile = f"~/{remote_fn}.png"
    localfile = f"{fn}.png"
    print(f"\nCopying over picture as {localfile}...")
    os.system('scp "%s:%s" %s' % (REMOTEHOST, remotefile, localfile))

    if rgb:

        img = load_image(localfile, verbose=True)

    else:

        # get white balance gains
        red_gain = float(result_dict["Red gain"])
        blue_gain = float(result_dict["Blue gain"])

        # load image
        print("\nLoading picture...")
        img = load_image(localfile, verbose=True, bayer=True, bg=blue_gain, rg=red_gain)

        # write RGB data
        cv2.imwrite(localfile, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # plot RGB
    ax = plot_image(img, gamma=gamma)
    ax.set_title("RGB")

    # plot red channel
    if source == "red":
        img_1chan = img[:, :, 0]
    elif source == "green":
        img_1chan = img[:, :, 1]
    elif source == "blue":
        img_1chan = img[:, :, 2]
    else:
        img_1chan = rgb2gray(img)
    ax = plot_image(img_1chan)
    if source == "white":
        ax.set_title("Gray scale")
    else:
        ax.set_title(f"{source} channel")

    # plot histogram, useful for checking clipping
    pixel_histogram(img)
    pixel_histogram(img_1chan)

    plt.show()


if __name__ == "__main__":
    liveview()
