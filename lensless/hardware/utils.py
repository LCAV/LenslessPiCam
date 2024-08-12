import csv
import os
import socket
import subprocess
import time
import paramiko
from pprint import pprint
from paramiko.ssh_exception import AuthenticationException, BadHostKeyException, SSHException
from lensless.hardware.sensor import SensorOptions
import cv2
from lensless.utils.image import print_image_info
from lensless.utils.io import load_image
import logging

logging.getLogger("paramiko").setLevel(logging.WARNING)

if os.name == "nt":
    NULL_FILE = "nul"
else:
    NULL_FILE = "/dev/null 2>&1"


def capture(
    rpi_username,
    rpi_hostname,
    sensor,
    bayer,
    exp,
    fn="capture",
    iso=100,
    config_pause=2,
    sensor_mode="0",
    nbits_out=12,
    legacy=True,
    rgb=False,
    gray=False,
    nbits=12,
    down=None,
    awb_gains=None,
    rpi_python="~/LenslessPiCam/lensless_env/bin/python",
    capture_script="~/LenslessPiCam/scripts/measure/on_device_capture.py",
    verbose=False,
    output_path=None,
    **kwargs,
):
    """
    Capture image.

    Parameters
    ----------
    fn : str
        File name captured image.
    rpi_username : str
        Username of Raspberry Pi.
    rpi_hostname : str
        Hostname of Raspberry Pi.
    sensor : str
        Sensor name
    bayer : bool
        Whether to return bayer data (larger file size to transfer back).
    exp : int
        Exposure time in microseconds.
    iso : int
        ISO.
    config_pause : int
        Time to pause after configuring camera.
    sensor_mode : str
        Sensor mode.
    nbits_out : int
        Number of bits of output image.
    legacy : bool
        Whether to use legacy capture software of Raspberry Pi.
    rgb : bool
        Whether to capture RGB image.
    gray : bool
        Whether to capture grayscale image.
    nbits : int
        Number of bits of image.
    down : int
        Downsample factor.
    awb_gains : list
        AWB gains (red, blue).
    rpi_python : str
        Path to Python on Raspberry Pi.
    capture_script : str
        Path to capture script on Raspberry Pi.
    output_path : str
        Path to save image.
    verbose : bool
        Whether to print extra info.

    """

    # check_username_hostname(rpi_username, rpi_hostname)
    assert sensor in SensorOptions.values(), f"Sensor must be one of {SensorOptions.values()}"

    # form command
    remote_fn = "remote_capture"
    pic_command = (
        f"{rpi_python} {capture_script} sensor={sensor} bayer={bayer} fn={remote_fn} exp={exp} iso={iso} "
        f"config_pause={config_pause} sensor_mode={sensor_mode} nbits_out={nbits_out} "
        f"legacy={legacy} rgb={rgb} gray={gray} "
    )
    if nbits > 8:
        pic_command += " sixteen=True"
    if down:
        pic_command += f" down={down}"
    if awb_gains:
        pic_command += f" awb_gains=[{awb_gains[0]},{awb_gains[1]}]"

    if verbose:
        print(f"COMMAND : {pic_command}")

    # take picture
    ssh = subprocess.Popen(
        ["ssh", "%s@%s" % (rpi_username, rpi_hostname), pic_command],
        shell=False,
        # stdout=DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    result = ssh.stdout.readlines()
    error = ssh.stderr.readlines()

    if error != [] and legacy:  # new camera software seems to return error even if it works
        print("ERROR: %s" % error)
        return
    if result == []:
        error = ssh.stderr.readlines()
        print("ERROR: %s" % error)
        return
    else:
        result = [res.decode("UTF-8") for res in result]
        result = [res for res in result if len(res) > 3]
        result_dict = dict()
        for res in result:
            _key = res.split(":")[0].strip()
            _val = "".join(res.split(":")[1:]).strip()
            result_dict[_key] = _val
        # result_dict = dict(map(lambda s: map(str.strip, s.split(":")), result))
        if verbose:
            print("COMMAND OUTPUT : ")
            pprint(result_dict)

    # copy over file
    if (
        "RPi distribution" in result_dict.keys()
        and "bullseye" in result_dict["RPi distribution"]
        and not legacy
    ):

        if bayer:

            # copy over DNG file
            remotefile = f"~/{remote_fn}.dng"
            localfile = f"{fn}.dng"
            if output_path is not None:
                localfile = os.path.join(output_path, localfile)
            if verbose:
                print(f"\nCopying over picture as {localfile}...")
            os.system(
                'scp "%s@%s:%s" %s >%s'
                % (rpi_username, rpi_hostname, remotefile, localfile, NULL_FILE)
            )

            img = load_image(localfile, verbose=True, bayer=bayer, nbits_out=nbits_out)

            # print image properties
            print_image_info(img)

            # save as PNG
            png_out = f"{fn}.png"
            print(f"Saving RGB file as: {png_out}")
            cv2.imwrite(png_out, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        else:

            remotefile = f"~/{remote_fn}.png"
            localfile = f"{fn}.png"
            if output_path is not None:
                localfile = os.path.join(output_path, localfile)
            if verbose:
                print(f"\nCopying over picture as {localfile}...")
            os.system(
                'scp "%s@%s:%s" %s >%s'
                % (rpi_username, rpi_hostname, remotefile, localfile, NULL_FILE)
            )

            img = load_image(localfile, verbose=True)

    # legacy software running on RPi
    else:
        # copy over file
        # more pythonic? https://stackoverflow.com/questions/250283/how-to-scp-in-python
        remotefile = f"~/{remote_fn}.png"
        localfile = f"{fn}.png"
        if output_path is not None:
            localfile = os.path.join(output_path, localfile)
        if verbose:
            print(f"\nCopying over picture as {localfile}...")
        os.system(
            'scp "%s@%s:%s" %s >%s' % (rpi_username, rpi_hostname, remotefile, localfile, NULL_FILE)
        )

        if rgb or gray:
            img = load_image(localfile, verbose=verbose)

        else:

            if not bayer:
                # red_gain = config.camera.red_gain
                # blue_gain = config.camera.blue_gain
                red_gain = awb_gains[0]
                blue_gain = awb_gains[1]
            else:
                red_gain = None
                blue_gain = None

            # load image
            if verbose:
                print("\nLoading picture...")

            img = load_image(
                localfile,
                verbose=True,
                bayer=bayer,
                blue_gain=blue_gain,
                red_gain=red_gain,
                nbits_out=nbits_out,
            )

            # write RGB data
            if not bayer:
                cv2.imwrite(localfile, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    return localfile, img


def display(
    fp,
    rpi_username,
    rpi_hostname,
    screen_res,
    image_res=None,
    brightness=100,
    rot90=0,
    pad=0,
    vshift=0,
    hshift=0,
    verbose=False,
    **kwargs,
):
    """
    Display image on a screen.

    Assumes setup described here: https://lensless.readthedocs.io/en/latest/measurement.html#remote-display

    Parameters
    ----------
    fp : str
        File path to image.
    rpi_username : str
        Username of Raspberry Pi.
    rpi_hostname : str
        Hostname of Raspberry Pi.
    screen_res : tuple
        Screen resolution of Raspberry Pi.
    """

    # assumes that `LenslessPiCam` is in home directory and environment inside `LenslessPiCam`
    rpi_python = "~/LenslessPiCam/lensless_env/bin/python"
    script = "~/LenslessPiCam/scripts/measure/prep_display_image.py"
    remote_tmp_file = "~/tmp_display.png"
    display_path = "~/LenslessPiCam_display/test.png"

    os.system(f"scp {fp} {rpi_username}@{rpi_hostname}:{remote_tmp_file} > {NULL_FILE}")

    # run script on Raspberry Pi to prepare image to display
    prep_command = f"{rpi_python} {script} --fp {remote_tmp_file} \
        --pad {pad} --vshift {vshift} --hshift {hshift} --screen_res {screen_res[0]} {screen_res[1]} \
        --brightness {brightness} --rot90 {rot90} --output_path {display_path} "
    if image_res is not None:
        prep_command += f" --image_res {image_res[0]} {image_res[1]}"
    if verbose:
        print(f"COMMAND : {prep_command}")
    subprocess.Popen(
        ["ssh", "%s@%s" % (rpi_username, rpi_hostname), prep_command],
        shell=False,
        # stdout=DEVNULL
    )


def check_username_hostname(username, hostname, timeout=10):

    assert username is not None, "Raspberry Pi username must be specified"
    assert hostname is not None, "Raspberry Pi hostname must be specified"

    client = paramiko.client.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        # with suppress_stdout():
        client.connect(hostname, username=username, timeout=timeout)
    except (BadHostKeyException, AuthenticationException, SSHException, socket.error) as e:
        raise ValueError(f"Could not connect to {username}@{hostname}\n{e}")

    return client


def get_distro():
    """
    Get current OS distribution.
    Returns
    -------
    result : str
        Name and version of OS.
    """
    # https://majornetwork.net/2019/11/get-linux-distribution-name-and-version-with-python/
    RELEASE_DATA = {}
    with open("/etc/os-release") as f:
        reader = csv.reader(f, delimiter="=")
        for row in reader:
            if row:
                RELEASE_DATA[row[0]] = row[1]
    if RELEASE_DATA["ID"] in ["debian", "raspbian"]:
        with open("/etc/debian_version") as f:
            DEBIAN_VERSION = f.readline().strip()
        major_version = DEBIAN_VERSION.split(".")[0]
        version_split = RELEASE_DATA["VERSION"].split(" ", maxsplit=1)
        if version_split[0] == major_version:
            # Just major version shown, replace it with the full version
            RELEASE_DATA["VERSION"] = " ".join([DEBIAN_VERSION] + version_split[1:])
    return f"{RELEASE_DATA['NAME']} {RELEASE_DATA['VERSION']}"


def set_mask_sensor_distance(
    distance, rpi_username, rpi_hostname, motor=1, max_distance=16, timeout=5
):
    """
    Set the distance between the mask and sensor.

    This functions assumes that `StepperDriver <https://github.com/adrienhoffet/StepperDriver>`_ is installed.
    is downloaded on the Raspberry Pi.

    Parameters
    ----------
    distance : float
        Distance in mm. Positive values move the mask away from the sensor.
    rpi_username : str
        Username of Raspberry Pi.
    rpi_hostname : str
        Hostname of Raspberry Pi.
    """

    client = check_username_hostname(rpi_username, rpi_hostname)
    assert motor in [0, 1]
    assert distance >= 0, "Distance must be non-negative"
    assert distance <= max_distance, f"Distance must be less than {max_distance} mm"

    # assumes that `StepperDriver` is in home directory
    rpi_python = "python3"
    script = "~/StepperDriver/Python/serial_motors.py"

    # reset to zero
    print("Resetting to zero distance...")
    try:
        command = f"{rpi_python} {script} {motor} REV {max_distance * 1000}"
        _stdin, _stdout, _stderr = client.exec_command(command, timeout=timeout)
    except socket.timeout:  # socket.timeout
        pass

    client.close()
    time.sleep(5)  # TODO reduce this time
    client = check_username_hostname(rpi_username, rpi_hostname)

    # set to desired distance
    if distance != 0:
        print(f"Setting distance to {distance} mm...")
        distance_um = distance * 1000
        if distance_um >= 0:
            command = f"{rpi_python} {script} {motor} FWD {distance_um}"
        else:
            command = f"{rpi_python} {script} {motor} REV {-1 * distance_um}"
        print(f"COMMAND : {command}")
        try:
            _stdin, _stdout, _stderr = client.exec_command(command, timeout=timeout)
            print(_stdout.read().decode())
        except socket.timeout:  # socket.timeout
            client.close()

    client.close()
