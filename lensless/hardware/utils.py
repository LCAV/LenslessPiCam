import csv
import os
import socket
import subprocess
import time

import paramiko
from paramiko.ssh_exception import AuthenticationException, BadHostKeyException, SSHException


def display(
    fp,
    rpi_username,
    rpi_hostname,
    screen_res,
    brightness=100,
    rot90=0,
    pad=0,
    vshift=0,
    hshift=0,
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

    os.system('scp %s "%s@%s:%s" ' % (fp, rpi_username, rpi_hostname, remote_tmp_file))

    # run script on Raspberry Pi to prepare image to display
    prep_command = f"{rpi_python} {script} --fp {remote_tmp_file} \
        --pad {pad} --vshift {vshift} --hshift {hshift} --screen_res {screen_res[0]} {screen_res[1]} \
        --brightness {brightness} --rot90 {rot90} --output_path {display_path} "
    # print(f"COMMAND : {prep_command}")
    subprocess.Popen(
        ["ssh", "%s@%s" % (rpi_username, rpi_hostname), prep_command],
        shell=False,
    )


def check_username_hostname(username, hostname, timeout=10):

    client = paramiko.client.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
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


def set_mask_sensor_distance(distance, rpi_username, rpi_hostname, motor=1):
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

    MAX_DISTANCE = 16  # mm
    timeout = 5

    client = check_username_hostname(rpi_username, rpi_hostname)
    assert motor in [0, 1]
    assert distance >= 0, "Distance must be non-negative"
    assert distance < MAX_DISTANCE, f"Distance must be less than {MAX_DISTANCE} mm"

    # assumes that `StepperDriver` is in home directory
    rpi_python = "python3"
    script = "~/StepperDriver/Python/serial_motors.py"

    # reset to zero
    print("Resetting to zero distance...")
    try:
        command = f"{rpi_python} {script} {motor} REV {MAX_DISTANCE * 1000}"
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
