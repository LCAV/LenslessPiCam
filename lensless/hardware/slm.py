# #############################################################################
# slm.py
# =================
# Authors :
# Eric BEZZAM [ebezzam@gmail.com]
# #############################################################################


import os
import numpy as np
from lensless.hardware.utils import check_username_hostname
from slm_controller.hardware import SLMParam, slm_devices


SUPPORTED_DEVICE = {
    "adafruit": "~/slm-controller/examples/adafruit_slm.py",
    "nokia": "~/slm-controller/examples/nokia_slm.py",
    "holoeye": "~/slm-controller/examples/holoeye_slm.py",
}


def set_programmable_mask(pattern, device, rpi_username, rpi_hostname):
    """
    Set LCD pattern on Raspberry Pi.

    This function assumes that `slm-controller <https://github.com/ebezzam/slm-controller>`_
    is installed on the Raspberry Pi.

    Parameters
    ----------
    pattern : :py:class:`~numpy.ndarray`
        Pattern to set on programmable mask.
    device : str
        Name of device to set pattern on. Supported devices: "adafruit", "nokia", "holoeye".
    rpi_username : str
        Username of Raspberry Pi.
    rpi_hostname : str
        Hostname of Raspberry Pi.

    """

    client = check_username_hostname(rpi_username, rpi_hostname)

    # get path to python executable on Raspberry Pi
    rpi_python = "~/slm-controller/slm_controller_env/bin/python"
    assert (
        device in SUPPORTED_DEVICE.keys()
    ), f"Device {device} not supported. Supported devices: {SUPPORTED_DEVICE.keys()}"
    script = SUPPORTED_DEVICE[device]

    # check that pattern is correct shape
    expected_shape = slm_devices[device][SLMParam.SLM_SHAPE]
    if not slm_devices[device][SLMParam.MONOCHROME]:
        expected_shape = (3, *expected_shape)
    assert (
        pattern.shape == expected_shape
    ), f"Pattern shape {pattern.shape} does not match expected shape {expected_shape}"

    # save pattern
    pattern_fn = "tmp_pattern.npy"
    local_path = os.path.join(os.getcwd(), pattern_fn)
    np.save(local_path, pattern)

    # copy pattern to Raspberry Pi
    remote_path = f"~/{pattern_fn}"
    print(f"PUTTING {local_path} to {remote_path}")

    os.system('scp %s "%s@%s:%s" ' % (local_path, rpi_username, rpi_hostname, remote_path))
    # # -- not sure why this doesn't work... permission denied
    # sftp = client.open_sftp()
    # sftp.put(local_path, remote_path, confirm=True)
    # sftp.close()

    # run script on Raspberry Pi to set mask pattern
    command = f"{rpi_python} {script} --file_path {remote_path}"
    print(f"COMMAND : {command}")
    _stdin, _stdout, _stderr = client.exec_command(command)
    print(_stdout.read().decode())
    client.close()

    os.remove(local_path)
