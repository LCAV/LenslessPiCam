# #############################################################################
# slm.py
# =================
# Authors :
# Eric BEZZAM [ebezzam@gmail.com]
# #############################################################################


import os
import numpy as np
from lensless.hardware.utils import check_username_hostname
from lensless.utils.io import get_ctypes
from slm_controller.hardware import SLMParam, slm_devices
from scipy.ndimage import rotate as rotate_func


try:
    import torch
    from torchvision import transforms
    from torchvision.transforms.functional import InterpolationMode

    torch_available = True
except ImportError:
    torch_available = False

try:
    from waveprop.spherical import spherical_prop
    from waveprop.color import ColorSystem
    from waveprop.rs import angular_spectrum
    from waveprop.slm import get_centers
    from waveprop.devices import SLMParam as SLMParam_wp

    waveprop_available = True
except ImportError:
    waveprop_available = False


SUPPORTED_DEVICE = {
    "adafruit": "~/slm-controller/examples/adafruit_slm.py",
    "nokia": "~/slm-controller/examples/nokia_slm.py",
    "holoeye": "~/slm-controller/examples/holoeye_slm.py",
}


def set_programmable_mask(pattern, device, rpi_username=None, rpi_hostname=None, verbose=False):
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

    if rpi_username is not None:
        assert (
            rpi_hostname is not None
        ), "rpi_hostname must be specified if rpi_username is specified"
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

    if rpi_username is not None:

        # copy pattern to Raspberry Pi
        remote_path = f"~/{pattern_fn}"
        if verbose:
            print(f"PUTTING {local_path} to {remote_path}")
        os.system(
            'scp %s "%s@%s:%s" >/dev/null 2>&1'
            % (local_path, rpi_username, rpi_hostname, remote_path)
        )
        # # -- not sure why this doesn't work... permission denied
        # sftp = client.open_sftp()
        # sftp.put(local_path, remote_path, confirm=True)
        # sftp.close()

        # run script on Raspberry Pi to set mask pattern
        command = f"{rpi_python} {script} --file_path {remote_path}"
        if verbose:
            print(f"COMMAND : {command}")
        _stdin, _stdout, _stderr = client.exec_command(command)
        if verbose:
            print(_stdout.read().decode())
        client.close()

    else:

        # run script on Raspberry Pi to set mask pattern
        command = f"{rpi_python} {script} --file_path {local_path} >/dev/null 2>&1"
        if verbose:
            print(f"COMMAND : {command}")
        os.system(command)

    os.remove(local_path)


def get_programmable_mask(
    vals, sensor, slm_param, rotate=None, flipud=False, nbits=8, color_filter=None, deadspace=True
):
    """
    Get mask as a numpy or torch array. Return same type.

    Parameters
    ----------
    vals : :py:class:`~numpy.ndarray` or :py:class:`~torch.Tensor`
        Values to set on programmable mask.
    sensor : :py:class:`~lensless.hardware.sensor.VirtualSensor`
        Sensor object.
    slm_param : dict
        SLM parameters.
    rotate : float, optional
        Rotation angle in degrees.
    flipud : bool, optional
        Flip mask vertically.
    nbits : int, optional
        Number of bits/levels to quantize mask to.
    deadspace: bool, optional
        Whether to include deadspace around mask. Default is True.

    """

    assert waveprop_available

    use_torch = False
    if torch_available:
        use_torch = isinstance(vals, torch.Tensor)
    dtype = vals.dtype

    # -- prepare SLM mask
    n_active_slm_pixels = vals.shape
    n_color_filter = np.prod(slm_param["color_filter"].shape[:2])

    # -- prepare color filter
    if color_filter is None and SLMParam_wp.COLOR_FILTER in slm_param.keys():
        color_filter = slm_param[SLMParam_wp.COLOR_FILTER]
        if isinstance(vals, torch.Tensor):
            color_filter = torch.tensor(color_filter).to(vals)

    if color_filter is not None:

        if isinstance(color_filter, np.ndarray):
            if flipud:
                color_filter = np.flipud(color_filter)
        elif isinstance(color_filter, torch.Tensor):
            if flipud:
                color_filter = torch.flip(color_filter, dims=(0,))
        else:
            raise ValueError("color_filter must be numpy array or torch tensor")

    # -- prepare mask
    if use_torch:
        mask = torch.zeros((n_color_filter,) + tuple(sensor.resolution)).to(vals)
        slm_vals_flat = vals.flatten()
    else:
        mask = np.zeros((n_color_filter,) + tuple(sensor.resolution), dtype=dtype)
        slm_vals_flat = vals.reshape(-1)
    pixel_pitch = slm_param[SLMParam_wp.PITCH]
    d1 = sensor.pitch
    if deadspace:

        centers = get_centers(n_active_slm_pixels, pixel_pitch=pixel_pitch)

        _height_pixel, _width_pixel = (slm_param[SLMParam_wp.CELL_SIZE] / d1).astype(int)

        for i, _center in enumerate(centers):

            _center_pixel = (_center / d1 + sensor.resolution / 2).astype(int)
            _center_top_left_pixel = (
                _center_pixel[0] - np.floor(_height_pixel / 2).astype(int),
                _center_pixel[1] + 1 - np.floor(_width_pixel / 2).astype(int),
            )
            color_filter_idx = i // n_active_slm_pixels[1] % n_color_filter

            mask_val = slm_vals_flat[i] * color_filter[color_filter_idx][0]
            if isinstance(mask_val, np.ndarray):
                mask_val = mask_val[:, np.newaxis, np.newaxis]
            elif isinstance(mask_val, torch.Tensor):
                mask_val = mask_val.unsqueeze(-1).unsqueeze(-1)
            mask[
                :,
                _center_top_left_pixel[0] : _center_top_left_pixel[0] + _height_pixel,
                _center_top_left_pixel[1] : _center_top_left_pixel[1] + _width_pixel,
            ] = mask_val

    else:

        # use color filter to turn mask into RGB
        if use_torch:
            active_mask_rgb = torch.zeros((n_color_filter,) + n_active_slm_pixels).to(vals)
        else:
            active_mask_rgb = np.zeros((n_color_filter,) + n_active_slm_pixels, dtype=dtype)

        # TODO avoid for loop
        for i in range(n_active_slm_pixels[0]):
            row_idx = i % color_filter.shape[0]
            for j in range(n_active_slm_pixels[1]):

                col_idx = j % color_filter.shape[1]
                color_filter_idx = color_filter[row_idx, col_idx]
                active_mask_rgb[
                    :, n_active_slm_pixels[0] - i - 1, n_active_slm_pixels[1] - j - 1
                ] = (vals[i, j] * color_filter_idx)

        # size of active pixels in pixels
        n_active_dim = np.around(slm_param[SLMParam_wp.PITCH] * n_active_slm_pixels / d1).astype(
            int
        )
        # n_active_dim = np.around(slm_param[SLMParam_wp.CELL_SIZE] * n_active_slm_pixels / d1).astype(int)

        # resize to n_active_dim
        if use_torch:
            mask_active = transforms.functional.resize(
                active_mask_rgb, n_active_dim, interpolation=InterpolationMode.NEAREST
            )
        else:
            # TODO check
            mask_active = np.zeros((n_color_filter,) + tuple(n_active_dim), dtype=dtype)
            for i in range(n_color_filter):
                mask_active[i] = np.resize(active_mask_rgb[i], n_active_dim)

        # pad to full mask
        top_left = (sensor.resolution - n_active_dim) // 2
        mask[
            :,
            top_left[0] : top_left[0] + n_active_dim[0],
            top_left[1] : top_left[1] + n_active_dim[1],
        ] = mask_active

    # # quantize mask
    # if use_torch:
    #     mask = mask / torch.max(mask)
    #     mask = torch.round(mask * (2**nbits - 1)) / (2**nbits - 1)
    # else:
    #     mask = mask / np.max(mask)
    #     mask = np.round(mask * (2**nbits - 1)) / (2**nbits - 1)

    # rotate
    if rotate is not None:
        if use_torch:
            mask = transforms.functional.rotate(mask, angle=rotate)
        else:
            mask = rotate_func(mask, axes=(2, 1), angle=rotate, reshape=False)

    return mask


def adafruit_sub2full(
    subpattern,
    center,
):
    sub_shape = subpattern.shape
    controllable_shape = (3, sub_shape[0] // 3, sub_shape[1])
    subpattern_rgb = subpattern.reshape(controllable_shape, order="F")
    subpattern_rgb *= 255

    # pad to full pattern
    pattern = np.zeros((3, 128, 160), dtype=np.uint8)
    top_left = [center[0] - controllable_shape[1] // 2, center[1] - controllable_shape[2] // 2]
    pattern[
        :,
        top_left[0] : top_left[0] + controllable_shape[1],
        top_left[1] : top_left[1] + controllable_shape[2],
    ] = subpattern_rgb.astype(np.uint8)
    return pattern


def full2subpattern(
    pattern,
    shape,
    center,
    slm=None,
):
    shape = np.array(shape)
    center = np.array(center)

    # extract region
    idx_1 = center[0] - shape[0] // 2
    idx_2 = center[1] - shape[1] // 2
    subpattern = pattern[:, idx_1 : idx_1 + shape[0], idx_2 : idx_2 + shape[1]]
    subpattern = subpattern / 255.0
    if slm == "adafruit":
        # flatten color channel along rows
        subpattern = subpattern.reshape((-1, subpattern.shape[-1]), order="F")
    return subpattern


def get_intensity_psf(
    mask,
    waveprop=False,
    sensor=None,
    scene2mask=None,
    mask2sensor=None,
    color_system=None,
):
    """
    Get intensity PSF from mask pattern. Return same type of data.

    Parameters
    ----------
    mask : :py:class:`~numpy.ndarray` or :py:class:`~torch.Tensor`
        Mask pattern.
    waveprop : bool, optional
        Whether to use wave propagation to compute PSF. Default is False,
        namely to return squared intensity of mask pattern as the PSF (i.e.,
        no wave propagation and just shadow of pattern).
    sensor : :py:class:`~lensless.hardware.sensor.VirtualSensor`
        Sensor object. Not used if ``waveprop=False``.
    scene2mask : float
        Distance from scene to mask. Not used if ``waveprop=False``.
    mask2sensor : float
        Distance from mask to sensor. Not used if ``waveprop=False``.
    color_system : :py:class:`~waveprop.color.ColorSystem`, optional
        Color system. Not used if ``waveprop=False``.

    """
    assert waveprop_available

    if color_system is None:
        color_system = ColorSystem.rgb()

    is_torch = False
    device = None
    if torch_available and isinstance(mask, torch.Tensor):
        is_torch = True
        device = mask.device

    dtype = mask.dtype
    ctype, _ = get_ctypes(dtype, is_torch)

    if is_torch:
        psfs = torch.zeros(mask.shape, dtype=ctype, device=device)
    else:
        psfs = np.zeros(mask.shape, dtype=ctype)

    if waveprop:

        assert sensor is not None, "sensor must be specified"
        assert scene2mask is not None, "scene2mask must be specified"
        assert mask2sensor is not None, "mask2sensor must be specified"

        assert (
            len(color_system.wv) == mask.shape[0]
        ), "Number of wavelengths must match number of color channels"

        # spherical wavefronts to mask
        spherical_wavefront = spherical_prop(
            in_shape=sensor.resolution,
            d1=sensor.pitch,
            wv=color_system.wv,
            dz=scene2mask,
            return_psf=True,
            is_torch=is_torch,
            device=device,
            dtype=dtype,
        )
        u_in = spherical_wavefront * mask

        # free space propagation to sensor
        for i, wv in enumerate(color_system.wv):
            psfs[i], _, _ = angular_spectrum(
                u_in=u_in[i],
                wv=wv,
                d1=sensor.pitch,
                dz=mask2sensor,
                dtype=dtype,
                device=device,
            )

    else:

        psfs = mask

    # -- intensity PSF
    if is_torch:
        psf_in = torch.square(torch.abs(psfs))
    else:
        psf_in = np.square(np.abs(psfs))

    return psf_in
