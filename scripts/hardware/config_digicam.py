import warnings
import hydra
from datetime import datetime
import numpy as np
from slm_controller import slm
from slm_controller.hardware import SLMParam, slm_devices
import matplotlib.pyplot as plt

from lensless.hardware.slm import set_programmable_mask
from lensless.hardware.aperture import rect_aperture, circ_aperture
from lensless.hardware.utils import set_mask_sensor_distance


@hydra.main(version_base=None, config_path="../../configs", config_name="digicam_config")
def config_digicam(config):

    rpi_username = config.rpi.username
    rpi_hostname = config.rpi.hostname
    device = config.device

    shape = slm_devices[device][SLMParam.SLM_SHAPE]
    if not slm_devices[device][SLMParam.MONOCHROME]:
        shape = (3, *shape)
    pixel_pitch = slm_devices[device][SLMParam.PIXEL_PITCH]

    # set mask to sensor distance
    if config.z is not None and not config.virtual:
        set_mask_sensor_distance(config.z, rpi_username, rpi_hostname)

    center = np.array(config.center) * pixel_pitch

    # create random pattern
    pattern = None
    if config.pattern.endswith(".npy"):
        pattern = np.load(config.pattern)
    elif config.pattern == "random":
        rng = np.random.RandomState(1)
        # pattern = rng.randint(low=0, high=np.iinfo(np.uint8).max, size=shape, dtype=np.uint8)
        pattern = rng.uniform(low=config.min_val, high=1, size=shape)
        pattern = (pattern * np.iinfo(np.uint8).max).astype(np.uint8)

    elif config.pattern == "rect":
        rect_shape = config.rect_shape
        apert_dim = rect_shape[0] * pixel_pitch[0], rect_shape[1] * pixel_pitch[1]
        ap = rect_aperture(
            apert_dim=apert_dim,
            slm_shape=slm_devices[device][SLMParam.SLM_SHAPE],
            pixel_pitch=pixel_pitch,
            center=center,
        )
        pattern = ap.values
    elif config.pattern == "circ":
        ap = circ_aperture(
            radius=config.radius * pixel_pitch[0],
            slm_shape=slm_devices[device][SLMParam.SLM_SHAPE],
            pixel_pitch=pixel_pitch,
            center=center,
        )
        pattern = ap.values
    else:
        raise ValueError(f"Pattern {config.pattern} not supported")

    # apply aperture
    if config.aperture is not None:

        # aperture = np.zeros(shape, dtype=np.uint8)
        # top_left = np.array(config.aperture.center) - np.array(config.aperture.shape) // 2
        # bottom_right = top_left + np.array(config.aperture.shape)
        # aperture[:, top_left[0] : bottom_right[0], top_left[1] : bottom_right[1]] = 1

        apert_dim = np.array(config.aperture.shape) * np.array(pixel_pitch)
        ap = rect_aperture(
            apert_dim=apert_dim,
            slm_shape=slm_devices[device][SLMParam.SLM_SHAPE],
            pixel_pitch=pixel_pitch,
            center=np.array(config.aperture.center) * pixel_pitch,
        )
        aperture = ap.values
        aperture[aperture > 0] = 1
        pattern = pattern * aperture

    # save pattern
    if not config.pattern.endswith(".npy") and config.save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pattern_fn = f"{device}_{config.pattern}_pattern_{timestamp}.npy"
        np.save(pattern_fn, pattern)
        print(f"Saved pattern to {pattern_fn}")

    print("Pattern shape : ", pattern.shape)
    print("Pattern dtype : ", pattern.dtype)
    print("Pattern min   : ", pattern.min())
    print("Pattern max   : ", pattern.max())

    assert pattern is not None

    n_nonzero = np.count_nonzero(pattern)
    print(f"Nonzero pixels: {n_nonzero}")

    if not config.virtual:
        set_programmable_mask(pattern, device, rpi_username, rpi_hostname)

    # preview mask
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = slm.create(device)
        if config.preview:
            s._show_preview(pattern)
        plt.savefig("preview.png")


if __name__ == "__main__":
    config_digicam()
