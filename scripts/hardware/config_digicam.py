import warnings
import hydra
from datetime import datetime
import numpy as np
from slm_controller import slm
from slm_controller.hardware import SLMParam, slm_devices

from lensless.hardware.slm import set_programmable_mask
from lensless.hardware.aperture import rect_aperture, circ_aperture
from lensless.hardware.utils import set_mask_sensor_distance


@hydra.main(version_base=None, config_path="../../configs", config_name="digicam")
def config_digicam(config):

    rpi_username = config.rpi.username
    rpi_hostname = config.rpi.hostname
    device = config.device

    shape = slm_devices[device][SLMParam.SLM_SHAPE]
    if not slm_devices[device][SLMParam.MONOCHROME]:
        shape = (3, *shape)
    pixel_pitch = slm_devices[device][SLMParam.PIXEL_PITCH]

    # set mask to sensor distance
    if config.z is not None:
        set_mask_sensor_distance(config.z, rpi_username, rpi_hostname)

    # create random pattern
    pattern = None
    if config.pattern.endswith(".npy"):
        pattern = np.load(config.pattern)
    elif config.pattern == "random":
        rng = np.random.RandomState(1)
        # pattern = rng.randint(low=0, high=np.iinfo(np.uint8).max, size=shape, dtype=np.uint8)
        pattern = rng.uniform(low=config.min_val, high=1, size=shape)
        pattern = (pattern * np.iinfo(np.uint8).max).astype(np.uint8)

        # save pattern
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pattern_fn = f"random_pattern_{timestamp}.npy"
        np.save(pattern_fn, pattern)
        print(f"Saved pattern to {pattern_fn}")

    elif config.pattern == "rect":
        rect_shape = config.rect_shape
        apert_dim = rect_shape[0] * pixel_pitch[0], rect_shape[1] * pixel_pitch[1]
        ap = rect_aperture(
            apert_dim=apert_dim,
            slm_shape=slm_devices[device][SLMParam.SLM_SHAPE],
            pixel_pitch=pixel_pitch,
            center=None,
        )
        pattern = ap.values
    elif config.pattern == "circ":
        ap = circ_aperture(
            radius=config.radius * pixel_pitch[0],
            slm_shape=slm_devices[device][SLMParam.SLM_SHAPE],
            pixel_pitch=pixel_pitch,
            center=None,
        )
        pattern = ap.values
    else:
        raise ValueError(f"Pattern {config.pattern} not supported")

    assert pattern is not None

    print("Pattern shape : ", pattern.shape)
    print("Pattern dtype : ", pattern.dtype)
    print("Pattern min   : ", pattern.min())
    print("Pattern max   : ", pattern.max())

    set_programmable_mask(pattern, device, rpi_username, rpi_hostname)

    # preview mask
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = slm.create(device)
        s._show_preview(pattern)


if __name__ == "__main__":
    config_digicam()
