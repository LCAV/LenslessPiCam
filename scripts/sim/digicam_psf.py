import numpy as np
import os
import time
import hydra
from hydra.utils import to_absolute_path
import matplotlib.pyplot as plt
from slm_controller import slm
from lensless.utils.io import save_image
from lensless.hardware.sensor import VirtualSensor
from lensless.hardware.slm import get_programmable_mask, get_intensity_psf
from waveprop.devices import slm_dict


@hydra.main(version_base=None, config_path="../../configs", config_name="sim_digicam_psf")
def digicam_psf(config):

    output_folder = os.getcwd()

    fp = to_absolute_path(config.digicam.pattern)
    bn = os.path.basename(fp).split(".")[0]

    # digicam config
    ap_center = np.array(config.digicam.ap_center)
    ap_shape = np.array(config.digicam.ap_shape)
    rotate_angle = config.digicam.rotate
    slm_param = slm_dict[config.digicam.slm]
    sensor = VirtualSensor.from_name(config.digicam.sensor)

    # simulation parameters
    scene2mask = config.sim.scene2mask
    mask2sensor = config.sim.mask2sensor

    torch_device = config.torch_device
    dtype = config.dtype

    """
    Load pattern
    """
    pattern = np.load(fp)

    # -- apply aperture
    aperture = np.zeros(pattern.shape, dtype=np.uint8)
    top_left = np.array(ap_center) - np.array(ap_shape) // 2
    bottom_right = top_left + np.array(ap_shape)
    aperture[:, top_left[0] : bottom_right[0], top_left[1] : bottom_right[1]] = 1
    pattern = pattern * aperture

    # -- extract aperture region
    idx_1 = ap_center[0] - ap_shape[0] // 2
    idx_2 = ap_center[1] - ap_shape[1] // 2

    pattern_sub = pattern[
        :,
        idx_1 : idx_1 + ap_shape[0],
        idx_2 : idx_2 + ap_shape[1],
    ]
    print("Controllable region shape: ", pattern_sub.shape)
    print("Total number of pixels: ", np.prod(pattern_sub.shape))

    # -- plot full
    s = slm.create(config.digicam.slm)
    s.set_preview(True)
    s.imshow(pattern)
    plt.savefig(os.path.join(output_folder, "pattern.png"))

    # -- plot sub pattern
    plt.imshow(pattern_sub.transpose(1, 2, 0))
    plt.savefig(os.path.join(output_folder, "pattern_sub.png"))

    """
    Simulate PSF
    """
    start_time = time.time()
    slm_vals = pattern_sub / 255.0
    mask = get_programmable_mask(
        vals=slm_vals,
        sensor=sensor,
        slm_param=slm_param,
        rotate=rotate_angle,
        flipud=config.sim.flipud,
        as_torch=config.use_torch,
        torch_device=torch_device,
        dtype=dtype,
        requires_grad=config.requires_grad,
    )

    # -- plot mask
    if config.use_torch:
        mask_np = mask.cpu().detach().numpy()
    else:
        mask_np = mask.copy()
    mask_np = np.transpose(mask_np, (1, 2, 0))
    plt.imshow(mask_np)
    plt.savefig(os.path.join(output_folder, "mask.png"))

    # -- propagate to sensor
    psf_in = get_intensity_psf(
        mask=mask,
        sensor=sensor,
        scene2mask=scene2mask,
        mask2sensor=mask2sensor,
        waveprop=config.sim.waveprop,
        torch_device=torch_device,
        dtype=dtype,
    )

    # -- plot PSF
    if config.use_torch:
        psf_in_np = psf_in.cpu().detach().numpy()
    else:
        psf_in_np = psf_in.copy()
    psf_in_np = np.transpose(psf_in_np, (1, 2, 0))

    # plot
    plt.imshow(psf_in_np)
    fp = os.path.join(output_folder, "psf_plot.png")
    plt.savefig(fp)

    # save PSF as png
    fp = os.path.join(output_folder, f"{bn}_SIM_psf.png")
    save_image(psf_in_np, fp)

    proc_time = time.time() - start_time
    print(f"\nProcessing time: {proc_time:.2f} seconds")

    print(f"\nFiles saved to : {output_folder}")


if __name__ == "__main__":
    digicam_psf()
