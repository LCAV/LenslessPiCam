import numpy as np
import os
import time
import hydra
import torch
from hydra.utils import to_absolute_path
import matplotlib.pyplot as plt
from slm_controller import slm
from lensless.utils.io import save_image, get_dtype, load_psf
from lensless.utils.plot import plot_image
from lensless.hardware.sensor import VirtualSensor
from lensless.hardware.slm import get_programmable_mask, get_intensity_psf
from waveprop.devices import slm_dict
from waveprop.devices import SLMParam as SLMParam_wp
from huggingface_hub import hf_hub_download
from lensless.utils.image import gamma_correction


@hydra.main(version_base=None, config_path="../../configs", config_name="sim_digicam_psf")
def digicam_psf(config):

    output_folder = os.getcwd()

    # get file paths
    fp_psf = None
    if config.huggingface_repo is not None:
        # download from huggingface
        fp = hf_hub_download(
            repo_id=config.huggingface_repo,
            filename=config.huggingface_mask_pattern,
            repo_type="dataset",
        )
        if config.huggingface_psf is not None:
            fp_psf = hf_hub_download(
                repo_id=config.huggingface_repo,
                filename=config.huggingface_psf,
                repo_type="dataset",
            )
    else:
        fp = to_absolute_path(config.digicam.pattern)
        if config.digicam.psf is not None:
            fp_psf = to_absolute_path(config.digicam.psf)
    bn = os.path.basename(fp).split(".")[0]

    # digicam config
    ap_center = np.array(config.digicam.ap_center)
    ap_shape = np.array(config.digicam.ap_shape)
    rotate_angle = config.digicam.rotate
    slm_param = slm_dict[config.digicam.slm]
    sensor = VirtualSensor.from_name(
        config.digicam.sensor,
        downsample=config.digicam.downsample if config.digicam.downsample > 1 else None,
    )

    # simulation parameters
    scene2mask = config.sim.scene2mask
    mask2sensor = config.sim.mask2sensor

    torch_device = config.torch_device
    dtype = get_dtype(config.dtype, config.use_torch)

    """
    Load pattern
    """
    pattern = np.load(fp)
    # - make random pattern like original
    # pattern = np.random.rand(*pattern.shape) * 255
    # pattern = pattern.astype(np.uint8)

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

    # prepare color filter
    if SLMParam_wp.COLOR_FILTER in slm_param.keys():
        color_filter = slm_param[SLMParam_wp.COLOR_FILTER]
        if config.use_torch:
            color_filter = torch.from_numpy(color_filter.copy()).to(
                device=torch_device, dtype=dtype
            )
        else:
            color_filter = color_filter.astype(dtype)

    if config.digicam.slm == "adafruit":
        # flatten color channel along rows
        slm_vals = slm_vals.reshape((-1, slm_vals.shape[-1]), order="F")

    # save extracted mask values
    np.save(os.path.join(output_folder, "mask_vals.npy"), slm_vals)

    if config.use_torch:
        slm_vals = torch.from_numpy(slm_vals).to(device=torch_device, dtype=dtype)
    else:
        slm_vals = slm_vals.astype(dtype)

    # -- get mask
    mask = get_programmable_mask(
        vals=slm_vals,
        sensor=sensor,
        slm_param=slm_param,
        rotate=rotate_angle,
        flipud=config.sim.flipud,
        color_filter=color_filter,
        deadspace=config.sim.deadspace,
    )

    if config.digicam.vertical_shift is not None:
        if config.use_torch:
            mask = torch.roll(
                mask, config.digicam.vertical_shift // config.digicam.downsample, dims=1
            )
        else:
            mask = np.roll(mask, config.digicam.vertical_shift // config.digicam.downsample, axis=1)

    if config.digicam.horizontal_shift is not None:
        if config.use_torch:
            mask = torch.roll(
                mask, config.digicam.horizontal_shift // config.digicam.downsample, dims=2
            )
        else:
            mask = np.roll(
                mask, config.digicam.horizontal_shift // config.digicam.downsample, axis=2
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
        waveprop=config.sim.waveprop,
        scene2mask=scene2mask,
        mask2sensor=mask2sensor,
    )

    # -- plot PSF
    if config.use_torch:
        psf_in_np = psf_in.cpu().detach().numpy()
    else:
        psf_in_np = psf_in.copy()
    psf_in_np = np.transpose(psf_in_np, (1, 2, 0))

    # plot
    psf_meas = None
    if fp_psf is not None:
        if os.path.exists(fp_psf):
            psf_meas = load_psf(fp_psf, downsample=config.digicam.downsample)
        else:
            print("Could not load PSF image from: ", fp_psf)

    fp = os.path.join(output_folder, "sim_psf_plot.png")
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    # ax.imshow(psf_in_np / np.max(psf_in_np))
    plot_image(psf_in_np, gamma=config.digicam.gamma, normalize=True, ax=ax)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(fp)

    if psf_meas is not None:

        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)
        plot_image(psf_meas, gamma=config.digicam.gamma, normalize=True, ax=ax)
        # remove axis values
        ax.set_xticks([])
        ax.set_yticks([])
        plt.savefig(os.path.join(output_folder, "meas_psf_plot.png"))

        # plot overlayed
        fp = os.path.join(output_folder, "psf_overlay.png")
        psf_meas_norm = psf_meas[0] / np.max(psf_meas)
        psf_in_np_norm = psf_in_np / np.max(psf_in_np)

        plt.figure()
        plt.imshow(psf_in_np_norm, alpha=0.7)
        plt.imshow(psf_meas_norm, alpha=0.4)
        plt.savefig(fp)

    # save PSF as png
    fp = os.path.join(output_folder, f"{bn}_SIM_psf.png")
    # if config.digicam.gamma > 1:
    #     psf_in_np = psf_in_np / psf_in_np.max()
    #     psf_in_np = gamma_correction(psf_in_np, gamma=config.digicam.gamma)
    save_image(psf_in_np, fp)

    proc_time = time.time() - start_time
    print(f"\nProcessing time: {proc_time:.2f} seconds")

    print(f"\nFiles saved to : {output_folder}")


if __name__ == "__main__":
    digicam_psf()
