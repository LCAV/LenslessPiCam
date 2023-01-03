"""

Simulation steps

1) Resize image to PSF dimensions while keeping aspect ratio and setting object height to desired value.
2) Convolve each channel of the image with the PSF.
3) Add noise according to desired SNR.

Reconstruct with ADMM and evaluate.

"""

import hydra
from hydra.utils import to_absolute_path
from lensless.io import load_image, load_psf, save_image
from lensless.util import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from lensless import ADMM
from lensless.plot import plot_image
from lensless.metric import mse, psnr, ssim, lpips
from waveprop.simulation import ConvolutionWithPSF


@hydra.main(version_base=None, config_path="../../configs", config_name="simulate_single_file")
def simulate(config):

    fp = to_absolute_path(config.files.original)
    psf_fp = to_absolute_path(config.files.psf)

    # simulation parameters
    object_height = config.simulation.object_height
    scene2mask = config.simulation.scene2mask
    mask2sensor = config.simulation.mask2sensor
    sensor = config.simulation.sensor
    snr_db = config.simulation.snr_db
    downsample = config.simulation.downsample
    grayscale = config.simulation.grayscale
    max_val = config.simulation.max_val

    # load image as numpy array
    print("\nImage:")
    image = load_image(fp, verbose=True)
    if grayscale and len(image.shape) == 3:
        image = rgb2gray(image)

    # load psf as numpy array
    print("\nPSF:")
    psf = load_psf(psf_fp, verbose=True, downsample=downsample)
    if grayscale and len(psf.shape) == 3:
        psf = rgb2gray(psf)
    if downsample > 1:
        print(f"Downsampled to {psf.shape}.")

    """ Simulation"""
    simulator = ConvolutionWithPSF(
        psf=psf,
        object_height=object_height,
        scene2mask=scene2mask,
        mask2sensor=mask2sensor,
        sensor=sensor,
        snr_db=snr_db,
        max_val=max_val,
    )
    image_plane, object_plane = simulator.propagate(image, return_object_plane=True)
    if config.save:
        save_image(object_plane, "object_plane.png")
        save_image(image_plane, "image_plane.png", max_val=max_val)
    ax = plot_image(image_plane)
    ax.set_title("Simulated lensless image")

    """ Reconstruction """
    recon = ADMM(psf, **config.admm)
    recon.set_data(image_plane)
    res = recon.apply(n_iter=config.admm.n_iter, disp_iter=config.admm.disp_iter)
    recovered = res[0]

    """ Evaluation """
    object_plane = object_plane.astype(np.float32)
    recovered = recovered.astype(np.float32)

    _, ax = plt.subplots(ncols=2, nrows=1, figsize=(10, 5))
    plot_image(recovered, ax=ax[0])
    ax[0].set_title("Reconstruction")
    plot_image(object_plane, ax=ax[1])
    ax[1].set_title("Original")

    print("\nEvaluation:")
    print("MSE", mse(object_plane, recovered))
    print("PSNR", psnr(object_plane, recovered))
    if grayscale:
        print("SSIM", ssim(object_plane, recovered, channel_axis=None))
    else:
        print("SSIM", ssim(object_plane, recovered))
        print("LPIPS", lpips(object_plane, recovered))

    plt.show()


if __name__ == "__main__":
    simulate()
