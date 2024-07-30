import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import scipy.fft as sfft
import torch
from hydra.utils import to_absolute_path

from lensless import ADMM
from lensless.recon.rfft_convolve import RealFFTConvolve2D
from lensless.utils.image import resize
from lensless.utils.io import load_image, load_data


# import cupy as cp TODO
class Simulation:
    """
    imulate the lensless imaging process including resizing, convolution, noise addition, and reconstruction.

    This class encapsulates the steps necessary for simulating the lensless imaging process. It includes resizing the
    target image to match the PSF dimensions, convolving the resized image with the PSF, adding noise based on a
    specified signal-to-noise ratio (SNR), and reconstructing the image using the Alternating Direction Method of
    Multipliers (ADMM) technique.
    """


def square_laplace_primitive_noise(N, seed=None, mu=0, sigma=1, image_min=None, image_max=None):
    if seed is None:
        seed = np.random.randint(1000)
    print(f"Seed: {seed}")

    rng = np.random.default_rng(seed=seed)

    gaussian = rng.normal(mu, sigma, (N, N))
    gaussian_freq = sfft.fft2(gaussian)

    frequency_response = 2 * (np.cos(2 * np.pi * np.arange(N) / N)[None, :] +
                              np.cos(2 * np.pi * np.arange(N) / N)[:, None] - 2)
    frequency_response[0, 0] = 1.

    laplace_prim = sfft.ifft2(gaussian_freq / frequency_response ** 2).real

    # Normalize laplace_prim to [0, 1]
    laplace_prim_min, laplace_prim_max = laplace_prim.min(), laplace_prim.max()
    laplace_prim_normalized = (laplace_prim - laplace_prim_min) / (laplace_prim_max - laplace_prim_min)

    # Assuming image_min and image_max are provided, scale laplace_prim to the image's range
    if image_min is not None and image_max is not None:
        laplace_prim_scaled = laplace_prim_normalized * (image_max - image_min) + image_min
    else:
        # If image range is not provided, use the normalized version
        laplace_prim_scaled = laplace_prim_normalized

    return laplace_prim_scaled.astype(np.float32)


def normalize_image(image, target_min=0, target_max=1):
    """
    Normalize an image to a specified range.
    Parameters:
    - image (np.ndarray or torch.Tensor): The image to be normalized.
    - target_min (float): The minimum value of the target range.
    - target_max (float): The maximum value of the target range.
    Returns:
    - np.ndarray or torch.Tensor: The normalized image.
    """
    if isinstance(image, np.ndarray):
        # Find the minimum and maximum values in the image
        min_val = np.min(image)
        max_val = np.max(image)
    elif isinstance(image, torch.Tensor):
        min_val = torch.min(image)
        max_val = torch.max(image)
    else:
        raise TypeError("Unsupported image type. Expected np.ndarray or torch.Tensor.")

    # Normalize the image to the [0, 1] range
    normalized_image = (image - min_val) / (max_val - min_val)

    # Scale to target range if not [0, 1]
    if target_min != 0 or target_max != 1:
        normalized_image = normalized_image * (target_max - target_min) + target_min

    return normalized_image


def simulate_measurement(image, psf, snr_db=30, measured_background=None, artificial_background=None):
    """
    Simulate the lensless measurement process.

    Parameters:
    - image: The target image as a numpy array.
    - psf: The PSF as a numpy array.
    - alpha: Scaling factor for the background.
    - measured_background: The measured background noise (convoluted with the PSF)
    - simulated_background: The simulated background noise (to be convoluted with the PSF)

    Returns:
    - Simulated measurement as a numpy array.
    - SPF with added background noise
    """

    ### Compute alpha
    sig_var = torch.var(image.flatten())  # TODO precompute
    noise_var = torch.var(measured_background.flatten()) if measured_background is not None else np.var(
        artificial_background.flatten())

    alpha = torch.sqrt(sig_var / noise_var / (10 ** (snr_db / 10)))

    print(f"\nFor snr_db = {snr_db}, alpha = {alpha}")

    # Convolve background with PSF to simulate actual noise (as it's affected by the PSF) # TODO space varying convolution
    if artificial_background is not None:
        convolver = RealFFTConvolve2D(psf)
        measured_background = convolver.convolve(artificial_background)[0]

    # Add scaled background to the target image
    image_with_background = (image + alpha * measured_background)
    # Add dimension
    if len(image_with_background.shape) == 3:
        image_with_background = image_with_background[np.newaxis, :, :, :]

    return image_with_background, measured_background, alpha


def plot_different_snr(snr_values, config, psf, X, B):
    fig, axs = plt.subplots(len(snr_values), 2, figsize=(10, len(snr_values) * 5))
    recon_A = ADMM(normalize_image(psf), n_iter=config.admm.n_iter)  # TODO add preprocessor & other params
    recon_B = ADMM(normalize_image(psf), n_iter=config.admm.n_iter)

    # Debugging
    X_plt = None
    Y_plt = None
    psf_plt = None
    B_plt = None
    L_B_plt = None
    recovered_A_plt = None
    recovered_B_plt = None

    for i, snr in enumerate(snr_values):
        ### Simulate measurement
        if not config.files.is_noisy_measurement:
            # With simulated noise
            if B is None:
                B = square_laplace_primitive_noise(psf.shape[1], image_min=X.min(), image_max=X.max())
                # Replicate B across the third dimension to create a 3-channel image
                B_replicated = np.repeat(B[:, :, np.newaxis], 3, axis=2)

                # Reshape B_replicated to add an extra dimension at the start
                B = B_replicated[np.newaxis, :, :, :]
                B = resize(B, shape=psf.shape)

                Y, L_B, alpha = simulate_measurement(image=X, psf=psf, snr_db=snr, artificial_background=B)

            else:
                L_B = B.reshape(psf.shape)
                Y, _, alpha = simulate_measurement(image=X, psf=psf, snr_db=snr, measured_background=B)

        ### Measurement is already noisy
        else:
            Y = X
            L_B = B.reshape(psf.shape)
            alpha = 1

        # Direct recovery
        recon_A.set_data(normalize_image(Y))
        recovered_A = recon_A.apply()

        # Background noise subtraction recovery
        recon_B.set_data(normalize_image(Y) - alpha * normalize_image(L_B))  # TODO replace by X and normalize
        recovered_B = recon_B.apply()

        # Plotting
        axs[i, 0].imshow(normalize_image(recovered_A[0].cpu().numpy()), cmap='gray')  # TODO make it tensor
        #axs[i, 0].imshow(recovered_A[0].cpu().numpy(), cmap='gray')  # TODO make it tensor
        axs[i, 0].set_title(f'{snr}')
        axs[i, 0].axis('off')  # This removes the entire axis

        axs[i, 1].imshow(normalize_image(recovered_B[0].cpu().numpy()), cmap='gray')
        #axs[i, 1].imshow(recovered_B[0].cpu().numpy(), cmap='gray')
        axs[i, 1].set_title(f'{snr}')
        axs[i, 1].axis('off')  # This removes the entire axis
        axs[i, 1].get_xaxis().set_visible(False)
        axs[i, 1].get_yaxis().set_visible(False)

        # For debugging
        if i == len(snr_values) // 2:
            X_plt = X
            Y_plt = Y
            psf_plt = psf
            B_plt = B
            L_B_plt = L_B
            recovered_A_plt = recovered_A
            recovered_B_plt = recovered_B

    axs[0, 0].set_title('Direct Recovery')
    axs[0, 1].set_title('Subtraction Recovery')

    plt.tight_layout()
    plt.savefig("data/simulateBG/SNR_Comparaison.png")
    plt.show()
    plt.close()

    # Debugging
    plot_and_save(
        [X_plt, Y_plt, psf_plt, B_plt, L_B_plt, recovered_A_plt, recovered_B_plt],
        ['Original Image', 'Measured Image', 'PSF', 'Noise', 'L_B Image', 'Direct recovery',
         'Background noise subtraction recovery'],
        ['data/simulateBG/original.png', 'data/simulateBG/measured.png', 'data/simulateBG/psf.png',
         'data/simulateBG/noise.png', 'data/simulateBG/L_B.png', 'data/simulateBG/Direct_Recovery.png',
         'data/simulateBG/Subtraction_Recovery.png'])


def plot_and_save(items_to_plot, names, save_paths):
    """
    Plots items and saves them to specified paths.

    Parameters:
    - items_to_plot (list): List of items to plot. Each item is assumed to be an image array.
    - names (list): List of names for each plot.
    - save_paths (list): List of file paths where each plot should be saved.
    """
    for item, name, path in zip(items_to_plot, names, save_paths):
        plt.imshow(normalize_image(item[0].cpu().numpy()))  # TODO remove the normalize as I want to do it in the main
        plt.title(name)
        plt.savefig(path)
        plt.clf()


@hydra.main(version_base=None, config_path="../../configs", config_name="simulate_background.yaml")
def simulate(config):
    ### Load data
    psf_fp = to_absolute_path(config.files.psf)
    assert os.path.exists(psf_fp), f"PSF {psf_fp} does not exist."
    data_fp = to_absolute_path(config.files.original)
    assert os.path.exists(data_fp), f"File {data_fp} does not exist."
    bg_fp = None if config.files.background is None else to_absolute_path(config.files.background)

    downsample = config.files.downsample
    return_float = False
    bayer = False
    rg = 2.1
    bg = 1.15

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    psf, X, B = load_data(psf_fp=psf_fp, data_fp=data_fp, background_fp=bg_fp, return_bg=True, downsample=downsample,
                          return_float=return_float, bayer=bayer, red_gain=rg, blue_gain=bg)

    # TODO fix load data to remove need for this
    X = load_image(data_fp, downsample=downsample, verbose=True, return_float=return_float, bayer=bayer, red_gain=2.1,
                   blue_gain=1.15)  # TODO put the normalize parameter here
    X = X[np.newaxis, :, :, :]
    B = load_image(bg_fp, downsample=downsample, verbose=True, return_float=return_float, bayer=bayer, red_gain=2.1,
                   blue_gain=1.15)
    B = B[np.newaxis, :, :, :]

    # Torchification
    psf_torch = torch.from_numpy(psf).float().to(device=device)
    B_torch = torch.from_numpy(B).float().to(device=device)
    X_torch = torch.from_numpy(X).float().to(device=device)

    ### Plot reconstructions
    plot_different_snr([-10, 1, 5, 10, 40], config, psf_torch, X_torch, B_torch)


if __name__ == "__main__":
    simulate()
