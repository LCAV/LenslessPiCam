import hydra
from hydra.utils import to_absolute_path
from lensless.utils.io import load_image, load_psf, save_image
from lensless.utils.image import rgb2gray, resize, rgb_to_bayer4d, bayer4d_to_rgb
import numpy as np
from lensless import ADMM
from lensless.eval.metric import mse, psnr, ssim, lpips, LPIPS_MIN_DIM
from waveprop.simulation import FarFieldSimulator
import glob
import os
from tqdm import tqdm
from numpy.linalg import multi_dot
from scipy.linalg import circulant
from waveprop.noise import add_shot_noise
from lensless.hardware.mask import CodedAperture, PhaseContour, FresnelZoneAperture
from lensless.recon.tikhonov import CodedApertureReconstruction
import matplotlib.pyplot as plt


def conv_matrices(img_shape, mask):
    P = circulant(np.resize(mask.col, mask.mask.shape[0]))[:, : img_shape[0]]
    Q = circulant(np.resize(mask.row, mask.mask.shape[1]))[:, : img_shape[1]]
    return P, Q


def fc_simulation(img, mask, P=None, Q=None, format="RGB", SNR=40):
    """
    Simulation function
    """
    format = format.lower()
    assert format in [
        "grayscale",
        "rgb",
        "bayer_rggb",
        "bayer_bggr",
        "bayer_grbg",
        "bayer_gbrg",
    ], "color_profile must be in ['grayscale', 'rgb', 'bayer_rggb', 'bayer_bggr', 'bayer_grbg', 'bayer_gbrg']"
    # print(f"######### SQUEEZE SHAPE:", img.squeeze().shape, "#########")
    if len(img.squeeze().shape) == 2:
        # print('here 1')
        n_channels = 1
        img_ = img.copy()
    elif format == "grayscale":
        # print('here 2')
        n_channels = 1
        img_ = rgb2gray(img)
    elif format == "rgb":
        n_channels = 3
        img_ = img.copy()
    else:
        n_channels = 4
        img_ = rgb_to_bayer4d(img, pattern=format[-4:])
    if P is None:
        P = circulant(np.resize(mask.col, mask.mask.shape[0]))[:, : img.shape[0]]
    if Q is None:
        Q = circulant(np.resize(mask.row, mask.mask.shape[1]))[:, : img.shape[1]]
    Y = np.dstack([multi_dot([P, img_[:, :, c], Q.T]) for c in range(n_channels)])
    # Y = (Y - Y.min()) / (Y.max() - Y.min())
    # Y = add_shot_noise(Y, snr_db=SNR)
    Y = (Y - Y.min()) / (Y.max() - Y.min())

    return Y


@hydra.main(version_base=None, config_path="../../configs", config_name="simulate_dataset_custom")
def simulate(config):

    # set seed
    np.random.seed(config.seed)

    dataset = to_absolute_path(config.files.dataset)
    if not os.path.isdir(dataset):
        print(f"No dataset found at {dataset}")
        try:
            from torchvision.datasets.utils import download_and_extract_archive, download_url
        except ImportError:
            exit()
        msg = "Do you want to download the sample CelebA dataset (764KB)?"

        # default to yes if no input is given
        valid = input("%s (Y/n) " % msg).lower() != "n"
        if valid:
            url = "https://drive.switch.ch/index.php/s/Q5OdDQMwhucIlt8/download"
            filename = "celeb_mini.zip"
            download_and_extract_archive(
                url, os.path.dirname(dataset), filename=filename, remove_finished=True
            )

    # psf_fp = to_absolute_path(config.files.psf)
    # assert os.path.exists(psf_fp), f"PSF {psf_fp} does not exist."

    if config.save.bool:
        save_dir = to_absolute_path(config.save.dir + config.simulation.mask_type)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            os.makedirs(os.path.join(save_dir, "sensor_plane"))
            os.makedirs(os.path.join(save_dir, "object_plane"))
            os.makedirs(os.path.join(save_dir, "reconstruction"))

    ## load psf as numpy array
    # print("\nPSF:")
    # psf = load_psf(psf_fp, verbose=True, downsample=config.simulation.downsample)

    # remove depth dimension, as 3D not supported by FarFieldSimulator
    # psf_sim = psf[0]
    # feature_size = np.random.uniform(config.simulation.object_height[0], config.simulation.object_height[1])
    if config.simulation.mask_type.upper() in ["MURA", "MLS"]:
        mask = CodedAperture.from_sensor(
            sensor_name="rpi_hq",
            downsample=8,
            method=config.simulation.mask_type,
            n_bits=config.simulation.n_bits,
            distance_sensor=config.simulation.mask2sensor,
        )
        psf_sim = mask.psf / np.linalg.norm(mask.psf.ravel())
    elif config.simulation.mask_type.upper() == "FZA":
        mask = FresnelZoneAperture.from_sensor(
            sensor_name="rpi_hq",
            downsample=8,
            radius=config.simulation.radius,
            distance_sensor=config.simulation.mask2sensor,
        )
        psf_sim = mask.psf / np.linalg.norm(mask.psf.ravel())
    elif config.simulation.mask_type.lower() == "phase":
        mask = PhaseContour.from_sensor(
            sensor_name="rpi_hq",
            downsample=8,
            noise_period=config.simulation.noise_period,
            refractive_index=config.simulation.refractive_index,
            n_iter=config.simulation.phase_mask_iter,
            distance_sensor=config.simulation.mask2sensor,
        )
        psf_sim = mask.psf / np.linalg.norm(mask.psf.ravel())
    else:
        psf_fp = to_absolute_path(config.files.psf)
        assert os.path.exists(psf_fp), f"PSF {psf_fp} does not exist."
        psf = load_psf(psf_fp, verbose=True, downsample=config.simulation.downsample)
        psf_sim = psf[0]

    if config.simulation.grayscale and len(psf_sim.shape) == 3:
        psf_sim = rgb2gray(psf_sim)
    print("\n", psf_sim.min(), psf_sim.max(), "\n")
    plt.figure(figsize=(10, 10)), plt.imshow(psf_sim * 255, cmap="gray"), plt.colorbar()
    plt.title("PSF for the simulation")  # , plt.xticks([]), plt.yticks([])
    plt.show()
    if config.simulation.downsample > 1:
        print(f"Downsampled to {psf_sim.shape}.")

    # prepare simulator object
    simulator = FarFieldSimulator(psf=psf_sim, **config.simulation)

    # loop over files in dataset
    print("\nSimulating dataset...")
    files = glob.glob(os.path.join(dataset, f"*.{config.files.image_ext}"))
    if config.files.n_files is not None:
        files = files[: config.files.n_files]
    for fp in tqdm(files):

        # load image as numpy array
        image = load_image(fp)
        if config.simulation.grayscale and len(image.shape) == 3:
            image = rgb2gray(image)

        # simulate image
        image_plane, object_plane = simulator.propagate(image, return_object_plane=True)
        if config.save.bool:

            bn = os.path.basename(fp).split(".")[0] + ".png"

            # can serve as ground truth
            object_plane_fp = os.path.join(save_dir, "object_plane", bn)
            save_image(object_plane, object_plane_fp)  # use max range of 255

            # lensless image
            lensless_fp = os.path.join(save_dir, "sensor_plane", bn)
            save_image(image_plane, lensless_fp, max_val=config.simulation.max_val)

    # reconstruction
    if config.admm.bool:

        print("\nReconstructing lensless measurements...")

        output_dim = image_plane.shape
        if config.simulation.output_dim is not None:
            # best would be to incorporate downsampling in the reconstruction
            # for now downsample the PSF
            print("-- Resizing PSF to", config.simulation.output_dim, "for reconstruction.")
            psf = resize(psf, shape=config.simulation.output_dim)

        # -- initialize reconstruction object
        # recon = ADMM(psf, **config.admm)
        recon = ADMM(psf_sim[np.newaxis, :, :, :], **config.admm)

        # -- metrics
        mse_vals = []
        psnr_vals = []
        ssim_vals = []
        if not config.simulation.grayscale and np.min(output_dim[:2]) >= LPIPS_MIN_DIM:
            lpips_vals = []
        else:
            lpips_vals = None

        # -- loop over files in dataset
        files = glob.glob(os.path.join(save_dir, "sensor_plane", "*.png"))
        if config.files.n_files is not None:
            files = files[: config.files.n_files]

        for fp in tqdm(files):

            lensless = load_image(fp, as_4d=True)
            lensless = lensless / np.max(lensless)
            recon.set_data(lensless)
            res, _ = recon.apply(n_iter=config.admm.n_iter, disp_iter=config.admm.disp_iter)
            recovered = res[0]  # first depth

            if config.save.bool:
                bn = os.path.basename(fp).split(".")[0] + ".png"
                lensless_fp = os.path.join(save_dir, "reconstruction", bn)

                # need float image for gamma correction and plotting
                # gamma = 2.2
                # img_norm = img / img.max()
                # if gamma and gamma > 1:
                #    img_norm = gamma_correction(img_norm, gamma=gamma)
                save_image(recovered, lensless_fp, max_val=config.simulation.max_val)

            # compute metrics
            object_plane_fp = os.path.join(save_dir, "object_plane", os.path.basename(fp))
            object_plane = load_image(object_plane_fp)

            if config.simulation.output_dim is not None:
                # best would be to incorporate downsampling in the reconstruction
                # for now downsample the PSF
                # print("-- Resizing object plane to", config.simulation.output_dim, "for metric.")
                object_plane = resize(object_plane, shape=config.simulation.output_dim)

            mse_vals.append(mse(object_plane, recovered))
            psnr_vals.append(psnr(object_plane, recovered))
            if config.simulation.grayscale:
                ssim_vals.append(ssim(object_plane, recovered, channel_axis=None))
            else:
                ssim_vals.append(ssim(object_plane, recovered))
            if lpips_vals is not None:
                lpips_vals.append(lpips(object_plane, recovered))

    print("\nMSE (avg)", np.mean(mse_vals))
    print("PSNR (avg)", np.mean(psnr_vals))
    print("SSIM (avg)", np.mean(ssim_vals))
    if lpips_vals is not None:
        print("LPIPS (avg)", np.mean(lpips_vals))


if __name__ == "__main__":
    simulate()
