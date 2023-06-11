import os
import hydra
from hydra.utils import to_absolute_path
import numpy as np
import time
from lensless.plot import plot_image
from lensless.io import load_image, load_psf, save_image
from lensless.util import resize, gamma_correction
import matplotlib.pyplot as plt
from lensless import FISTA, ADMM


@hydra.main(version_base=None, config_path="../../configs", config_name="demo")
def demo(config):

    if config.save:
        if config.output is not None:
            # make sure output directory exists
            os.makedirs(config.output, exist_ok=True)
            save = config.output
        else:
            save = os.getcwd()
    else:
        save = False

    # prepare data
    # -- PSF
    psf, bg = load_psf(
        to_absolute_path(config.camera.psf),
        downsample=config.recon.downsample,
        return_float=True,
        return_bg=True,
        dtype=config.recon.dtype,
    )
    psf = np.array(psf, dtype=config.recon.dtype)
    ax = plot_image(psf[0], gamma=config.recon.gamma)
    ax.set_title("PSF")
    if save:
        plt.savefig(os.path.join(save, "psf.png"))

    # -- data
    red_gain = config.camera.red_gain
    blue_gain = config.camera.blue_gain

    # load image
    localfile = f"{config.capture.raw_data_fn}.png"
    if save:
        localfile = os.path.join(save, localfile)
    print("\nLoading picture...")
    img = load_image(
        localfile,
        verbose=True,
        bayer=True,
        blue_gain=blue_gain,
        red_gain=red_gain,
        nbits_out=config.capture.nbits_out,
    )
    data = np.array(img, dtype=config.recon.dtype)
    data -= bg
    data = np.clip(data, a_min=0, a_max=data.max())

    if len(data.shape) == 3:
        data = data[np.newaxis, :, :, :]
    elif len(data.shape) == 2:
        data = data[np.newaxis, :, :, np.newaxis]

    if data.shape != psf.shape:
        # in DiffuserCam dataset, images are already reshaped
        data = resize(data, shape=psf.shape)
    data /= np.linalg.norm(data.ravel())
    data = np.array(data, dtype=config.recon.dtype)

    if config.recon.use_torch:
        import torch

        if config.recon.dtype == "float32":
            torch_dtype = torch.float32
        elif config.recon.dtype == "float64":
            torch_dtype = torch.float64
        else:
            raise ValueError("dtype must be float32 or float64")

        psf = torch.from_numpy(psf).type(torch_dtype).to(config.recon.torch_device)
        data = torch.from_numpy(data).type(torch_dtype).to(config.recon.torch_device)

    # -- apply algo
    start_time = time.time()

    if config.recon.algo == "fista":
        algo_params = config.recon.fista
        recon = FISTA(
            psf,
            **algo_params,
        )
    elif config.recon.algo == "admm":
        algo_params = config.recon.admm
        recon = ADMM(
            psf,
            **algo_params,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {config.recon.algo}")

    print("Applying : ", config.recon.algo)
    recon.set_data(data)
    final_image, ax = recon.apply(
        gamma=config.recon.gamma,
        save=save,
        plot=config.plot,
        disp_iter=algo_params["disp_iter"],
    )
    print(f"Processing time : {time.time() - start_time} s")
    # save final image ax
    if save:

        # take first depth
        final_image = final_image[0]
        if config.recon.use_torch:
            img = final_image.cpu().numpy()
        else:
            img = final_image

        if config.postproc.crop_hor is not None:
            img = img[
                :,
                int(config.postproc.crop_hor[0] * img.shape[1]) : int(
                    config.postproc.crop_hor[1] * img.shape[1]
                ),
            ]
        if config.postproc.crop_vert is not None:
            img = img[
                int(config.postproc.crop_vert[0] * img.shape[0]) : int(
                    config.postproc.crop_vert[1] * img.shape[0]
                ),
                :,
            ]

        if config.recon.gamma:
            img = gamma_correction(img, gamma=config.recon.gamma)

        output_fp = os.path.join(save, "reconstructed.png")
        save_image(img, output_fp)

    if config.plot:
        plt.show()

    return save


if __name__ == "__main__":
    demo()