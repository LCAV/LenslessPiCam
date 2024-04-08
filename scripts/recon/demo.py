import os
import hydra
from hydra.utils import to_absolute_path
import numpy as np
import time
from lensless.utils.plot import plot_image
from lensless.utils.io import save_image
from lensless.utils.image import resize, gamma_correction
import matplotlib.pyplot as plt
from lensless import FISTA, ADMM
from lensless.utils.io import load_image, load_psf
import omegaconf
from lensless.hardware.trainable_mask import AdafruitLCD


@hydra.main(version_base=None, config_path="../../configs", config_name="demo")
def demo(config):

    start_time = time.time()

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
    # -- load raw data
    localfile = f"{config.capture.raw_data_fn}.png"
    if save:
        localfile = os.path.join(save, localfile)
    print("\nLoading picture...")
    if config.capture.bayer:
        img = load_image(
            localfile,
            verbose=False,
            bayer=False if config.capture.rgb else True,
            blue_gain=config.camera.blue_gain,
            red_gain=config.camera.red_gain,
            nbits_out=config.capture.nbits_out,
        )
    else:
        img = load_image(localfile, verbose=True, bayer=False)

    data = np.array(img, dtype=config.recon.dtype)
    if len(data.shape) == 3:
        data = data[np.newaxis, :, :, :]
    elif len(data.shape) == 2:
        data = data[np.newaxis, :, :, np.newaxis]

    # -- PSF
    flipud = False
    if isinstance(config.camera.psf, omegaconf.dictconfig.DictConfig):
        import torch

        np.random.seed(config.camera.psf.seed % (2**32 - 1))
        mask_vals = np.random.uniform(0, 1, config.camera.psf.mask_shape)
        mask_vals_torch = torch.from_numpy(mask_vals.astype(np.float32))
        flipud = config.camera.psf.flipud
        mask = AdafruitLCD(
            initial_vals=mask_vals_torch,
            sensor=config.capture.sensor,
            slm=config.camera.psf.device,
            downsample=config.recon.downsample,
            flipud=flipud,
        )
        psf = mask.get_psf().detach().numpy()
        bg = np.zeros(psf.shape[-1])

    elif config.camera.background is not None:
        psf = load_psf(
            to_absolute_path(config.camera.psf),
            downsample=config.recon.downsample,
            return_float=True,
            return_bg=False,
            dtype=config.recon.dtype,
        )
        bg = np.load(to_absolute_path(config.camera.background))

    else:
        psf, bg = load_psf(
            to_absolute_path(config.camera.psf),
            downsample=config.recon.downsample,
            return_float=True,
            return_bg=True,
            dtype=config.recon.dtype,
        )
    psf = np.array(psf, dtype=config.recon.dtype)
    if config.plot:
        ax = plot_image(psf[0], gamma=config.recon.gamma)
        ax.set_title("PSF")
        if save:
            plt.savefig(os.path.join(save, "psf.png"))

    # -- prepare data
    if data.min() > 0:
        data -= bg
    data = np.clip(data, a_min=0, a_max=data.max())

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
        if flipud:
            data = torch.rot90(data, dims=(-3, -2), k=2)
    else:
        if flipud:
            data = np.rot90(data, k=2, axes=(-3, -2))

    print(f"Setup time : {time.time() - start_time} s")

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
    elif config.recon.algo == "unrolled":
        assert config.recon.use_torch, "Unrolled ADMM only available with torch"
        from lensless import UnrolledADMM

        algo_params = config.recon.unrolled_admm
        recon = UnrolledADMM(
            psf,
            **algo_params,
        )
        print("Loading checkpoint from : ", algo_params.checkpoint_fp)
        assert os.path.exists(algo_params.checkpoint_fp), "Checkpoint does not exist"
        recon.load_state_dict(
            torch.load(algo_params.checkpoint_fp, map_location=config.recon.torch_device)
        )
    else:
        raise ValueError(f"Unsupported algorithm: {config.recon.algo}")

    print("Applying : ", config.recon.algo)

    if config.recon.use_torch:
        with torch.no_grad():
            recon.set_data(data)
            res = recon.apply(
                gamma=config.recon.gamma,
                save=save,
                plot=config.plot,
                disp_iter=algo_params["disp_iter"],
            )
    else:
        recon.set_data(data)
        res = recon.apply(
            gamma=config.recon.gamma,
            save=save,
            plot=config.plot,
            disp_iter=algo_params["disp_iter"],
        )
    print(f"Processing time : {time.time() - start_time} s")

    if config.plot:
        final_image = res[0]
    else:
        final_image = res

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

        img_norm = img / img.max()
        if config.recon.gamma:
            img_norm = gamma_correction(img_norm, gamma=config.recon.gamma)

        output_fp = os.path.join(save, "reconstructed.png")
        save_image(img_norm, output_fp)

    if config.plot:
        plt.show()

    return save


if __name__ == "__main__":
    demo()
