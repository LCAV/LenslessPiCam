"""
Apply ADMM reconstruction.
```
python scripts/recon/admm.py
```

"""

import hydra
from hydra.utils import to_absolute_path
import os
import time
import pathlib as plib
import matplotlib.pyplot as plt
import numpy as np
from lensless.utils.io import load_data, load_image
from lensless import ADMM
from lensless.utils.plot import plot_image


@hydra.main(version_base=None, config_path="../../configs/recon", config_name="defaults")
def admm(config):
    if config.torch:
        try:
            import torch
        except ImportError:
            raise ImportError("Pytorch not found. Please install pytorch to use torch mode.")

    psf, data = load_data(
        psf_fp=to_absolute_path(config.input.psf),
        data_fp=to_absolute_path(config.input.data),
        background_fp=(
            to_absolute_path(config.input.background)
            if config.input.background is not None
            else None
        ),
        dtype=config.input.dtype,
        downsample=config["preprocess"]["downsample"],
        bayer=config["preprocess"]["bayer"],
        blue_gain=config["preprocess"]["blue_gain"],
        red_gain=config["preprocess"]["red_gain"],
        plot=config["display"]["plot"],
        flip=config["preprocess"]["flip"],
        gamma=config["display"]["gamma"],
        gray=config["preprocess"]["gray"],
        single_psf=config["preprocess"]["single_psf"],
        shape=config["preprocess"]["shape"],
        use_torch=config.torch,
        torch_device=config.torch_device,
        bg_pix=config.preprocess.bg_pix,
        normalize=config.preprocess.normalize,
    )

    disp = config["display"]["disp"]
    if disp < 0:
        disp = None

    save = config["save"]
    if save:
        save = os.getcwd()

    if save:
        if config.torch:
            org_data = data.cpu().numpy()
        else:
            org_data = data
        ax = plot_image(org_data, gamma=config["display"]["gamma"])
        ax.set_title("Raw data")
        plt.savefig(plib.Path(save) / "lensless.png")

        # close axes
        fig = plt.gcf()
        plt.close(fig)

    # load model
    start_time = time.time()
    if not config.admm.unrolled:
        recon = ADMM(psf, **config.admm)
    else:
        assert config.torch, "Unrolled ADMM only works with torch"
        from lensless.recon.unrolled_admm import UnrolledADMM
        import lensless.recon.utils

        pre_process, _ = lensless.recon.utils.create_process_network(
            network=config.admm.pre_process_model.network,
            depth=config.admm.pre_process_model.depth,
            device=config.torch_device,
        )
        post_process, _ = lensless.recon.utils.create_process_network(
            network=config.admm.post_process_model.network,
            depth=config.admm.post_process_model.depth,
            device=config.torch_device,
        )

        recon = UnrolledADMM(psf, pre_process=pre_process, post_process=post_process, **config.admm)
        path = to_absolute_path(config.admm.checkpoint_fp)
        print("Loading checkpoint from : ", path)
        assert os.path.exists(path), "Checkpoint does not exist"
        recon.load_state_dict(torch.load(path, map_location=config.torch_device))

    recon.set_data(data)
    print(f"Setup time : {time.time() - start_time} s")

    start_time = time.time()
    if config.torch:
        with torch.no_grad():
            if config.admm.unrolled:
                res = recon.apply(
                    disp_iter=disp,
                    save=save,
                    gamma=config["display"]["gamma"],
                    plot=config["display"]["plot"],
                    output_intermediate=True,
                )
            else:
                res = recon.apply(
                    disp_iter=disp,
                    save=save,
                    gamma=config["display"]["gamma"],
                    plot=config["display"]["plot"],
                )
    else:
        res = recon.apply(
            disp_iter=disp,
            save=save,
            gamma=config["display"]["gamma"],
            plot=config["display"]["plot"],
        )
    print(f"Processing time : {time.time() - start_time} s")

    if config.torch:
        img = res[0].cpu().numpy()
    else:
        img = res[0]

    if config["display"]["plot"]:
        plt.show()
    if save:

        if config.admm.unrolled:
            # Save intermediate results
            if res[1] is not None:
                pre_processed_image = res[1].cpu().numpy()
                ax = plot_image(pre_processed_image, gamma=config["display"]["gamma"])
                ax.set_title("Image after preprocessing")
                plt.savefig(plib.Path(save) / "pre_processed.png")

            if res[2] is not None:
                pre_post_process_image = res[2].cpu().numpy()
                ax = plot_image(pre_post_process_image, gamma=config["display"]["gamma"])
                ax.set_title("Image prior to post-processing")
                plt.savefig(plib.Path(save) / "pre_post_process.png")

        np.save(plib.Path(save) / "final_reconstruction.npy", img)

        if config.input.original is not None:
            original = load_image(
                to_absolute_path(config.input.original),
                flip=config["preprocess"]["flip"],
                red_gain=config["preprocess"]["red_gain"],
                blue_gain=config["preprocess"]["blue_gain"],
                shape=img.shape[-3:],
            )
            ax = plot_image(original, gamma=config["display"]["gamma"])
            ax.set_title("Ground truth image")
            plt.savefig(plib.Path(save) / "original.png")

            # compute metrics
            from torchmetrics.image import lpip, psnr

            lpips_func = lpip.LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True)
            psnr_funct = psnr.PeakSignalNoiseRatio()

            img_torch = torch.from_numpy(img).squeeze(0)
            original_torch = torch.from_numpy(original).unsqueeze(0)

            # channel as first dimension
            img_torch = img_torch.movedim(-1, -3)
            original_torch = original_torch.movedim(-1, -3)

            # normalize, TODO img max value is 14 which seems strange
            img_torch = img_torch / torch.amax(img_torch)

            # compute metrics
            lpips = lpips_func(img_torch, original_torch)
            psnr = psnr_funct(img_torch, original_torch)
            print(f"LPIPS : {lpips}")
            print(f"PSNR : {psnr}")

        # If the recon algorithm is unrolled and has a preprocessing step, plot result without preprocessing
        if config.admm.unrolled and recon.pre_process is not None:
            recon.set_data(data)
            recon.pre_process = None
            with torch.no_grad():
                res = recon.apply(
                    disp_iter=disp,
                    save=False,
                    gamma=config["display"]["gamma"],
                    plot=config["display"]["plot"],
                    output_intermediate=True,
                )

            img = res[0].cpu().numpy()
            np.save(plib.Path(save) / "final_reconstruction_no_preprocessing.npy", img[0])
            ax = plot_image(img, gamma=config["display"]["gamma"])
            plt.savefig(plib.Path(save) / "final_reconstruction_no_preprocessing.png")
            pre_post_process_image = res[2].cpu().numpy()
            ax = plot_image(pre_post_process_image, gamma=config["display"]["gamma"])
            plt.savefig(plib.Path(save) / "pre_post_process_no_preprocessing.png")

        print(f"Files saved to : {save}")


if __name__ == "__main__":
    admm()
