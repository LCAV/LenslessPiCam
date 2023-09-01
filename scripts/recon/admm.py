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
from lensless.utils.io import load_data
from lensless import ADMM


@hydra.main(version_base=None, config_path="../../configs", config_name="defaults_recon")
def admm(config):
    if config.torch:
        try:
            import torch
        except ImportError:
            raise ImportError("Pytorch not found. Please install pytorch to use torch mode.")

    psf, data = load_data(
        psf_fp=to_absolute_path(config.input.psf),
        data_fp=to_absolute_path(config.input.data),
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
        torch=config.torch,
        torch_device=config.torch_device,
        bg_pix=config.preprocess.bg_pix,
        normalize_data=not config.admm.unrolled,
    )

    disp = config["display"]["disp"]
    if disp < 0:
        disp = None

    save = config["save"]
    if save:
        save = os.getcwd()

    start_time = time.time()
    if not config.admm.unrolled:
        # Normal ADMM
        recon = ADMM(psf, **config.admm)
    elif config.admm.pre_trained:
        assert config.torch, "Unrolled ADMM only works with torch"
        from lensless.recon.unrolled_admm import UnrolledADMM

        # load from pre-trained checkpoint
        recon = UnrolledADMM.load_pretrained(
            config.admm.pre_trained_name, config.admm.pre_trained_psf, device=config.torch_device
        )
    else:
        # load from custom checkpoint
        assert config.torch, "Unrolled ADMM only works with torch"
        from lensless.recon.unrolled_admm import UnrolledADMM
        import lensless.recon.utils

        pre_process = lensless.recon.utils.create_process_network(
            network=config.admm.pre_process_model.network,
            depth=config.admm.pre_process_depth.depth,
            device=config.torch_device,
        )
        post_process = lensless.recon.utils.create_process_network(
            network=config.admm.post_process_model.network,
            depth=config.admm.post_process_depth.depth,
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
        np.save(plib.Path(save) / "final_reconstruction.npy", img)
        print(f"Files saved to : {save}")


if __name__ == "__main__":
    admm()
