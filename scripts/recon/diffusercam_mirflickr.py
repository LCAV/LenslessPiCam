import hydra
import yaml
import torch
from lensless.utils.plot import plot_image
from lensless.utils.dataset import DiffuserCamMirflickr
from torch.utils.data import Subset
import numpy as np
import os
from lensless.utils.io import save_image
import time
from lensless import ADMM
from lensless.recon.model_dict import load_model, download_model


repo_path = "/home/bezzam/LenslessPiCam"


@hydra.main(
    version_base=None, config_path="../../configs/recon", config_name="recon_diffusercam_mirflickr"
)
def apply_model(config):
    idx = config.idx
    save = config.save
    device = config.device
    n_trials = config.n_trials
    legacy_denoiser = config.legacy_denoiser

    # load dataset
    dataset = DiffuserCamMirflickr(
        dataset_dir=config.files.dataset,
        psf_path=os.path.join(repo_path, config.files.psf),
        downsample=config.files.downsample,
    )
    test_indices = dataset.allowed_idx[dataset.allowed_idx <= 1000]
    test_set = Subset(dataset, test_indices)
    psf = dataset.psf.to(device)
    print("Test set size:", len(test_set))
    print(f"Data shape :  {test_set[0][0].shape}")

    # load model
    model_name = config.model_name
    if model_name is None:
        print("Using traditional ADMM")
        recon = ADMM(psf, **config.admm)

    else:

        # load config
        model_path = download_model(camera="diffusercam", dataset="mirflickr", model=model_name)
        config_path = os.path.join(model_path, ".hydra", "config.yaml")
        with open(config_path, "r") as stream:
            model_config = yaml.safe_load(stream)

        # -- set seed
        seed = model_config["seed"]
        torch.manual_seed(seed)
        np.random.seed(seed)

        # load model
        recon = load_model(model_path, psf=psf, device=device, legacy_denoiser=legacy_denoiser)

    # apply reconstruction
    lensless, lensed = test_set[idx]

    start_time = time.time()
    for _ in range(n_trials):
        with torch.no_grad():

            recon.set_data(lensless.to(device))
            res = recon.apply(
                disp_iter=-1,
                save=False,
                gamma=None,
                plot=False,
            )
    end_time = time.time()
    avg_time_ms = (end_time - start_time) / n_trials * 1000
    print(f"Avg inference [ms] : {avg_time_ms:.3} ms")

    res_np = res[0].cpu().numpy().squeeze()

    plot_image(res_np)
    plot_image(lensed)

    if save:
        print(f"Saving images to {os.getcwd()}")
        lensed_np = lensed[0].cpu().numpy()
        save_image(lensed_np, f"{idx}_original_idx.png")
        save_image(res_np, f"{idx}_reconstruction_idx.png")
        save_image(lensless[0].cpu().numpy(), f"{idx}_lensless.png")


if __name__ == "__main__":
    apply_model()
