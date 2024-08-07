import hydra
import yaml
import torch
from lensless import ADMM
from lensless.utils.plot import plot_image
from lensless.utils.dataset import HFDataset
import os
from lensless.utils.io import save_image
import time
from lensless.recon.model_dict import download_model, load_model
from huggingface_hub import hf_hub_download
from lensless.utils.io import load_image


@hydra.main(version_base=None, config_path="../../configs", config_name="recon_digicam_mirflickr")
def apply_pretrained(config):
    idx = config.idx
    save = config.save
    device = config.device
    n_trials = config.n_trials
    model_name = config.model

    # load config
    if model_name == "admm":
        # take config from unrolled ADMM for dataset
        model_path = download_model(camera="digicam", dataset="mirflickr_single_25k", model="U10")
        config_path = os.path.join(model_path, ".hydra", "config.yaml")
        with open(config_path, "r") as stream:
            model_config = yaml.safe_load(stream)

    else:
        model_path = download_model(
            camera="digicam", dataset="mirflickr_single_25k", model=model_name
        )
        config_path = os.path.join(model_path, ".hydra", "config.yaml")
        with open(config_path, "r") as stream:
            model_config = yaml.safe_load(stream)

    # load data
    test_set = HFDataset(
        huggingface_repo=model_config["files"]["dataset"],
        psf=model_config["files"]["huggingface_psf"]
        if "huggingface_psf" in model_config["files"]
        else None,
        split="test",
        display_res=model_config["files"]["image_res"],
        rotate=model_config["files"]["rotate"],
        downsample=model_config["files"]["downsample"],
        alignment=model_config["alignment"],
        save_psf=model_config["files"]["save_psf"],
        simulation_config=model_config["simulation"],
        force_rgb=model_config["files"].get("force_rgb", False),
        cache_dir=config.cache_dir,
    )
    psf = test_set.psf.to(device)
    print("Test set size: ", len(test_set))

    if config.fn is not None:
        raw_data_fp = hf_hub_download(
            repo_id=model_config["files"]["dataset"], filename=config.fn, repo_type="dataset"
        )
        lensless = load_image(
            fp=raw_data_fp,
            return_float=True,
            as_4d=True,
        )
        lensless = torch.from_numpy(lensless).to(psf)
        if config.rotate:
            lensless = torch.rot90(lensless, dims=(-3, -2), k=2)

        lensed = None

    else:

        lensless, lensed = test_set[idx]
        lensless = lensless.to(device)

    # load model
    if model_name == "admm":
        recon = ADMM(psf, n_iter=config.n_iter)
    else:
        # load best model
        recon = load_model(model_path, psf, device)

    # print data shape
    print(f"Data shape :  {lensless.shape}")

    # apply reconstruction
    start_time = time.time()
    for _ in range(n_trials):
        with torch.no_grad():

            recon.set_data(lensless)
            res = recon.apply(
                disp_iter=-1,
                save=False,
                gamma=None,
                plot=False,
            )
    end_time = time.time()
    avg_time_ms = (end_time - start_time) / n_trials * 1000
    print(f"Avg inference [ms] : {avg_time_ms} ms")

    img = res[0].cpu().numpy().squeeze()

    plot_image(img)
    if lensed is not None:
        plot_image(lensed)

    if save:
        print(f"Saving images to {os.getcwd()}")

        if config.fn is not None:
            dim = config.alignment.dim
            top_left = config.alignment.top_left
            res_np = img[top_left[0] : top_left[0] + dim[0], top_left[1] : top_left[1] + dim[1]]
            bn = config.fn.split(".")[0]
            save_image(res_np, f"{model_name}_{bn}.png")
        else:
            alignment = test_set.alignment
            top_left = alignment["top_left"]
            height = alignment["height"]
            width = alignment["width"]
            res_np = img[top_left[0] : top_left[0] + height, top_left[1] : top_left[1] + width]
            save_image(res_np, f"{model_name}_idx{idx}.png")
        if lensed is not None:
            lensed_np = lensed[0].cpu().numpy()
            save_image(lensed_np, f"original_idx{idx}.png")
        save_image(lensless[0].cpu().numpy(), f"lensless_idx{idx}.png")


if __name__ == "__main__":
    apply_pretrained()
