import hydra
import yaml
import torch
from lensless import ADMM
from lensless.utils.plot import plot_image
from lensless.utils.dataset import DigiCam
import os
from lensless.utils.io import save_image
import time
from lensless.recon.model_dict import download_model, load_model


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

    # load dataset
    test_set = DigiCam(
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
    )
    test_set.psf = test_set.psf.to(device)
    print("Test set size: ", len(test_set))

    # load model
    if model_name == "admm":
        recon = ADMM(test_set.psf, n_iter=config.n_iter)
    else:
        # load best model
        recon = load_model(model_path, test_set.psf, device)

    # apply reconstruction
    lensless, lensed = test_set[idx]
    lensless = lensless.to(device)

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
    plot_image(lensed)

    if save:
        print(f"Saving images to {os.getcwd()}")
        alignment = test_set.alignment
        top_right = alignment["topright"]
        height = alignment["height"]
        width = alignment["width"]
        res_np = img[top_right[0] : top_right[0] + height, top_right[1] : top_right[1] + width]
        lensed_np = lensed[0].cpu().numpy()
        save_image(lensed_np, f"original_idx{idx}.png")
        save_image(res_np, f"{model_name}_idx{idx}.png")
        save_image(lensless[0].cpu().numpy(), f"lensless_idx{idx}.png")


if __name__ == "__main__":
    apply_pretrained()
