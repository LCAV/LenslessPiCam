import hydra
import yaml
import torch
from lensless import ADMM
from lensless.utils.plot import plot_image
import os
from lensless.utils.io import save_image
import time
from lensless.recon.model_dict import download_model, load_model
from lensless.utils.dataset import get_dataset


@hydra.main(version_base=None, config_path="../../configs/recon", config_name="diffusercam")
def apply_pretrained(config):
    save = config.save
    device = config.device
    n_trials = config.n_trials
    model_name = config.model
    dataset_name = "mirflickr"

    # load config
    if model_name == "admm":
        # take config from any model just to get dataset hyperparameters
        model_path = download_model(camera="diffusercam", dataset=dataset_name, model="U20")
        config_path = os.path.join(model_path, ".hydra", "config.yaml")
        with open(config_path, "r") as stream:
            model_config = yaml.safe_load(stream)

    else:
        model_path = download_model(camera="diffusercam", dataset=dataset_name, model=model_name)
        config_path = os.path.join(model_path, ".hydra", "config.yaml")
        with open(config_path, "r") as stream:
            model_config = yaml.safe_load(stream)
    model_config["files"]["cache_dir"] = config.cache_dir

    # load data
    test_set = get_dataset(
        dataset_name="diffusercam_mirflickr",
        split="test",
        **model_config["files"],
    )
    psf = test_set.psf.to(device)
    print("Test set size: ", len(test_set))

    # reconstruction specified example
    idx = config.idx
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
        save_image(img, f"{model_name}_{idx}.png")
        if lensed is not None:
            lensed_np = lensed[0].cpu().numpy()
            save_image(lensed_np, f"original_{idx}.png")
        save_image(lensless[0].cpu().numpy(), f"lensless_{idx}.png")


if __name__ == "__main__":
    apply_pretrained()
