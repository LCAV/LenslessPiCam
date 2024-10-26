import hydra
import yaml
import torch
from lensless import ADMM
from lensless.utils.dataset import HFDataset
import os
from lensless.utils.io import save_image
from tqdm import tqdm
from lensless.recon.model_dict import download_model, load_model
import numpy as np
from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics.image import lpip, psnr
import json
from matplotlib import pyplot as plt


@hydra.main(
    version_base=None, config_path="../../configs", config_name="recon_digicam_mirflickr_err"
)
def apply_pretrained(config):
    device = config.device
    model_name = config.model
    percent_pixels_wrong = config.percent_pixels_wrong

    if config.metrics_fp is not None:

        # load metrics from file
        with open(config.metrics_fp, "r") as f:
            metrics_values = json.load(f)

        # # if not normalized... all PSFs have roughtly same norm
        # metrics_values["psf_err"] = np.array(metrics_values["psf_err"]) / 1.7302357e-06
        # metrics_values["psf_err"] = metrics_values["psf_err"].tolist()

        # # resave metrics dict to JSON
        # with open(config.metrics_fp, "w") as f:
        #     json.dump(metrics_values, f, indent=4)

    else:

        # load config
        if model_name == "admm":
            # take config from unrolled ADMM for dataset
            model_path = download_model(
                camera="digicam", dataset="mirflickr_multi_25k", model="Unet4M+U5+Unet4M_wave"
            )
            config_path = os.path.join(model_path, ".hydra", "config.yaml")
            with open(config_path, "r") as stream:
                model_config = yaml.safe_load(stream)

        else:
            model_path = download_model(
                camera="digicam", dataset="mirflickr_multi_25k", model=model_name
            )
            config_path = os.path.join(model_path, ".hydra", "config.yaml")
            with open(config_path, "r") as stream:
                model_config = yaml.safe_load(stream)

        metrics = {
            "PSNR": psnr.PeakSignalNoiseRatio(reduction=None, dim=(1, 2, 3), data_range=(0, 1)).to(
                device
            ),
            "SSIM": StructuralSimilarityIndexMeasure(reduction=None, data_range=(0, 1)).to(device),
            "LPIPS_Vgg": lpip.LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=True, reduction="sum"
            ).to(device),
            "psf_err": torch.nn.functional.mse_loss,
        }

        # load data
        # TODO missing simulation parameters???
        test_set = HFDataset(
            huggingface_repo=model_config["files"]["dataset"]
            if config.hf_repo is None
            else config.hf_repo,
            psf=(
                model_config["files"]["huggingface_psf"]
                if "huggingface_psf" in model_config["files"]
                else None
            ),
            split="test",
            display_res=model_config["files"]["image_res"],
            rotate=model_config["files"]["rotate"],
            flipud=model_config["files"]["flipud"],
            flip_lensed=model_config["files"]["flip_lensed"],
            downsample=model_config["files"]["downsample"],
            alignment=model_config["alignment"],
            simulation_config=model_config["simulation"],
            force_rgb=model_config["files"].get("force_rgb", False),
            cache_dir=config.cache_dir,
            save_psf=False,
            return_mask_label=True,
        )

        # # create Dataset loader
        # batch_size = 4
        # dataloader = torch.utils.data.DataLoader(
        #     dataset=test_set,
        #     batch_size=batch_size,
        #     shuffle=False,
        #     pin_memory=(device != "cpu"),
        # )

        psf_norms = []
        for mask_label in test_set.psf.keys():
            psf_norms.append(np.mean(test_set.psf[mask_label].cpu().numpy().flatten() ** 2))
        psf_norms = np.array(psf_norms)

        n_files = config.n_files
        if n_files is None:
            n_files = len(test_set)
        percent_pixels_wrong = config.percent_pixels_wrong

        # initialize metrics dict
        metrics_values = {k: np.zeros((len(percent_pixels_wrong), n_files)) for k in metrics.keys()}

        for i in config.save_idx:
            # make folder
            save_dir = str(i)
            os.makedirs(save_dir, exist_ok=True)

        for idx in tqdm(range(n_files)):

            # get data
            lensless, lensed, mask_label = test_set[idx]
            lensless = lensless.to(device)

            if idx in config.save_idx:
                if lensed is not None:
                    lensed_np = lensed[0].cpu().numpy()
                    save_image(lensed_np, os.path.join(str(idx), f"original_idx{idx}.png"))
                save_image(
                    lensless[0].cpu().numpy(), os.path.join(str(idx), f"lensless_idx{idx}.png")
                )

            # -- reshape for torchmetrics
            lensed = lensed.reshape(-1, *lensed.shape[-3:]).movedim(-1, -3)
            lensed_max = torch.amax(lensed, dim=(1, 2, 3), keepdim=True)
            lensed = lensed / lensed_max
            lensed = lensed.to(device)

            _metrics_idx = {k: [] for k in metrics.keys()}

            for percent_wrong in percent_pixels_wrong:

                # perturb mask
                mask_vals = test_set.get_mask_vals(mask_label)

                noisy_mask_vals = mask_vals.copy()
                if percent_wrong > 0:

                    n_pixels = mask_vals.size
                    n_wrong_pixels = int(n_pixels * percent_wrong / 100)
                    wrong_pixels = np.random.choice(n_pixels, n_wrong_pixels, replace=False)
                    noisy_mask_vals = noisy_mask_vals.flatten()

                    if config.flip:
                        noisy_mask_vals[wrong_pixels] = (
                            1 - noisy_mask_vals[wrong_pixels]
                        )  # flip pixel value
                    else:
                        # reset values randomly
                        noisy_mask_vals[wrong_pixels] = np.random.uniform(size=n_wrong_pixels)
                    noisy_mask_vals = noisy_mask_vals.reshape(mask_vals.shape)

                    # noise = np.random.uniform(size=mask_vals.shape)
                    # # -- rescale noise to desired SNR
                    # mask_var = ndimage.variance(mask_vals)
                    # noise_var = ndimage.variance(noise)
                    # fact = np.sqrt(mask_var / noise_var / (10 ** (mask_snr_db / 10)))
                    # noisy_mask_vals = mask_vals + fact * noise
                    # # -- clip to [0, 1]
                    # noisy_mask_vals = np.clip(noisy_mask_vals, 0, 1)

                # simulate PSF
                psf = test_set.simulate_psf(noisy_mask_vals)
                psf = psf.to(device)

                # compute L2 error with normal PSF
                _metrics_idx["psf_err"].append(
                    metrics["psf_err"](psf, test_set.psf[mask_label].to(device)).item()
                    / psf_norms[mask_label]
                )

                # load model
                if model_name == "admm":
                    recon = ADMM(psf, n_iter=config.n_iter)
                else:
                    # load best model
                    recon = load_model(model_path, psf, device, verbose=False)

                # reconstruct
                with torch.no_grad():
                    recon.set_data(lensless)
                    res = recon.apply(
                        disp_iter=-1,
                        save=False,
                        gamma=None,
                        plot=False,
                    )
                recon = res[0]

                # prepare for metrics
                # -- convert to [N*D, C, H, W] for torchmetrics
                prediction = recon.reshape(-1, *recon.shape[-3:]).movedim(-1, -3)
                # - extract ROI
                prediction = test_set.extract_roi(prediction, axis=(-2, -1))
                # -- normalize
                prediction_max = torch.amax(prediction, dim=(1, 2, 3), keepdim=True)
                prediction = prediction / prediction_max

                for k, metric in metrics.items():
                    if k == "psf_err":
                        continue
                    _metrics_idx[k].append(metric(prediction, lensed).item())

                # save
                if idx in config.save_idx:
                    img = recon.cpu().numpy().squeeze()
                    alignment = test_set.alignment
                    top_left = alignment["top_left"]
                    height = alignment["height"]
                    width = alignment["width"]
                    res_np = img[
                        top_left[0] : top_left[0] + height, top_left[1] : top_left[1] + width
                    ]
                    fp = os.path.join(str(idx), f"{model_name}_percentwrong{percent_wrong}.png")
                    save_image(res_np, fp)

            # save metrics
            for k, v in _metrics_idx.items():
                metrics_values[k][:, idx] = v

        # save metric dict to JSON
        # -- make sure to convert numpy arrays to lists
        for k, v in metrics_values.items():
            metrics_values[k] = v.tolist()
        with open(f"{model_name}_metrics.json", "w") as f:
            json.dump(metrics_values, f, indent=4)

    # plot each metrics vs percent_wrong
    for k, v in metrics_values.items():
        plt.figure()
        plt.xlabel("Percent pixels wrong [%]")
        if k == "psf_err":
            plt.plot(percent_pixels_wrong, np.mean(v, axis=1) * 100)
            plt.ylabel("Relative PSF error [%]")
        else:
            plt.plot(percent_pixels_wrong, np.mean(v, axis=1))
            plt.ylabel(k)

        # save plot
        # - tight
        plt.tight_layout()
        plt.savefig(f"{k}_{model_name}.png")


if __name__ == "__main__":
    apply_pretrained()
