import hydra
import yaml
import torch
from lensless.recon.utils import create_process_network
from lensless.recon.unrolled_admm import UnrolledADMM
from lensless.utils.plot import plot_image
from lensless.utils.dataset import DigiCamCelebA
import numpy as np
import os
from lensless.utils.io import save_image
import time
from lensless.recon.model_dict import download_model


@hydra.main(version_base=None, config_path="../../configs", config_name="recon_digicam_celeba")
def apply_unrolled(config):
    idx = config.idx
    save = config.save
    device = config.device
    n_trials = config.n_trials
    model_name = config.model

    # load config
    model_path = download_model(camera="digicam", dataset="celeba_26k", model=model_name)
    config_path = os.path.join(model_path, ".hydra", "config.yaml")
    with open(config_path, "r") as stream:
        config = yaml.safe_load(stream)

    # -- set seed
    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    generator = torch.Generator().manual_seed(seed)

    # load PSF
    # TODO don't use PSF from config
    psf_path = "/home/bezzam/LenslessPiCam/outputs/2023-10-25/09-40-03/adafruit_random_pattern_20231004_174047_SIM_psf.png"
    # psf_path = os.path.join(config["files"]["psf"])
    print("PSF path : ", psf_path)
    if not os.path.exists(psf_path):
        raise FileNotFoundError("PSF file not found")

    # load dataset
    dataset = DigiCamCelebA(
        data_dir=config["files"]["dataset"],
        celeba_root=config["files"]["celeba_root"],
        psf_path=psf_path,
        downsample=config["files"]["downsample"],
        simulation_config=config["simulation"],
        crop=config["files"]["crop"],
    )
    dataset.psf = dataset.psf.to(device)
    print(f"Data shape :  {dataset[0][0].shape}")

    # -- train-test split
    train_size = int((1 - config["files"]["test_size"]) * len(dataset))
    test_size = len(dataset) - train_size
    _, test_set = torch.utils.data.random_split(
        dataset, [train_size, test_size], generator=generator
    )
    print("Test set size: ", len(test_set))

    if "skip_unrolled" not in config["reconstruction"].keys():
        config["reconstruction"]["skip_unrolled"] = False

    # load best model
    model_checkpoint = os.path.join(model_path, "recon_epochBEST")
    model_state_dict = torch.load(model_checkpoint, map_location=device)

    pre_process = None
    post_process = None

    if config["reconstruction"]["pre_process"]["network"] is not None:

        pre_process, _ = create_process_network(
            network=config["reconstruction"]["pre_process"]["network"],
            depth=config["reconstruction"]["pre_process"]["depth"],
            nc=config["reconstruction"]["pre_process"]["nc"]
            if "nc" in config["reconstruction"]["pre_process"].keys()
            else None,
            device=device,
        )

    if config["reconstruction"]["post_process"]["network"] is not None:

        post_process, _ = create_process_network(
            network=config["reconstruction"]["post_process"]["network"],
            depth=config["reconstruction"]["post_process"]["depth"],
            nc=config["reconstruction"]["post_process"]["nc"]
            if "nc" in config["reconstruction"]["post_process"].keys()
            else None,
            device=device,
        )

    if config["reconstruction"]["method"] == "unrolled_admm":
        recon = UnrolledADMM(
            dataset.psf,
            pre_process=pre_process,
            post_process=post_process,
            n_iter=config["reconstruction"]["unrolled_admm"]["n_iter"],
            skip_unrolled=config["reconstruction"]["skip_unrolled"],
        )

        recon.load_state_dict(model_state_dict)

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
    print(f"Avg inference [ms] : {avg_time_ms} ms")

    img = res[0].cpu().numpy().squeeze()

    plot_image(img)
    plot_image(lensed)

    if save:
        print(f"Saving images to {os.getcwd()}")
        crop = dataset.crop
        res_np = img[
            crop["vertical"][0] : crop["vertical"][1],
            crop["horizontal"][0] : crop["horizontal"][1],
        ]
        lensed_np = lensed[0].cpu().numpy()
        lensed_np = lensed_np[
            crop["vertical"][0] : crop["vertical"][1],
            crop["horizontal"][0] : crop["horizontal"][1],
        ]
        save_image(lensed_np, f"original_idx{idx}.png")
        save_image(res_np, f"{model_name}_idx{idx}.png")
        save_image(lensless[0].cpu().numpy(), f"lensless_idx{idx}.png")


if __name__ == "__main__":
    apply_unrolled()


# #### OLD STUFF


# # apply reconstruction on a test image
# idx = 9
# save = True
# device = "cuda:0"
# n_trials = 1


# # -- unrolled10
# model_path = "/home/bezzam/LenslessPiCam/outputs/2023-10-24/19-30-13"

# # # -- Unet
# # model_path = "/home/bezzam/LenslessPiCam/outputs/2023-10-29/12-47-05"

# # unrolled10 + P8
# model_path = "/home/bezzam/LenslessPiCam/outputs/2023-10-25/17-59-47"

# # # fine-tuned unrolled10 + P8
# # model_path = "/home/bezzam/LenslessPiCam/outputs/2023-10-28/05-28-17"

# # # P8 + unrolled10
# # model_path = "/home/bezzam/LenslessPiCam/outputs/2023-10-26/11-52-05"

# # # P4 + unrolled10 + P4
# # model_path = "/home/bezzam/LenslessPiCam/outputs/2023-10-31/18-26-08"

# # # pre + unrolled10 + post + FT
# # model_path = "/home/bezzam/LenslessPiCam/outputs/2023-11-01/12-35-20"

# # model_path = metrics_dict["unrolled10"]
# # model_path = metrics_dict["unrolled10_sim"]
# # model_path = metrics_dict["unrolled10_sim_post"]
# # model_path = metrics_dict["unrolled10_ft_mask+cf"]
# # model_path = metrics_dict["unet"]
# # model_path = metrics_dict["unrolled10_sim_pre"]
# # model_path = metrics_dict["unrolled10_pre_post"]

# # Read YAML file
# config_path = os.path.join(model_path, ".hydra", "config.yaml")
# with open(config_path, "r") as stream:
#     config = yaml.safe_load(stream)


# seed = config["seed"]
# torch.manual_seed(seed)
# np.random.seed(seed)
# generator = torch.Generator().manual_seed(seed)

# # TODO hack
# # # -- bad
# # psf_path = "outputs/2023-11-26/18-35-03/adafruit_random_pattern_20230719_SIM_psf.png"
# # # -- almost good
# # psf_path = "outputs/2023-11-26/22-38-52/adafruit_random_pattern_20231004_174047_SIM_psf.png"
# # # -- good
# # psf_path = "outputs/2023-11-26/22-43-43/adafruit_random_pattern_20231004_174047_SIM_psf.png"
# # # -- zero one line
# # psf_path = "outputs/2023-11-26/22-48-43/adafruit_random_pattern_20231004_174047_SIM_psf.png"
# # # -- 5% different
# # psf_path = "outputs/2023-11-26/22-57-46/adafruit_random_pattern_20231004_174047_SIM_psf.png"
# psf_path = os.path.join(config["files"]["psf"])
# print("PSF path : ", psf_path)
# if not os.path.exists(psf_path):
#     raise FileNotFoundError("PSF file not found")

# dataset = DigiCamCelebA(
#     data_dir=config["files"]["dataset"],
#     celeba_root=config["files"]["celeba_root"],
#     psf_path=psf_path,
#     downsample=config["files"]["downsample"],
#     simulation_config=config["simulation"],
#     crop=config["files"]["crop"],
# )
# dataset.psf = dataset.psf.to(device)
# print(f"Data shape :  {dataset[0][0].shape}")

# # import pudb; pudb.set_trace()

# # train-test split
# train_size = int((1 - config["files"]["test_size"]) * len(dataset))
# test_size = len(dataset) - train_size
# train_set, test_set = torch.utils.data.random_split(
#     dataset, [train_size, test_size], generator=generator
# )
# print("Test set size: ", len(test_set))


# if "skip_unrolled" not in config["reconstruction"].keys():
#     config["reconstruction"]["skip_unrolled"] = False


# # load best model
# model_checkpoint = os.path.join(model_path, "recon_epochBEST")
# # assert os.path.exists(model_checkpoint), "Checkpoint does not exist"
# # print("Loading checkpoint from : ", model_checkpoint)
# model_state_dict = torch.load(model_checkpoint, map_location=device)
# # assert "_mu1_p" in model_state_dict.keys()

# pre_process = None
# post_process = None

# if config["reconstruction"]["pre_process"]["network"] is not None:

#     pre_process, _ = lensless.recon.utils.create_process_network(
#         network=config["reconstruction"]["pre_process"]["network"],
#         depth=config["reconstruction"]["pre_process"]["depth"],
#         nc=config["reconstruction"]["pre_process"]["nc"]
#         if "nc" in config["reconstruction"]["pre_process"].keys()
#         else None,
#         device=device,
#     )

# if config["reconstruction"]["post_process"]["network"] is not None:

#     post_process, _ = lensless.recon.utils.create_process_network(
#         network=config["reconstruction"]["post_process"]["network"],
#         depth=config["reconstruction"]["post_process"]["depth"],
#         nc=config["reconstruction"]["post_process"]["nc"]
#         if "nc" in config["reconstruction"]["post_process"].keys()
#         else None,
#         device=device,
#     )

# if config["reconstruction"]["method"] == "unrolled_admm":
#     recon = UnrolledADMM(
#         dataset.psf,
#         pre_process=pre_process,
#         post_process=post_process,
#         n_iter=config["reconstruction"]["unrolled_admm"]["n_iter"],
#         skip_unrolled=config["reconstruction"]["skip_unrolled"],
#     )

#     recon.load_state_dict(model_state_dict)

# # apply reconstruction

# # import pudb; pudb.set_trace()

# # lensless, lensed = train_set[idx]
# lensless, lensed = test_set[idx]
# # lensless, lensed = dataset[idx]


# start_time = time.time()
# for _ in range(n_trials):
#     with torch.no_grad():

#         recon.set_data(lensless.to(device))
#         res = recon.apply(
#             disp_iter=-1,
#             save=False,
#             gamma=None,
#             plot=False,
#         )
# end_time = time.time()
# avg_time_ms = (end_time - start_time) / n_trials * 1000
# print(f"Avg inference [ms] : {avg_time_ms} ms")

# img = res[0].cpu().numpy().squeeze()
# print(img.shape)

# plot_image(img)
# plot_image(lensed)

# if save:
#     crop = dataset.crop
#     res_np = img[
#         crop["vertical"][0] : crop["vertical"][1],
#         crop["horizontal"][0] : crop["horizontal"][1],
#     ]
#     lensed_np = lensed[0].cpu().numpy()
#     lensed_np = lensed_np[
#         crop["vertical"][0] : crop["vertical"][1],
#         crop["horizontal"][0] : crop["horizontal"][1],
#     ]
#     save_image(lensed_np, f"original_{idx}.png")
#     save_image(res_np, f"{idx}.png")
#     save_image(lensless[0].cpu().numpy(), f"lensless_{idx}.png")
