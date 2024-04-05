"""
First key is camera, second key is training data, third key is model name.

Download link corresponds to output folder from training
script, which contains the model checkpoint and config file,
and other intermediate files. Models are stored on Hugging Face.
"""


import os
import numpy as np
import torch
from lensless.recon.utils import create_process_network
from lensless.recon.unrolled_admm import UnrolledADMM
from lensless.recon.trainable_inversion import TrainableInversion
from lensless.hardware.trainable_mask import prep_trainable_mask
import yaml
from huggingface_hub import snapshot_download
from collections import OrderedDict


model_dir_path = os.path.join(os.path.dirname(__file__), "..", "..", "models")

model_dict = {
    "diffusercam": {
        "mirflickr": {
            # -- only unrolled20
            "U20": "bezzam/diffusercam-mirflickr-unrolled-admm20",
            "U20_0db": "bezzam/diffusercam-mirflickr-unrolled-admm20-0db",
            "U20_10db": "bezzam/diffusercam-mirflickr-unrolled-admm20-10db",
            "U20_20db": "bezzam/diffusercam-mirflickr-unrolled-admm20-20db",
            # -- only pre-process
            "Unet+U20": "bezzam/diffusercam-mirflickr-unet2-unrolled-admm20",
            "Unet+U20_0dB": "bezzam/diffusercam-mirflickr-unet2-unrolled-admm20-0db",
            "Unet+U20_10db": "bezzam/diffusercam-mirflickr-unet2-unrolled-admm20-10db",
            "Unet+U20_20db": "bezzam/diffusercam-mirflickr-unet2-unrolled-admm20-20db",
            # -- only post-process
            "U20+Unet": "bezzam/diffusercam-mirflickr-unrolled-admm20-unet2",
            "U20+Unet_0db": "bezzam/diffusercam-mirflickr-unrolled-admm20-unet2-0db",
            "U20+Unet_10db": "bezzam/diffusercam-mirflickr-unrolled-admm20-unet2-10db",
            "U20+Unet_20db": "bezzam/diffusercam-mirflickr-unrolled-admm20-unet2-20db",
            # -- only post-process (Drunet)
            "U20+Drunet": "bezzam/diffusercam-mirflickr-unrolled-admm20-drunet",
            "TrainInv+Drunet": "bezzam/diffusercam-mirflickr-trainable-inv-drunet",
            # -- both
            "Unet+TrainInv+Unet": "bezzam/diffusercam-mirflickr-unet2-trainable-inv-unet2",
            "Unet+U20+Unet": "bezzam/diffusercam-mirflickr-unet2-unrolled-admm20-unet2",  # init with 0.01 and only pre-proc
            "Unet+U20+Unet_aux0.01": "bezzam/diffusercam-mirflickr-unet2-unrolled-admm20-unet2-aux0.01",
            "Unet+U20+Unet_aux0.03": "bezzam/diffusercam-mirflickr-unet2-unrolled-admm20-unet2-aux0.03",
            "Unet+U20+Unet_aux0.1": "bezzam/diffusercam-mirflickr-unet2-unrolled-admm20-unet2-aux0.1",  # init with 0.01
            "Unet+U20+Unet_aux1": "bezzam/diffusercam-mirflickr-unet2-unrolled-admm20-unet2-aux1",  # init with 0.01 and only pre-proc
            # baseline benchmarks which don't have model file but use ADMM
            "admm_fista": "bezzam/diffusercam-mirflickr-admm-fista",
            "admm_pnp": "bezzam/diffusercam-mirflickr-admm-pnp",
        },
    },
    "digicam": {
        "celeba_26k": {
            "unrolled_admm10": "bezzam/digicam-celeba-unrolled-admm10",
            "unrolled_admm10_ft_psf": "bezzam/digicam-celeba-unrolled-admm10-ft-psf",
            "unet8M": "bezzam/digicam-celeba-unet8M",
            "TrainInv+Unet8M": "bezzam/digicam-celeba-trainable-inv-unet8M",
            "unrolled_admm10_post8M": "bezzam/digicam-celeba-unrolled-admm10-post8M",
            "unrolled_admm10_ft_psf_post8M": "bezzam/digicam-celeba-unrolled-admm10-ft-psf-post8M",
            "pre8M_unrolled_admm10": "bezzam/digicam-celeba-pre8M-unrolled-admm10",
            "pre4M_unrolled_admm10_post4M": "bezzam/digicam-celeba-pre4M-unrolled-admm10-post4M",
            "pre4M_unrolled_admm10_ft_psf_post4M": "bezzam/digicam-celeba-pre4M-unrolled-admm10-ft-psf-post4M",
            "Unet4M+TrainInv+Unet4M": "bezzam/digicam-celeba-unet4M-trainable-inv-unet4M",
            # baseline benchmarks which don't have model file but use ADMM
            "admm_measured_psf": "bezzam/digicam-celeba-admm-measured-psf",
            "admm_simulated_psf": "bezzam/digicam-celeba-admm-simulated-psf",
        },
        "mirflickr_single_25k": {
            "U10": "bezzam/digicam-mirflickr-single-25k-unrolled-admm10",
            "Unet8M": "bezzam/digicam-mirflickr-single-25k-unet8M",
            "TrainInv+Unet8M": "bezzam/digicam-mirflickr-single-25k-trainable-inv-unet8M",
            "U10+Unet8M": "bezzam/digicam-mirflickr-single-25k-unrolled-admm10-unet8M",
            "Unet4M+TrainInv+Unet4M": "bezzam/digicam-mirflickr-single-25k-unet4M-trainable-inv-unet4M",
            "Unet4M+U10+Unet4M": "bezzam/digicam-mirflickr-single-25k-unet4M-unrolled-admm10-unet4M",
        },
        "mirflickr_multi_25k": {
            "Unet8M": "bezzam/digicam-mirflickr-multi-25k-unet8M",
            "Unet4M+U10+Unet4M": "bezzam/digicam-mirflickr-multi-25k-unet4M-unrolled-admm10-unet4M",
        },
    },
}


def remove_data_parallel(old_state_dict):
    new_state_dict = OrderedDict()

    for k, v in old_state_dict.items():
        # remove `module.` from k
        if "module." in k:
            name = k.replace("module.", "")
            # name = k[7:] # remove `module.`
            new_state_dict[name] = v

    return new_state_dict


def download_model(camera, dataset, model, local_model_dir=None):

    """
    Download model from model_dict (if needed).

    Parameters
    ----------
    dataset : str
        Dataset used for training.
    model_name : str
        Name of model.
    """

    if local_model_dir is None:
        local_model_dir = model_dir_path

    if camera not in model_dict:
        raise ValueError(f"Camera {camera} not found in model_dict.")

    if dataset not in model_dict[camera]:
        raise ValueError(f"Dataset {dataset} not found in model_dict.")

    if model not in model_dict[camera][dataset]:
        raise ValueError(f"Model {model} not found in model_dict.")

    repo_id = model_dict[camera][dataset][model]
    model_dir = os.path.join(local_model_dir, camera, dataset, model)

    if not os.path.exists(model_dir):
        snapshot_download(repo_id=repo_id, local_dir=model_dir)

    return model_dir


def load_model(model_path, psf, device="cpu", legacy_denoiser=False, verbose=True):

    """
    Load best model from model path.

    Parameters
    ----------
    model_path : str
        Path to model.
    psf : py:class:`~torch.Tensor`
        PSF tensor.
    device : str
        Device to load model on.
    """

    # load Hydra config
    config_path = os.path.join(model_path, ".hydra", "config.yaml")
    with open(config_path, "r") as stream:
        config = yaml.safe_load(stream)

    # load learning mask
    downsample = config["files"]["downsample"] * 4  # TODO: particular to DiffuserCam?
    mask = prep_trainable_mask(config, psf, downsample=downsample)
    if mask is not None:

        if config["trainable_mask"]["mask_type"] == "TrainablePSF":
            # print("loading best PSF...")

            psf_learned = np.load(os.path.join(model_path, "psf_epochBEST.npy"))
            psf_learned = torch.Tensor(psf_learned).to(psf).unsqueeze(0)

            # -- set values and get new PSF
            with torch.no_grad():
                mask._mask = torch.nn.Parameter(torch.tensor(psf_learned))
                psf = mask.get_psf().to(device)

    # load best model config
    model_checkpoint = os.path.join(model_path, "recon_epochBEST")
    assert os.path.exists(model_checkpoint), "Checkpoint does not exist"
    if verbose:
        print("Loading checkpoint from : ", model_checkpoint)
    model_state_dict = torch.load(model_checkpoint, map_location=device)

    # load model
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
            psf if mask is None else psf_learned,
            pre_process=pre_process,
            post_process=post_process,
            n_iter=config["reconstruction"]["unrolled_admm"]["n_iter"],
            skip_unrolled=config["reconstruction"]["skip_unrolled"],
            legacy_denoiser=legacy_denoiser,
        )
    elif config["reconstruction"]["method"] == "trainable_inv":
        recon = TrainableInversion(
            psf,
            pre_process=pre_process,
            post_process=post_process,
            K=config["reconstruction"]["trainable_inv"]["K"],
            legacy_denoiser=legacy_denoiser,
        )

    if mask is not None:
        psf_learned = torch.nn.Parameter(psf_learned)
        recon._set_psf(psf_learned)

    if config["device_ids"] is not None:
        model_state_dict = remove_data_parallel(model_state_dict)

    # # return model_state_dict
    # if "_psf" in model_state_dict:
    #     # TODO: should not have to do this...
    #     del model_state_dict["_psf"]

    recon.load_state_dict(model_state_dict)

    return recon
