"""
First key is camera, second key is training data, third key is model name.

Download link corresponds to output folder from training
script, which contains the model checkpoint and config file,
and other intermediate files. Models are stored on Hugging Face.
"""

import os
from huggingface_hub import snapshot_download


model_dir_path = os.path.join(os.path.dirname(__file__), "..", "..", "models")

model_dict = {
    "digicam": {
        "celeba_26k": {
            "unrolled_admm10": "bezzam/digicam-celeba-unrolled-admm10",
            "unrolled_admm10_ft_psf": "bezzam/digicam-celeba-unrolled-admm10-ft-psf",
            "unet8M": "bezzam/digicam-celeba-unet8M",
            "unrolled_admm10_post8M": "bezzam/digicam-celeba-unrolled-admm10-post8M",
            "unrolled_admm10_ft_psf_post8M": "bezzam/digicam-celeba-unrolled-admm10-ft-psf-post8M",
            "pre8M_unrolled_admm10": "bezzam/digicam-celeba-pre8M-unrolled-admm10",
            "pre4M_unrolled_admm10_post4M": "bezzam/digicam-celeba-pre4M-unrolled-admm10-post4M",
            "pre4M_unrolled_admm10_post4M_OLD": "bezzam/digicam-celeba-pre4M-unrolled-admm10-post4M_OLD",
            "pre4M_unrolled_admm10_ft_psf_post4M": "bezzam/digicam-celeba-pre4M-unrolled-admm10-ft-psf-post4M",
            # baseline benchmarks which don't have model file but use ADMM
            "admm_measured_psf": "bezzam/digicam-celeba-admm-measured-psf",
            "admm_simulated_psf": "bezzam/digicam-celeba-admm-simulated-psf",
        }
    }
}


def download_model(camera, dataset, model):

    """
    Download model from model_dict (if needed).

    Parameters
    ----------
    dataset : str
        Dataset used for training.
    model_name : str
        Name of model.
    """

    if camera not in model_dict:
        raise ValueError(f"Camera {camera} not found in model_dict.")

    if dataset not in model_dict[camera]:
        raise ValueError(f"Dataset {dataset} not found in model_dict.")

    if model not in model_dict[camera][dataset]:
        raise ValueError(f"Model {model} not found in model_dict.")

    repo_id = model_dict[camera][dataset][model]
    model_dir = os.path.join(model_dir_path, camera, dataset, model)

    if not os.path.exists(model_dir):
        snapshot_download(repo_id=repo_id, local_dir=model_dir)

    return model_dir
