# #############################################################################
# benchmark_recon.py
# =========
# Authors :
# Yohann PERRON
# #############################################################################

"""
Benchmark reconstruction algorithms
==============
This script benchmarks reconstruction algorithms on the DiffuserCam dataset.
The algorithm benchmarked and the number of iterations can be set in the config file : benchmark.yaml.
For unrolled algorithms, the results of the unrolled training (json file) are loaded from the benchmark/results folder.
"""

import hydra
from hydra.utils import get_original_cwd

import glob
import json
import os
import pathlib as plib
from lensless.benchmark import benchmark, DiffuserCamTestDataset
import matplotlib.pyplot as plt
from lensless import ADMM, FISTA, GradientDescent, NesterovGradientDescent

try:
    import torch
except ImportError:
    raise ImportError("Torch and torchmetrics are needed to benchmark reconstruction algorithm")


@hydra.main(version_base=None, config_path="../../configs", config_name="benchmark")
def benchmark_recon(config):
    downsample = config.downsample
    n_files = config.n_files
    n_iter_range = config.n_iter_range

    # check if GPU is available
    if torch.cuda.is_available() and config.device[:4] == "cuda":
        device = config.device
    else:
        device = "cpu"

    # Benchmark dataset
    benchmark_dataset = DiffuserCamTestDataset(
        data_dir=os.path.join(get_original_cwd(), "data"), n_files=n_files, downsample=downsample
    )
    psf = benchmark_dataset.psf.to(device)

    model_list = []  # list of algoritms to benchmark
    if "ADMM" in config.algorithms:
        model_list.append(
            (
                "ADMM",
                ADMM(
                    psf,
                    mu1=config.admm.mu1,
                    mu2=config.admm.mu2,
                    mu3=config.admm.mu3,
                    tau=config.admm.tau,
                ),
            )
        )
    if "ADMM_Monakhova2019" in config.algorithms:
        model_list.append(("ADMM_Monakhova2019", ADMM(psf, mu1=1e-4, mu2=1e-4, mu3=1e-4, tau=2e-3)))
    if "FISTA" in config.algorithms:
        model_list.append(("FISTA", FISTA(psf, tk=config.fista.tk)))
    if "GradientDescent" in config.algorithms:
        model_list.append(("GradientDescent", GradientDescent(psf)))
    if "NesterovGradientDescent" in config.algorithms:
        model_list.append(
            (
                "NesterovGradientDescent",
                NesterovGradientDescent(psf, p=config.nesterov.p, mu=config.nesterov.mu),
            )
        )
    # APGD is not supported yet
    # if "APGD" in config.algorithms:
    #     from lensless import APGD

    #     model_list.append(("APGD", APGD(psf)))
    if "GradientDescentPnPBm3D" in config.algorithms:
        import bm3d
        import numpy as np

        def denoiser(x):
            np.clip(x, 0, 1, out=x)
            return bm3d.bm3d(x, sigma_psd=10 / 255, stage_arg=bm3d.BM3DStages.ALL_STAGES)

        model_list.append(("GradientDescentPnPBm3D", GradientDescent(psf.cpu(), proj=denoiser)))
    if (
        "GradientDescentPnPDruNet" in config.algorithms
        or "FISTAPnPDruNet" in config.algorithms
        or "ADMMPnPDruNet" in config.algorithms
    ):
        from lensless.util import load_drunet, apply_CWH_denoizer
    if "GradientDescentPnPDruNet" in config.algorithms:
        drunet = load_drunet(os.path.join(get_original_cwd(), "data/drunet_color.pth")).to(device)

        def denoiserG(x):
            x_max = torch.amax(x, dim=(-2, -3), keepdim=True)
            x_denoized = apply_CWH_denoizer(drunet, x / x_max, noise_level=0.5, device=device)
            x_denoized = torch.clip(x_denoized, min=0.0) * x_max.to(device)
            return x + 0.05 * (x_denoized - x)

        model_list.append(("GradientDescentPnPDruNet", GradientDescent(psf, proj=denoiserG)))
    if "FISTAPnPDruNet" in config.algorithms:
        drunet = load_drunet(os.path.join(get_original_cwd(), "data/drunet_color.pth")).to(device)

        def denoiserF(x):
            torch.clip(x, min=0.0, out=x)
            x_max = torch.amax(x, dim=(-2, -3), keepdim=True)
            x_denoized = apply_CWH_denoizer(drunet, x / x_max, noise_level=0.5, device=device)
            x_denoized = torch.clip(x_denoized, min=0.0) * x_max.to(device)
            return x_denoized

        model_list.append(("FISTAPnPDruNetcst0.5", FISTA(psf, tk=config.fista.tk, proj=denoiserF)))
    if "ADMMPnPDruNet" in config.algorithms:
        from lensless.admmPnP import ADMM_PnP

        drunet = load_drunet(os.path.join(get_original_cwd(), "data/drunet_color.pth")).to(device)

        def denoiserA(x):
            torch.clip(x, min=0.0, out=x)
            x_max = torch.amax(x, dim=(-2, -3), keepdim=True) + 1e-6
            x_denoized = apply_CWH_denoizer(drunet, x / x_max, noise_level=0.5, device=device)
            x_denoized = torch.clip(x_denoized, min=0.0) * x_max.to(device)
            return x_denoized

        model_list.append(
            (
                "ADMMPnPDruNetcst0.5",
                ADMM_PnP(
                    psf,
                    denoiserA,
                    mu1=config.admm.mu1,
                    mu2=config.admm.mu2,
                    mu3=config.admm.mu3,
                    tau=config.admm.tau,
                ),
            )
        )
    # if "FISTAPnPDruNet" in config.algorithms:
    #     drunet = load_drunet(os.path.join(get_original_cwd(), "data/drunet_color.pth")).to(device)

    #     def denoiser3(x, noise_level=1):
    #         torch.clip(x, min=0.0, out=x)
    #         x_max = torch.amax(x, dim=(-2, -3), keepdim=True)
    #         x_denoized = apply_CWH_denoizer(
    #             drunet, x / x_max, noise_level=noise_level, device=device
    #         )
    #         x_denoized = torch.clip(x_denoized, min=0.0) * x_max.to(device)
    #         return x_denoized

    #     model_list.append(("FISTAPnPDruNetlog", FISTA(psf, tk=config.fista.tk, proj=denoiser3)))

    results = {}
    # benchmark each model for different number of iteration and append result to results
    for model_name, model in model_list:
        results[model_name] = []
        print(f"Running benchmark for {model_name}")
        for n_iter in n_iter_range:
            result = benchmark(model, benchmark_dataset, batchsize=1, n_iter=n_iter)
            result["n_iter"] = n_iter
            results[model_name].append(result)

    # create folder to load results from trained algorithms
    result_dir = os.path.join(get_original_cwd(), "benchmark", "trained_results")
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)

    # try to load json files with results form unrolled training
    files = glob.glob(os.path.join(result_dir, "*.json"))
    unrolled_results = {}
    for file in files:
        model_name = plib.Path(file).stem
        unrolled_results[model_name] = {}
        with open(file, "r") as f:
            result = json.load(f)

            # get result for each metric
            for metric in result.keys():
                # if list take last value (last epoch)
                if isinstance(result[metric], list):
                    unrolled_results[model_name][metric] = result[metric][-1]
                else:
                    unrolled_results[model_name][metric] = result[metric]

    # for each metrics plot the results comparing each model
    metrics_to_plot = ["SSIM", "PSNR", "MSE", "LPIPS", "ReconstructionError"]
    for metric in metrics_to_plot:
        plt.figure()
        # plot benchmarked algorithm
        for model_name in results.keys():
            plt.plot(
                [result["n_iter"] for result in results[model_name]],
                [result[metric] for result in results[model_name]],
                label=model_name,
            )

        # plot unrolled algorithms results
        color_list = ["red", "green", "blue", "orange", "purple"]
        algorithm_colors = {}
        for model_name in unrolled_results.keys():
            # use algorithm name if defined, else use file name
            if "algorithm" in unrolled_results[model_name].keys():
                plot_name = unrolled_results[model_name]["algorithm"]
            else:
                plot_name = model_name

            # set color depending on plot name using same color for same algorithm
            if plot_name not in algorithm_colors.keys():
                algorithm_colors[plot_name] = color_list.pop()
            color = algorithm_colors[plot_name]

            # check if metric is defined
            if metric not in unrolled_results[model_name].keys():
                continue
            # if n_iter is undefined, plot as horizontal line
            if "n_iter" not in unrolled_results[model_name].keys():
                plt.hlines(
                    unrolled_results[model_name][metric],
                    0,
                    n_iter_range[-1],
                    label=plot_name,
                    linestyles="dashed",
                    colors=color,
                )
            else:
                # plot as point
                plt.plot(
                    unrolled_results[model_name]["n_iter"],
                    unrolled_results[model_name][metric],
                    label=plot_name,
                    marker="o",
                    color=color,
                )
        plt.title(metric)
        plt.xlabel("Number of iterations")
        plt.ylabel(metric)
        plt.legend()
        plt.savefig(f"{metric}.png")


if __name__ == "__main__":
    benchmark_recon()
