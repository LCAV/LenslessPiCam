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
from lensless.eval.benchmark import benchmark
import matplotlib.pyplot as plt
from lensless import ADMM, FISTA, GradientDescent, NesterovGradientDescent
from lensless.utils.dataset import DiffuserCamTestDataset

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

    # Baseline results
    baseline_results = {
        "MSE": 0.0618,
        "LPIPS_Alex": 0.4434,
        "ReconstructionError": 13.70,
    }

    # for each metrics plot the results comparing each model
    metrics_to_plot = ["SSIM", "PSNR", "MSE", "LPIPS_Vgg", "LPIPS_Alex", "ReconstructionError"]
    for metric in metrics_to_plot:
        plt.figure()
        # plot benchmarked algorithm
        for model_name in results.keys():
            plt.plot(
                [result["n_iter"] for result in results[model_name]],
                [result[metric] for result in results[model_name]],
                label=model_name,
            )
        # plot baseline as horizontal dotted line
        if metric in baseline_results.keys():
            plt.hlines(
                baseline_results[metric],
                0,
                max(n_iter_range),
                linestyles="dashed",
                label="Unrolled MONAKHOVA 5iter",
                color="orange",
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
            first = False
            if plot_name not in algorithm_colors.keys():
                algorithm_colors[plot_name] = color_list.pop()
                first = True
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
                if first:
                    plt.plot(
                        unrolled_results[model_name]["n_iter"],
                        unrolled_results[model_name][metric],
                        label=plot_name,
                        marker="o",
                        color=color,
                    )
                else:
                    plt.plot(
                        unrolled_results[model_name]["n_iter"],
                        unrolled_results[model_name][metric],
                        marker="o",
                        color=color,
                    )
        plt.xlabel("Number of iterations", fontsize="12")
        plt.ylabel(metric, fontsize="12")
        if metric == "ReconstructionError":
            plt.legend(fontsize="12")
        plt.grid()
        plt.savefig(f"{metric}.png")


if __name__ == "__main__":
    benchmark_recon()
