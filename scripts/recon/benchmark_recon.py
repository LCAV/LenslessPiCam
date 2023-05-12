import hydra
from hydra.utils import to_absolute_path, get_original_cwd

import glob
import json
import os
import pathlib as plib
from lensless.benchmark import benchmark, BenchmarkDataset
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

    model_list = []  # list of algoritms to benchmark
    if "ADMM" in config.algorithms:
        model_list.append(ADMM)
    if "FISTA" in config.algorithms:
        model_list.append(FISTA)
    if "GradientDescent" in config.algorithms:
        model_list.append(GradientDescent)
    if "NesterovGradientDescent" in config.algorithms:
        model_list.append(NesterovGradientDescent)
    if "APGD" in config.algorithms:
        from lensless import APGD

        model_list.append(APGD)

    # check if GPU is available
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Benchmark dataset
    benchmark_dataset = BenchmarkDataset(n_files=n_files, downsample=downsample)
    psf = benchmark_dataset.psf.to(device)

    results = {}
    # benchmark each model for different number of iteration and append result to results
    for Model in model_list:
        results[Model.__name__] = []
        print(f"Running benchmark for {Model.__name__}")
        for n_iter in n_iter_range:
            model = Model(psf, n_iter=n_iter)
            result = benchmark(model, benchmark_dataset, batchsize=1)
            result["n_iter"] = n_iter
            results[Model.__name__].append(result)

    # create folder to save plots
    if not os.path.isdir("benchmark"):
        os.mkdir("benchmark")

    # try to load json files with results form unrolled training
    files = glob.glob(os.path.join("benchmark", "*.json"))
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
        plt.savefig(os.path.join("benchmark", f"{metric}.png"))


if __name__ == "__main__":
    benchmark_recon()
