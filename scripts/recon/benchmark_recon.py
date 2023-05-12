import glob
import json
import os
import pathlib as plib
from datetime import datetime
from lensless.io import load_psf
from lensless.benchmark import benchmark, ParallelDataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from lensless import ADMM, FISTA, GradientDescent

try:
    import torch
except ImportError:
    raise ImportError("Torch and torchmetrics are needed to benchmark reconstruction algorithm")


if __name__ == "__main__":

    downsample = 8
    data_path = "data/DiffuserCam_Mirflickr_200_3011302021_11h43_seed11"
    n_iter_range = [5, 10, 30, 60, 100, 200, 300]  # numbers of iterations to benchmark
    model_list = [ADMM, FISTA, GradientDescent]  # list of models to benchmark

    # check if GPU is available
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # verify if dataset exist and download if necessary
    if not os.path.isdir(data_path):
        print("No dataset found for benchmarking.")
        try:
            from torchvision.datasets.utils import download_and_extract_archive
        except ImportError:
            exit()
        msg = "Do you want to download the sample dataset (725MB)?"

        # default to yes if no input is given
        valid = input("%s (Y/n) " % msg).lower() != "n"
        if valid:
            url = "https://drive.switch.ch/index.php/s/vmAZzryGI8U8rcE/download"
            filename = "DiffuserCam_Mirflickr_200_3011302021_11h43_seed11.zip"
            download_and_extract_archive(url, "data/", filename=filename, remove_finished=True)

    # load psf and compute background
    psf_fp = os.path.join(data_path, "psf.tiff")
    psf_float, background = load_psf(
        psf_fp,
        downsample=downsample,
        return_float=True,
        return_bg=True,
        bg_pix=(0, 15),
    )
    psf = torch.from_numpy(psf_float).to(device)

    # Benchmark dataset
    benchmark_dataset = ParallelDataset(
        data_path,
        n_files=200,
        background=background,
        downsample=downsample,
    )

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

    # # benchmark FISTA model for different number of iteration and tk, and append result to results
    # for tk in [0.1, 0.5, 1, 2, 5]:
    #     results[f"FISTA_{tk}"] = []
    #     print(f"Running benchmark for FISTA with tk={tk}")
    #     for n_iter in n_iter_range:
    #         model = FISTA(psf, tk=tk)
    #         result = benchmark(model, data, n_files=100, downsample=downsample, n_iter=n_iter)
    #         result["n_iter"] = n_iter
    #         results[f"FISTA_{tk}"].append(result)

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
    metrics_to_plot = ["SSIM", "PSNR", "MSE", "LPIPS"]
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
