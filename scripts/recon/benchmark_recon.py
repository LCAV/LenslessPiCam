import glob
import json
import os
import pathlib as plib
from datetime import datetime
from lensless.io import load_psf
from lensless.benchmark import benchmark
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from lensless import ADMM, FISTA, GradientDescent

try:
    import torch
except ImportError:
    raise ImportError("Torch and torchmetrics are needed to benchmark reconstruction algorithm")


if __name__ == "__main__":

    downsample = 4

    # check if GPU is available
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    data = "data/DiffuserCam_Mirflickr_200_3011302021_11h43_seed11"
    if not os.path.isdir(data):
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

    psf_fp = os.path.join(data, "psf.tiff")
    psf_float, background = load_psf(
        psf_fp,
        downsample=downsample,
        return_float=True,
        return_bg=True,
        bg_pix=(0, 15),
    )
    psf = torch.from_numpy(psf_float).to(device)
    results = {}
    n_iter_range = [5, 10, 30, 100, 300]
    # benchmark each model for different number of iteration and append result to results
    for Model in [ADMM, FISTA, GradientDescent]:
        results[Model.__name__] = []
        print(f"Running benchmark for {Model.__name__}")
        for n_iter in n_iter_range:
            model = Model(psf)
            result = benchmark(model, data, n_files=100, downsample=downsample, n_iter=n_iter)
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
            # get most recent result
            for metric in result.keys():
                unrolled_results[model_name][metric] = result[metric][-1]

    # for each metrics plot the results comparing each model
    metrics_to_plot = ["SSIM", "PSNR", "MSE", "LPIPS"]
    for metric in metrics_to_plot:
        plt.figure()
        # plot benchmark algorithm
        for model_name in results.keys():
            plt.plot(
                [result["n_iter"] for result in results[model_name]],
                [result[metric] for result in results[model_name]],
                label=model_name,
            )
        # plot unrolled algorithms results as horizontal line with different color
        colors = ["red", "green", "blue", "orange", "purple"]
        for model_name in unrolled_results.keys():
            plt.hlines(
                unrolled_results[model_name][metric],
                0,
                n_iter_range[-1],
                label=model_name,
                linestyles="dashed",
                colors=colors.pop(),
            )
        plt.title(metric)
        plt.xlabel("Number of iterations")
        plt.ylabel(metric)
        plt.legend()
        plt.savefig(os.path.join("benchmark", f"{metric}.png"))
