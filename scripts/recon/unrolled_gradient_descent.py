"""
Apply gradient descent.

```
python scripts/recon/gradient_descent.py
```

"""

import hydra
from hydra.utils import to_absolute_path, get_original_cwd
import os
import numpy as np
import time
import pathlib as plib
from datetime import datetime
import matplotlib.pyplot as plt
from lensless.io import load_data, load_psf
from lensless.unrolled_fista import unrolled_FISTA
from waveprop.dataset_util import SimulatedPytorchDataset
from lensless.util import rgb2gray
from lensless.benchmark import benchmark
import torch
from torchvision import transforms, datasets
from tqdm import tqdm


def simulate_dataset(config, psf):

    # load dataset
    transforms_list = [transforms.ToTensor()]
    data_path = os.path.join(get_original_cwd(), "data")
    if config.simulation.grayscale:
        transforms_list.append(transforms.Grayscale())
    transform = transforms.Compose(transforms_list)
    if config.files.dataset == "mnist":
        ds = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
    elif config.files.dataset == "fashion_mnist":
        ds = datasets.FashionMNIST(root=data_path, train=True, download=True, transform=transform)
    elif config.files.dataset == "cifar10":
        ds = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
    elif config.files.dataset == "CelebA":
        ds = datasets.CelebA(
            root="/scratch/bezzam/", split="train", download=True, transform=transform
        )
    else:
        raise NotImplementedError(f"Dataset {config.files.dataset} not implemented.")

    # convert PSF
    if config.simulation.grayscale:
        psf = rgb2gray(psf)
    if not isinstance(psf, torch.Tensor):
        psf = transforms.ToTensor()(psf)
    elif psf.shape[-1] == 3:
        # Waveprop syntetic dataset expect C H W
        psf = psf.permute(2, 0, 1)

    # batch_size = config.files.batch_size
    batch_size = config.training.batch_size
    n_files = config.files.n_files
    device_conv = config.torch_device
    target = config.target

    # check if gpu is available
    if device_conv == "cuda" and torch.cuda.is_available():
        device_conv = "cuda"
    else:
        device_conv = "cpu"

    # create Pytorch dataset and dataloader
    if n_files is not None:
        ds = torch.utils.data.Subset(ds, np.arange(n_files))
    ds_prop = SimulatedPytorchDataset(
        dataset=ds, psf=psf, device_conv=device_conv, target=target, **config.simulation
    )
    ds_loader = torch.utils.data.DataLoader(dataset=ds_prop, batch_size=batch_size, shuffle=True)
    return ds_loader


@hydra.main(version_base=None, config_path="../../configs", config_name="unrolled_recon")
def gradient_descent(
    config,
):
    if config.torch_device == "cuda" and torch.cuda.is_available():
        print("Using GPU for training.")
        device = "cuda"
    else:
        print("Using CPU for training.")
        device = "cpu"

    torch.autograd.set_detect_anomaly(True)

    # if using a portrait dataset
    flip = config.files.dataset in ["CelebA"]

    psf_float, background = load_psf(
        to_absolute_path(config.input.psf),
        downsample=config.simulation.downsample,
        return_float=True,
        return_bg=True,
        bg_pix=(0, 15),
        single_psf=config.preprocess.single_psf,
        shape=config.preprocess.shape,
        bayer=config.preprocess.bayer,
        blue_gain=config.preprocess.blue_gain,
        red_gain=config.preprocess.red_gain,
        flip=flip,
    )
    psf = torch.from_numpy(psf_float).to(device)

    # if using a portrait dataset
    if config.files.dataset in ["CelebA"]:
        psf = torch.rot90(psf, dims=[0, 1])

    disp = config.display.disp
    if disp < 0:
        disp = None

    save = config.save
    if save:
        save = os.getcwd()

    start_time = time.time()
    if config.gradient_descent.method == "unrolled_fista":
        recon = unrolled_FISTA(
            psf,
            n_iter=config.gradient_descent.n_iter,
            tk=config.gradient_descent.unrolled_fista.tk,
            pad=True,
            learn_tk=config.gradient_descent.unrolled_fista.learn_tk,
        ).to(device)
    else:
        raise ValueError(f"{config.gradient_descent.method} is not a supported algorithm")

    data_loader = simulate_dataset(config, psf)
    print(f"Setup time : {time.time() - start_time} s")

    start_time = time.time()

    # loss
    if config.loss == "l2":
        Loss = torch.nn.MSELoss()
    elif config.loss == "l1":
        Loss = torch.nn.L1Loss()
    else:
        raise ValueError(f"Unsuported loss : {config.loss}")

    # Lpips loss
    if config.lpips:
        try:
            import lpips

            loss_lpips = lpips.LPIPS(net="vgg").to(device)
        except ImportError:
            return ImportError(
                "lpips package is need for LPIPS loss. Install using : pip install lpips"
            )

    # optimizer
    if config.optimizer.type == "Adam":
        optimizer = torch.optim.Adam(recon.parameters(), lr=config.optimizer.lr)
    else:
        raise ValueError(f"Unsuported optimizer : {config.optimizer.type}")
    metrics = {"LOSS": [], "MSE": [], "MAE": [], "LPIPS": [], "PSNR": [], "SSIM": []}

    # Training loop
    for epoch in range(config.training.epoch):
        print(f"Epoch {epoch}")
        mean_loss = 0.0
        i = 1.0
        pbar = tqdm(data_loader)
        for X, y in pbar:
            y_pred = recon.batch_call(X.to(device))
            # normalizing each output
            y_pred_max = torch.amax(y_pred, dim=(1, 2, 3), keepdim=True)
            y_pred = y_pred / y_pred_max

            # normalizing y
            y = y.to(device)
            y_max = torch.amax(y, dim=(1, 2, 3), keepdim=True)
            y = y / y_max

            if i % disp == 1 and config.display.plot:
                img = y_pred[0].cpu().detach().permute(1, 2, 0).numpy()
                plt.imshow(img)
                plt.savefig(f"y_pred_{i-1}.png")
                img = y[0].cpu().detach().permute(1, 2, 0).numpy()
                plt.imshow(img)
                plt.savefig(f"y_{i-1}.png")

            optimizer.zero_grad(set_to_none=True)
            loss_v = Loss(y_pred, y)
            if config.lpips:
                loss_v = loss_v + config.lpips * torch.mean(loss_lpips(y_pred, y))
            loss_v.backward()
            optimizer.step()

            mean_loss += (loss_v.item() - mean_loss) * (1 / i)

            pbar.set_description(f"loss : {mean_loss}")
            i += 1

        # benchmarking
        data_path = "data/DiffuserCam_Mirflickr_200_3011302021_11h43_seed11"
        data_path = os.path.join(get_original_cwd(), data_path)
        current_metrics = benchmark(
            recon, data_path, downsample=config.simulation.downsample, flip=flip, batchsize=10
        )
        # update metrics with current metrics
        metrics["LOSS"].append(mean_loss)
        for key in current_metrics:
            metrics[key].append(current_metrics[key])

    print(f"Train time : {time.time() - start_time} s")

    # save dictionary metrics to file with json
    try:
        import json

        with open(os.path.join(save, "metrics.json"), "w") as f:
            json.dump(metrics, f)
    except ImportError:
        print("json package not found, metrics not saved")

    # save pytorch model recon
    torch.save(recon.state_dict(), "recon.pt")


if __name__ == "__main__":
    gradient_descent()
