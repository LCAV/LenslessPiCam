# #############################################################################
# train_unrolled.py
# =================
# Authors :
# Yohann PERRON [yohann.perron@gmail.com]
# #############################################################################

"""
Train unrolled version of reconstruction algorithm.

```
python scripts/recon/train_unrolled.py
```

"""

import hydra
from hydra.utils import get_original_cwd
import os
import numpy as np
import time
import matplotlib.pyplot as plt
from lensless import UnrolledFISTA, UnrolledADMM
from waveprop.dataset_util import SimulatedPytorchDataset
from lensless.utils.image import rgb2gray
from lensless.eval.benchmark import benchmark, DiffuserCamTestDataset
import torch
from torchvision import transforms, datasets
from tqdm import tqdm

try:
    import json
except ImportError:
    print("json package not found, metrics will not be saved")


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
        ds = datasets.CelebA(root=data_path, split="train", download=True, transform=transform)
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
    ds_loader = torch.utils.data.DataLoader(
        dataset=ds_prop, batch_size=batch_size, shuffle=True, pin_memory=(psf.device != "cpu")
    )
    return ds_loader


def measure_gradient(model):
    # return the L2 norm of the gradient
    total_norm = 0.0
    for p in model.parameters():
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm


@hydra.main(version_base=None, config_path="../../configs", config_name="unrolled_recon")
def train_unrolled(
    config,
):
    if config.torch_device == "cuda" and torch.cuda.is_available():
        print("Using GPU for training.")
        device = "cuda"
    else:
        print("Using CPU for training.")
        device = "cpu"

    # torch.autograd.set_detect_anomaly(True)

    # if using a portrait dataset rotate the PSF
    flip = config.files.dataset in ["CelebA"]

    # benchmarking dataset:
    path = os.path.join(get_original_cwd(), "data")
    benchmark_dataset = DiffuserCamTestDataset(data_dir=path)

    psf = benchmark_dataset.psf.to(device)
    background = benchmark_dataset.background

    # convert psf from BGR to RGB
    if config.files.dataset in ["DiffuserCam"]:
        psf = psf[..., [2, 1, 0]]

    # if using a portrait dataset rotate the PSF
    if flip:
        psf = torch.rot90(psf, dims=[0, 1])

    disp = config.display.disp
    if disp < 0:
        disp = None

    save = config.save
    if save:
        save = os.getcwd()

    start_time = time.time()
    # Load post process model
    if config.reconstruction.post_process.network == "DruNet":
        from lensless.util import load_drunet

        post_process_model = load_drunet(
            os.path.join(get_original_cwd(), "data/drunet_color.pth"), requires_grad=True
        ).to(device)
        post_process = True
    elif config.reconstruction.post_process.network == "UnetRes":
        from lensless.drunet.network_unet import UNetRes

        n_channels = 3
        post_process_model = UNetRes(
            in_nc=n_channels + 1,
            out_nc=n_channels,
            nc=[64, 128, 256, 512],
            nb=config.reconstruction.post_process.depth,
            act_mode="R",
            downsample_mode="strideconv",
            upsample_mode="convtranspose",
        )
        post_process = True

    # convert model to function
    if "post_process_model" in locals():
        from lensless.util import process_with_DruNet

        post_process = process_with_DruNet(post_process_model, device=device, mode="train")
    else:
        post_process = None
    pre_process = None

    if config.reconstruction.method == "unrolled_fista":
        recon = UnrolledFISTA(
            psf,
            n_iter=config.reconstruction.unrolled_fista.n_iter,
            tk=config.reconstruction.unrolled_fista.tk,
            pad=True,
            learn_tk=config.reconstruction.unrolled_fista.learn_tk,
            pre_process=pre_process,
            post_process=post_process,
        ).to(device)
        n_iter = config.reconstruction.unrolled_fista.n_iter
    elif config.reconstruction.method == "unrolled_admm":
        recon = UnrolledADMM(
            psf,
            n_iter=config.reconstruction.unrolled_admm.n_iter,
            mu1=config.reconstruction.unrolled_admm.mu1,
            mu2=config.reconstruction.unrolled_admm.mu2,
            mu3=config.reconstruction.unrolled_admm.mu3,
            tau=config.reconstruction.unrolled_admm.tau,
            pre_process=pre_process,
            post_process=post_process,
        ).to(device)
        n_iter = config.reconstruction.unrolled_admm.n_iter
    else:
        raise ValueError(f"{config.reconstruction.method} is not a supported algorithm")

    # print number of parameters
    print(f"Training model with {sum(p.numel() for p in recon.parameters())} parameters")
    if "post_process_model" in locals():
        print(
            f"Post processing model with {sum(p.numel() for p in post_process_model.parameters())} parameters"
        )
    # transform from BGR to RGB
    transform_BRG2RGB = transforms.Lambda(lambda x: x[..., [2, 1, 0]])

    # load dataset and create dataloader
    if config.files.dataset == "DiffuserCam":
        # Use a ParallelDataset
        from lensless.eval.benchmark import ParallelDataset

        data_path = os.path.join(get_original_cwd(), "data", "DiffuserCam")
        dataset = ParallelDataset(
            root_dir=data_path,
            n_files=config.files.n_files,
            background=background,
            psf=psf,
            lensless_fn="diffuser_images",
            lensed_fn="ground_truth_lensed",
            downsample=config.simulation.downsample,
            transform_lensless=transform_BRG2RGB,
            transform_lensed=transform_BRG2RGB,
        )
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            pin_memory=(device != "cpu"),
        )
    else:
        # Use a simulated dataset
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
        # the parameters of the base model and extra porcess must be added separatly
        parameters = [{"params": recon.parameters()}]
        if "post_process_model" in locals():
            parameters.append({"params": post_process_model.parameters()})
        optimizer = torch.optim.Adam(parameters, lr=config.optimizer.lr)
    else:
        raise ValueError(f"Unsuported optimizer : {config.optimizer.type}")
    algorithm = config.reconstruction.method
    if config.reconstruction.post_process.network == "DruNet":
        algorithm += "_DruNet"
    elif config.reconstruction.post_process.network == "UnetRes":
        algorithm += "_UnetRes"
    metrics = {
        "LOSS": [],
        "MSE": [],
        "MAE": [],
        "LPIPS_Vgg": [],
        "LPIPS_Alex": [],
        "PSNR": [],
        "SSIM": [],
        "ReconstructionError": [],
        "n_iter": n_iter,
        "algorithm": algorithm,
    }

    # Backward hook that detect NAN in the gradient and print the layer weights
    def detect_nan(grad):
        if torch.isnan(grad).any():
            print(grad, flush=True)
            for name, param in recon.named_parameters():
                if param.requires_grad:
                    print(name, param)
            raise ValueError("Gradient is NaN")
        return grad

    for param in recon.parameters():
        if param.requires_grad:
            param.register_hook(detect_nan)
    if "post_process_model" in locals():
        for param in post_process_model.parameters():
            if param.requires_grad:
                param.register_hook(detect_nan)

    # Training loop
    for epoch in range(config.training.epoch):
        print(f"Epoch {epoch}")
        mean_loss = 0.0
        i = 1.0
        pbar = tqdm(data_loader)
        for X, y in pbar:
            # send to device
            X = X.to(device)
            y = y.to(device)
            if X.shape[3] == 3:
                X = X
                y = y

            y_pred = recon.batch_call(X.to(device))
            # normalizing each output
            y_pred_max = torch.amax(y_pred, dim=(-1, -2, -3), keepdim=True)
            y_pred = y_pred / y_pred_max

            # normalizing y
            y = y.to(device)
            y_max = torch.amax(y, dim=(-1, -2, -3), keepdim=True)
            y = y / y_max

            if i % disp == 1 and config.display.plot:
                img_pred = y_pred[0, 0].cpu().detach().numpy()
                img_truth = y[0, 0].cpu().detach().numpy()

                plt.imshow(img_pred)
                plt.savefig(f"y_pred_{i-1}.png")
                plt.imshow(img_truth)
                plt.savefig(f"y_{i-1}.png")

            optimizer.zero_grad(set_to_none=True)
            # convert to CHW for loss and remove depth
            y_pred = y_pred.reshape(-1, *y_pred.shape[-3:]).movedim(-1, -3)
            y = y.reshape(-1, *y.shape[-3:]).movedim(-1, -3)

            loss_v = Loss(y_pred, y)
            if config.lpips:
                # value for LPIPS needs to be in range [-1, 1]
                loss_v = loss_v + config.lpips * torch.mean(loss_lpips(2 * y_pred - 1, 2 * y - 1))
            loss_v.backward()
            torch.nn.utils.clip_grad_norm_(recon.parameters(), 1.0)
            if "post_process_model" in locals():
                torch.nn.utils.clip_grad_norm_(post_process_model.parameters(), 1.0)
            optimizer.step()

            mean_loss += (loss_v.item() - mean_loss) * (1 / i)
            pbar.set_description(f"loss : {mean_loss}")
            i += 1

        # benchmarking
        current_metrics = benchmark(recon, benchmark_dataset, batchsize=10)
        # update metrics with current metrics
        metrics["LOSS"].append(mean_loss)
        for key in current_metrics:
            metrics[key].append(current_metrics[key])

    print(f"Train time : {time.time() - start_time} s")

    # save dictionary metrics to file with json
    with open(os.path.join(save, "metrics.json"), "w") as f:
        json.dump(metrics, f)

    # save pytorch model recon
    torch.save(recon.state_dict(), "recon.pt")


if __name__ == "__main__":
    train_unrolled()
