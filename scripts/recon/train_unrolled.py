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

import math
import hydra
from hydra.utils import get_original_cwd
import os
import numpy as np
import time
import matplotlib.pyplot as plt
from lensless import UnrolledFISTA, UnrolledADMM
from lensless.utils.dataset import DiffuserCamTestDataset, SimulatedDataset
from lensless.utils.image import rgb2gray
from lensless.eval.benchmark import benchmark
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

    # batch_size = config.files.batch_size
    batch_size = config.training.batch_size
    n_files = config.files.n_files
    device_conv = config.torch_device

    # check if gpu is available
    if device_conv == "cuda" and torch.cuda.is_available():
        device_conv = "cuda"
    else:
        device_conv = "cpu"

    # create Pytorch dataset and dataloader
    if n_files is not None:
        ds = torch.utils.data.Subset(ds, np.arange(n_files))
    ds_prop = SimulatedDataset(
        dataset=ds, psf=psf, dataset_is_CHW=True, device_conv=device_conv, **config.simulation
    )
    ds_loader = torch.utils.data.DataLoader(
        dataset=ds_prop, batch_size=batch_size, shuffle=True, pin_memory=(psf.device != "cpu")
    )
    return ds_loader


def create_process_network(network, depth, device="cpu"):
    if network == "DruNet":
        from lensless.recon.utils import load_drunet

        process = load_drunet(
            os.path.join(get_original_cwd(), "data/drunet_color.pth"), requires_grad=True
        ).to(device)
        process_name = "DruNet"
    elif network == "UnetRes":
        from lensless.recon.drunet.network_unet import UNetRes

        n_channels = 3
        process = UNetRes(
            in_nc=n_channels + 1,
            out_nc=n_channels,
            nc=[64, 128, 256, 512],
            nb=depth,
            act_mode="R",
            downsample_mode="strideconv",
            upsample_mode="convtranspose",
        ).to(device)
        process_name = "UnetRes_d" + str(depth)
    else:
        process = None
        process_name = None

    return (process, process_name)


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

    # benchmarking dataset:
    path = os.path.join(get_original_cwd(), "data")
    benchmark_dataset = DiffuserCamTestDataset(
        data_dir=path, downsample=config.simulation.downsample
    )

    psf = benchmark_dataset.psf.to(device)
    background = benchmark_dataset.background

    # convert psf from BGR to RGB
    if config.files.dataset in ["DiffuserCam"]:
        psf = psf[..., [2, 1, 0]]

    # if using a portrait dataset rotate the PSF

    disp = config.display.disp
    if disp < 0:
        disp = None

    save = config.save
    if save:
        save = os.getcwd()

    start_time = time.time()

    # Load pre process model
    pre_process, pre_process_name = create_process_network(
        config.reconstruction.pre_process.network,
        config.reconstruction.pre_process.depth,
        device=device,
    )
    # Load post process model
    post_process, post_process_name = create_process_network(
        config.reconstruction.post_process.network,
        config.reconstruction.post_process.depth,
        device=device,
    )
    # create reconstruction algorithm
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

    # constructing algorithm name by appending pre and post process
    algorithm_name = config.reconstruction.method
    if config.reconstruction.pre_process.network is not None:
        algorithm_name = pre_process_name + "_" + algorithm_name
    if config.reconstruction.post_process.network is not None:
        algorithm_name += "_" + post_process_name

    # print number of parameters
    print(f"Training model with {sum(p.numel() for p in recon.parameters())} parameters")
    # transform from BGR to RGB
    transform_BRG2RGB = transforms.Lambda(lambda x: x[..., [2, 1, 0]])

    # load dataset and create dataloader
    if config.files.dataset == "DiffuserCam":
        # Use a ParallelDataset
        from lensless.utils.dataset import ParallelDataset

        max_indices = 30000
        if config.files.n_files is not None:
            max_indices = config.files.n_files + 1000

        data_path = os.path.join(get_original_cwd(), "data", "DiffuserCam")
        dataset = ParallelDataset(
            root_dir=data_path,
            indices=range(1000, max_indices),
            background=background,
            psf=psf,
            lensless_fn="diffuser_images",
            lensed_fn="ground_truth_lensed",
            downsample=config.simulation.downsample / 4,
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
        # the parameters of the base model and non torch.Module process must be added separatly
        parameters = [{"params": recon.parameters()}]
        optimizer = torch.optim.Adam(parameters, lr=config.optimizer.lr)
    else:
        raise ValueError(f"Unsuported optimizer : {config.optimizer.type}")
    # Scheduler
    if config.training.slow_start:

        def learning_rate_function(epoch):
            if epoch == 0:
                return config.training.slow_start
            elif epoch == 1:
                return math.sqrt(config.training.slow_start)
            else:
                return 1

    else:

        def learning_rate_function(epoch):
            return 1

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=learning_rate_function)

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
        "algorithm": algorithm_name,
    }

    # Backward hook that detect NAN in the gradient and print the layer weights
    if not config.training.skip_NAN:

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
                if param.requires_grad:
                    param.register_hook(detect_nan)

    # Training loop
    for epoch in range(config.training.epoch):
        print(f"Epoch {epoch} with learning rate {scheduler.get_last_lr()}")
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
            eps = 1e-12
            y_pred_max = torch.amax(y_pred, dim=(-1, -2, -3), keepdim=True) + eps
            y_pred = y_pred / y_pred_max

            # normalizing y
            y = y.to(device)
            y_max = torch.amax(y, dim=(-1, -2, -3), keepdim=True) + eps
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

            # if any gradient is NaN, skip training step
            is_NAN = False
            for param in recon.parameters():
                if torch.isnan(param.grad).any():
                    is_NAN = True
                    break
            if is_NAN:
                print("NAN detected in gradiant, skipping training step")
                i += 1
                continue
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

        # Update learning rate
        scheduler.step()

    print(f"Train time : {time.time() - start_time} s")

    # save dictionary metrics to file with json
    with open(os.path.join(save, "metrics.json"), "w") as f:
        json.dump(metrics, f)

    # save pytorch model recon
    torch.save(recon.state_dict(), "recon.pt")


if __name__ == "__main__":
    train_unrolled()
