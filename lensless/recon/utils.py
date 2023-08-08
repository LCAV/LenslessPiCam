import json
import math
import time
from hydra.utils import get_original_cwd
import os
import matplotlib.pyplot as plt
import torch
from lensless.eval.benchmark import benchmark
from tqdm import tqdm
from lensless.recon.drunet.network_unet import UNetRes


def load_drunet(model_path, n_channels=3, requires_grad=False):
    """
    Load a pre-trained Drunet model.

    Parameters
    ----------
    model_path : str
        Path to pre-trained model.
    n_channels : int
        Number of channels in input image.
    requires_grad : bool
        Whether to require gradients for model parameters.

    Returns
    -------
    model : :py:class:`~torch.nn.Module`
        Loaded model.
    """

    model = UNetRes(
        in_nc=n_channels + 1,
        out_nc=n_channels,
        nc=[64, 128, 256, 512],
        nb=4,
        act_mode="R",
        downsample_mode="strideconv",
        upsample_mode="convtranspose",
    )
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for _, v in model.named_parameters():
        v.requires_grad = requires_grad

    return model


def apply_denoiser(model, image, noise_level=10, device="cpu", mode="inference"):
    """
    Apply a pre-trained denoising model with input in the format Channel, Height, Width.
    An additionnal channel is added for the noise level as done in Drunet.

    Parameters
    ----------
    model : :py:class:`~torch.nn.Module`
        Drunet compatible model. Its input must consist of 4 channels (RGB + noise level) and output an RGB image both in CHW format.
    image : :py:class:`~torch.Tensor`
        Input image.
    noise_level : float or :py:class:`~torch.Tensor`
        Noise level in the image.
    device : str
        Device to use for computation. Can be "cpu" or "cuda".
    mode : str
        Mode to use for model. Can be "inference" or "train".

    Returns
    -------
    image : :py:class:`~torch.Tensor`
        Reconstructed image.
    """
    # convert from NDHWC to NCHW
    depth = image.shape[-4]
    image = image.movedim(-1, -3)
    image = image.reshape(-1, *image.shape[-3:])
    # pad image H and W to next multiple of 8
    top = (8 - image.shape[-2] % 8) // 2
    bottom = (8 - image.shape[-2] % 8) - top
    left = (8 - image.shape[-1] % 8) // 2
    right = (8 - image.shape[-1] % 8) - left
    image = torch.nn.functional.pad(image, (left, right, top, bottom), mode="constant", value=0)
    # add noise level as extra channel
    image = image.to(device)
    if isinstance(noise_level, torch.Tensor):
        noise_level = noise_level / 255.0
    else:
        noise_level = torch.tensor([noise_level / 255.0]).to(device)
    image = torch.cat(
        (
            image,
            noise_level.repeat(image.shape[0], 1, image.shape[2], image.shape[3]),
        ),
        dim=1,
    )

    # apply model
    if mode == "inference":
        with torch.no_grad():
            image = model(image)
    elif mode == "train":
        image = model(image)
    else:
        raise ValueError("mode must be 'inference' or 'train'")

    # remove padding
    image = image[:, :, top:-bottom, left:-right]
    # convert back to NDHWC
    image = image.movedim(-3, -1)
    image = image.reshape(-1, depth, *image.shape[-3:])
    return image


def get_drunet_function(model, device="cpu", mode="inference"):
    """
    Return a porcessing function that applies the DruNet model to an image.

    Parameters
    ----------
    model : torch.nn.Module
        DruNet like denoiser model
    device : str
        Device to use for computation. Can be "cpu" or "cuda".
    mode : str
        Mode to use for model. Can be "inference" or "train".
    """

    def process(image, noise_level):
        x_max = torch.amax(image, dim=(-2, -3), keepdim=True) + 1e-6
        image = apply_denoiser(
            model,
            image,
            noise_level=noise_level,
            device=device,
            mode=mode,
        )
        image = torch.clip(image, min=0.0) * x_max
        return image

    return process


def measure_gradient(model):
    """Helper function to measure L2 norm of the gradient of a model.

    Parameters
    ----------
    model : ''torch.nn.Module''
        Model to measure gradient of.

    Returns
    -------
    Float
        L2 norm of the gradient of the model.
    """
    total_norm = 0.0
    for p in model.parameters():
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm


def create_process_network(network, depth, device="cpu"):
    """Helper function to create a process network.

    Parameters
    ----------
    network : str
        Name of network to use. Can be "DruNet" or "UnetRes".
    depth : int
        Depth of network.
    device : str
        Device to use for computation. Can be "cpu" or "cuda". Defaults to "cpu".

    Returns
    -------
    torch.nn.Module
        New process network. Already trained for Drunet."""
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


class Trainer:
    def __init__(
        self,
        recon,
        train_dataset,
        test_dataset,
        batch_size=4,
        loss="l2",
        lpips=False,
        optimizer="Adam",
        optimizer_lr=1e-6,
        slow_start=None,
        skip_NAN=False,
        algorithm_name="Unknow",
    ):
        self.device = recon._psf.device

        self.recon = recon
        self.train_dataloader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=(self.device != "cpu"),
        )
        self.test_dataset = test_dataset
        self.lpips = lpips
        self.skip_NAN = skip_NAN

        # loss
        if loss == "l2":
            self.Loss = torch.nn.MSELoss()
        elif loss == "l1":
            self.Loss = torch.nn.L1Loss()
        else:
            raise ValueError(f"Unsuported loss : {loss}")

        # Lpips loss
        if lpips:
            try:
                import lpips

                self.Loss_lpips = lpips.LPIPS(net="vgg").to(self.device)
            except ImportError:
                return ImportError(
                    "lpips package is need for LPIPS loss. Install using : pip install lpips"
                )

        # optimizer
        if optimizer == "Adam":
            # the parameters of the base model and non torch.Module process must be added separatly
            parameters = [{"params": recon.parameters()}]
            self.optimizer = torch.optim.Adam(parameters, lr=optimizer_lr)
        else:
            raise ValueError(f"Unsuported optimizer : {optimizer}")
        # Scheduler
        if slow_start:

            def learning_rate_function(epoch):
                if epoch == 0:
                    return slow_start
                elif epoch == 1:
                    return math.sqrt(slow_start)
                else:
                    return 1

        else:

            def learning_rate_function(epoch):
                return 1

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=learning_rate_function
        )

        self.metrics = {
            "LOSS": [],
            "MSE": [],
            "MAE": [],
            "LPIPS_Vgg": [],
            "LPIPS_Alex": [],
            "PSNR": [],
            "SSIM": [],
            "ReconstructionError": [],
            "n_iter": self.recon._n_iter,
            "algorithm": algorithm_name,
        }

        # Backward hook that detect NAN in the gradient and print the layer weights
        if not self.skip_NAN:

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

    def train_epoch(self, data_loader, disp=-1):
        mean_loss = 0.0
        i = 1.0
        pbar = tqdm(data_loader)
        for X, y in pbar:
            # send to device
            X = X.to(self.device)
            y = y.to(self.device)
            if X.shape[3] == 3:
                X = X
                y = y

            y_pred = self.recon.batch_call(X.to(self.device))
            # normalizing each output
            eps = 1e-12
            y_pred_max = torch.amax(y_pred, dim=(-1, -2, -3), keepdim=True) + eps
            y_pred = y_pred / y_pred_max

            # normalizing y
            y_max = torch.amax(y, dim=(-1, -2, -3), keepdim=True) + eps
            y = y / y_max

            if i % disp == 1:
                img_pred = y_pred[0, 0].cpu().detach().numpy()
                img_truth = y[0, 0].cpu().detach().numpy()

                plt.imshow(img_pred)
                plt.savefig(f"y_pred_{i-1}.png")
                plt.imshow(img_truth)
                plt.savefig(f"y_{i-1}.png")

            self.optimizer.zero_grad(set_to_none=True)
            # convert to CHW for loss and remove depth
            y_pred = y_pred.reshape(-1, *y_pred.shape[-3:]).movedim(-1, -3)
            y = y.reshape(-1, *y.shape[-3:]).movedim(-1, -3)

            loss_v = self.Loss(y_pred, y)
            if self.lpips:
                # value for LPIPS needs to be in range [-1, 1]
                loss_v = loss_v + self.lpips * torch.mean(
                    self.Loss_lpips(2 * y_pred - 1, 2 * y - 1)
                )
            loss_v.backward()

            torch.nn.utils.clip_grad_norm_(self.recon.parameters(), 1.0)

            # if any gradient is NaN, skip training step
            if self.skip_NAN:
                is_NAN = False
                for param in self.recon.parameters():
                    if torch.isnan(param.grad).any():
                        is_NAN = True
                        break
                if is_NAN:
                    print("NAN detected in gradiant, skipping training step")
                    i += 1
                    continue
            self.optimizer.step()

            mean_loss += (loss_v.item() - mean_loss) * (1 / i)
            pbar.set_description(f"loss : {mean_loss}")
            i += 1

        return mean_loss

    def evaluate(self, mean_loss, save_pt):
        # benchmarking
        current_metrics = benchmark(self.recon, self.test_dataset, batchsize=10)

        # update metrics with current metrics
        self.metrics["LOSS"].append(mean_loss)
        for key in current_metrics:
            self.metrics[key].append(current_metrics[key])

        if save_pt:
            # save dictionary metrics to file with json
            with open(os.path.join(save_pt, "metrics.json"), "w") as f:
                json.dump(self.metrics, f)

    def train(self, n_epoch=1, save_pt=False):
        start_time = time.time()

        for epoch in range(n_epoch):
            print(f"Epoch {epoch} with learning rate {self.scheduler.get_last_lr()}")
            mean_loss = self.train_epoch(self.train_dataloader)
            self.evaluate(mean_loss, save_pt)
            self.scheduler.step()

        print(f"Train time : {time.time() - start_time} s")

    def save(self, path="recon.pt"):
        torch.save(self.recon.state_dict(), path)
