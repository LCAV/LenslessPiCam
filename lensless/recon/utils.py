# #############################################################################
# utils.py
# =================
# Authors :
# Yohann PERRON [yohann.perron@gmail.com]
# Eric BEZZAM [ebezzam@gmail.com]
# #############################################################################


import json
import math
import time
from hydra.utils import get_original_cwd
import os
import matplotlib.pyplot as plt
import torch
from lensless.eval.benchmark import benchmark
from lensless.hardware.trainable_mask import TrainableMask
from tqdm import tqdm
from lensless.recon.drunet.network_unet import UNetRes


def load_drunet(model_path=None, n_channels=3, requires_grad=False):
    """
    Load a pre-trained Drunet model.

    Parameters
    ----------
    model_path : str, optional
        Path to pre-trained model. Download if not provided.
    n_channels : int
        Number of channels in input image.
    requires_grad : bool
        Whether to require gradients for model parameters.

    Returns
    -------
    model : :py:class:`torch.nn.Module`
        Loaded model.
    """

    if model_path is None:
        model_path = os.path.join(get_original_cwd(), "models", "drunet_color.pth")
        if not os.path.exists(model_path):
            try:
                from torchvision.datasets.utils import download_url
            except ImportError:
                exit()
            msg = "Do you want to download the pretrained DRUNet model (130MB)?"

            # default to yes if no input is given
            valid = input("%s (Y/n) " % msg).lower() != "n"
            output_path = os.path.join(get_original_cwd(), "models")
            if valid:
                url = "https://drive.switch.ch/index.php/s/jTdeMHom025RFRQ/download"
                filename = "drunet_color.pth"
                download_url(url, output_path, filename=filename)

    assert os.path.exists(model_path), f"Model path {model_path} does not exist"

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
    model : :py:class:`torch.nn.Module`
        Drunet compatible model. Its input must consist of 4 channels (RGB + noise level) and output an RGB image both in CHW format.
    image : :py:class:`torch.Tensor`
        Input image.
    noise_level : float or :py:class:`torch.Tensor`
        Noise level in the image.
    device : str
        Device to use for computation. Can be "cpu" or "cuda".
    mode : str
        Mode to use for model. Can be "inference" or "train".

    Returns
    -------
    image : :py:class:`torch.Tensor`
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
    model : :py:class:`torch.nn.Module`
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
    """
    Helper function to measure L2 norm of the gradient of a model.

    Parameters
    ----------
    model : :py:class:`torch.nn.Module`
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
    """
    Helper function to create a process network.

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
    :py:class:`torch.nn.Module`
        New process network. Already trained for Drunet.
    """
    if network == "DruNet":
        from lensless.recon.utils import load_drunet

        process = load_drunet(requires_grad=True).to(device)
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
        mask=None,
        batch_size=4,
        loss="l2",
        lpips=None,
        l1_mask=None,
        optimizer="Adam",
        optimizer_lr=1e-6,
        slow_start=None,
        skip_NAN=False,
        algorithm_name="Unknown",
    ):
        """
        Class to train a reconstruction algorithm. Inspired by Trainer from `HuggingFace <https://huggingface.co/docs/transformers/main_classes/trainer>`__.

        Parameters
        ----------
        recon : :py:class:`lensless.TrainableReconstructionAlgorithm`
            Reconstruction algorithm to train.
        train_dataset : :py:class:`torch.utils.data.Dataset`
            Dataset to use for training.
        test_dataset : :py:class:`torch.utils.data.Dataset`
            Dataset to use for testing.
        mask : TrainableMask, optional
            Trainable mask to use for training. If none, training with fix psf, by default None.
        batch_size : int, optional
            Batch size to use for training, by default 4
        loss : str, optional
            Loss function to use for training "l1" or "l2", by default "l2"
        lpips : float, optional
            the weight of the lpips(VGG) in the total loss. If None ignore. By default None
        l1_mask : float, optional
            the weight of the l1 norm of the mask in the total loss. If None ignore. By default None
        optimizer : str, optional
            Optimizer to use durring training. Available : "Adam". By default "Adam"
        optimizer_lr : float, optional
            Learning rate for the optimizer, by default 1e-6
        slow_start : float, optional
            Multiplicative factor to reduce the learning rate during the first two epochs. If None, ignored. Default is None.
        skip_NAN : bool, optional
            Whether to skip update if any gradiant are NAN (True) or to throw an error(False), by default False
        algorithm_name : str, optional
            Algorithm name for logging, by default "Unknown".

        """
        self.device = recon._psf.device

        self.recon = recon

        if test_dataset is None:
            # split train dataset
            train_size = int(0.9 * len(train_dataset))
            test_size = len(train_dataset) - train_size
            train_dataset, test_dataset = torch.utils.data.random_split(
                train_dataset, [train_size, test_size]
            )

        self.train_dataloader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=(self.device != "cpu"),
        )
        self.test_dataset = test_dataset
        self.lpips = lpips
        self.skip_NAN = skip_NAN

        if mask is not None:
            assert isinstance(mask, TrainableMask)
            self.mask = mask
            self.use_mask = True
        else:
            self.use_mask = False

        self.l1_mask = l1_mask

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
        """
        Train for one epoch.

        Parameters
        ----------
        data_loader : :py:class:`torch.utils.data.DataLoader`
            Data loader to use for training.
        disp : int
            Display interval, if -1, no display

        Returns
        -------
        float
            Mean loss of the epoch.
        """
        mean_loss = 0.0
        i = 1.0
        pbar = tqdm(data_loader)
        for X, y in pbar:
            # send to device
            X = X.to(self.device)
            y = y.to(self.device)

            # update psf according to mask
            if self.use_mask:
                self.recon._set_psf(self.mask.get_psf())

            # forward pass
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
            if self.use_mask and self.l1_mask:
                loss_v = loss_v + self.l1_mask * torch.mean(torch.abs(self.mask._mask))
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

            # update mask
            if self.use_mask:
                self.mask.update_mask()

            mean_loss += (loss_v.item() - mean_loss) * (1 / i)
            pbar.set_description(f"loss : {mean_loss}")
            i += 1

        return mean_loss

    def evaluate(self, mean_loss, save_pt):
        """
        Evaluate the reconstruction algorithm on the test dataset.

        Parameters
        ----------
        mean_loss : float
            Mean loss of the last epoch.
        save_pt : str
            Path to save metrics dictionary to. If None, no logging of metrics.
        """
        if self.test_dataset is None:
            return
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

    def on_epoch_end(self, mean_loss, save_pt):
        """
        Called at the end of each epoch.

        Parameters
        ----------
        mean_loss : float
            Mean loss of the last epoch.
        save_pt : str
            Path to save metrics dictionary to. If None, no logging of metrics.
        """
        if save_pt is None:
            # Use current directory
            save_pt = os.getcwd()

        # save model
        self.save(path=save_pt, include_optimizer=False)
        self.evaluate(mean_loss, save_pt)

    def train(self, n_epoch=1, save_pt=None, disp=-1):
        """
        Train the reconstruction algorithm.

        Parameters
        ----------
        n_epoch : int, optional
            Number of epochs to train for, by default 1
        save_pt : str, optional
            Path to save metrics dictionary to. If None, use current directory, by default None
        disp : int, optional
            Display interval, if -1, no display. Default is -1.
        """

        start_time = time.time()

        self.evaluate(-1, save_pt)
        for epoch in range(n_epoch):
            print(f"Epoch {epoch} with learning rate {self.scheduler.get_last_lr()}")
            mean_loss = self.train_epoch(self.train_dataloader, disp=disp)
            self.on_epoch_end(mean_loss, save_pt)
            self.scheduler.step()

        print(f"Train time : {time.time() - start_time} s")

    def save(self, path="recon", include_optimizer=False):
        # create directory if it does not exist
        if not os.path.exists(path):
            os.makedirs(path)
        # save mask
        if self.use_mask:
            torch.save(self.mask._mask, os.path.join(path, "mask.pt"))
            torch.save(self.mask._optimizer.state_dict(), os.path.join(path, "mask_optim.pt"))
            import matplotlib.pyplot as plt

            plt.imsave(
                os.path.join(path, "psf.png"), self.mask.get_psf().detach().cpu().numpy()[0, ...]
            )
        # save optimizer
        if include_optimizer:
            torch.save(self.optimizer.state_dict(), os.path.join(path, "optim.pt"))
        # save recon
        torch.save(self.recon.state_dict(), os.path.join(path, "recon.pt"))
