# #############################################################################
# train_unrolled.py
# =================
# Authors :
# Yohann PERRON [yohann.perron@gmail.com]
# Eric BEZZAM [ebezzam@gmail.com]
# #############################################################################

"""
Train unrolled version of reconstruction algorithm.

```
python scripts/recon/train_unrolled.py
```

By default it uses the configuration fro mthe file `configs/train_unrolledADMM.yaml`.

To train pre- and post-processing networks, use the following command:
```
python scripts/recon/train_unrolled.py -cn train_pre-post-processing
```

To fine-tune the DiffuserCam PSF, use the following command:
```
python scripts/recon/train_unrolled.py -cn fine-tune_PSF
```

"""

import logging
import hydra
from hydra.utils import get_original_cwd
import os
import numpy as np
import time
from lensless import UnrolledFISTA, UnrolledADMM
from lensless.utils.dataset import (
    DiffuserCamMirflickr,
    SimulatedFarFieldDataset,
    SimulatedDatasetTrainableMask,
)
from torch.utils.data import Subset
import lensless.hardware.trainable_mask
from lensless.recon.utils import create_process_network
from lensless.utils.image import rgb2gray
from lensless.utils.simulation import FarFieldSimulator
from lensless.recon.utils import Trainer
import torch
from torchvision import transforms, datasets
from lensless.utils.io import load_psf
from lensless.utils.io import save_image
from lensless.utils.plot import plot_image
import matplotlib.pyplot as plt

# A logger for this file
log = logging.getLogger(__name__)


def simulate_dataset(config):

    # prepare PSF
    psf_fp = os.path.join(get_original_cwd(), config.files.psf)
    psf, _ = load_psf(
        psf_fp,
        downsample=config.files.downsample,
        return_float=True,
        return_bg=True,
        bg_pix=(0, 15),
    )
    if config.files.diffusercam_psf:
        transform_BRG2RGB = transforms.Lambda(lambda x: x[..., [2, 1, 0]])
        psf = transform_BRG2RGB(torch.from_numpy(psf))

    # prepare mask
    if config.trainable_mask.mask_type is not None:
        mask_class = getattr(lensless.hardware.trainable_mask, config.trainable_mask.mask_type)
        if config.trainable_mask.initial_value == "random":
            mask = mask_class(
                torch.rand_like(psf), optimizer="Adam", lr=config.trainable_mask.mask_lr
            )
        # TODO : change to PSF
        elif config.trainable_mask.initial_value == "DiffuserCam":
            mask = mask_class(psf, optimizer="Adam", lr=config.trainable_mask.mask_lr)
        elif config.trainable_mask.initial_value == "DiffuserCam_gray":
            # TODO convert to grayscale
            mask = mask_class(
                psf[:, :, :, 0, None],
                optimizer="Adam",
                lr=config.trainable_mask.mask_lr,
                is_rgb=not config.simulation.grayscale,
            )
    else:
        mask = None

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

    n_files = config.files.n_files
    device_conv = config.torch_device

    # check if gpu is available
    if device_conv == "cuda" and torch.cuda.is_available():
        device_conv = "cuda"
    else:
        device_conv = "cpu"

    # create simulator
    simulator = FarFieldSimulator(
        psf=psf,
        is_torch=True,
        **config.simulation,
    )
    # create Pytorch dataset and dataloader
    if n_files is not None:
        ds = torch.utils.data.Subset(ds, np.arange(n_files))
    if mask is None:
        ds_prop = SimulatedFarFieldDataset(
            dataset=ds,
            simulator=simulator,
            dataset_is_CHW=True,
            device_conv=device_conv,
            flip=config.simulation.flip,
        )
    else:
        ds_prop = SimulatedDatasetTrainableMask(
            dataset=ds,
            mask=mask,
            simulator=simulator,
            dataset_is_CHW=True,
            device_conv=device_conv,
            flip=config.simulation.flip,
        )
    return ds_prop, mask


@hydra.main(version_base=None, config_path="../../configs", config_name="train_unrolledADMM")
def train_unrolled(config):

    disp = config.display.disp
    if disp < 0:
        disp = None

    save = config.save
    if save:
        save = os.getcwd()

    if config.torch_device == "cuda" and torch.cuda.is_available():
        print("Using GPU for training.")
        device = "cuda"
    else:
        print("Using CPU for training.")
        device = "cpu"

    # # benchmarking dataset:
    # eval_path = os.path.join(get_original_cwd(), config.files.eval_dataset)
    # benchmark_dataset = DiffuserCamTestDataset(
    #     data_dir=eval_path, downsample=config.files.downsample, n_files=config.files.n_files
    # )

    # diffusercam_psf = benchmark_dataset.psf.to(device)
    # # background = benchmark_dataset.background

    # # convert psf from BGR to RGB
    # diffusercam_psf = diffusercam_psf[..., [2, 1, 0]]

    # # create mask
    # if config.trainable_mask.mask_type is not None:
    #     mask_class = getattr(lensless.hardware.trainable_mask, config.trainable_mask.mask_type)
    #     if config.trainable_mask.initial_value == "random":
    #         mask = mask_class(
    #             torch.rand_like(diffusercam_psf), optimizer="Adam", lr=config.trainable_mask.mask_lr
    #         )
    #     elif config.trainable_mask.initial_value == "DiffuserCam":
    #         mask = mask_class(diffusercam_psf, optimizer="Adam", lr=config.trainable_mask.mask_lr)
    #     elif config.trainable_mask.initial_value == "DiffuserCam_gray":
    #         mask = mask_class(
    #             diffusercam_psf[:, :, :, 0, None],
    #             optimizer="Adam",
    #             lr=config.trainable_mask.mask_lr,
    #             is_rgb=not config.simulation.grayscale,
    #         )
    # else:
    #     mask = None

    # load dataset and create dataloader
    train_set = None
    test_set = None
    if "DiffuserCam" in config.files.dataset:

        original_path = os.path.join(get_original_cwd(), config.files.dataset)
        psf_path = os.path.join(get_original_cwd(), config.files.psf)
        dataset = DiffuserCamMirflickr(
            dataset_dir=original_path,
            psf_path=psf_path,
            downsample=config.files.downsample,
        )
        dataset.psf = dataset.psf.to(device)
        # train-test split as in https://waller-lab.github.io/LenslessLearning/dataset.html
        # first 1000 files for test, the rest for training
        train_indices = dataset.allowed_idx[dataset.allowed_idx > 1000]
        test_indices = dataset.allowed_idx[dataset.allowed_idx <= 1000]
        if config.files.n_files is not None:
            train_indices = train_indices[: config.files.n_files]
            test_indices = test_indices[: config.files.n_files]

        train_set = Subset(dataset, train_indices)
        test_set = Subset(dataset, test_indices)
        print("Train test size : ", len(train_set))
        print("Test test size : ", len(test_set))

        # -- if learning mask
        mask = None
        if config.trainable_mask.mask_type is not None:
            mask_class = getattr(lensless.hardware.trainable_mask, config.trainable_mask.mask_type)

            if config.trainable_mask.initial_value == "random":
                mask = mask_class(
                    torch.rand_like(dataset.psf), optimizer="Adam", lr=config.trainable_mask.mask_lr
                )
            # TODO : change to PSF
            elif config.trainable_mask.initial_value == "DiffuserCam":
                mask = mask_class(dataset.psf, optimizer="Adam", lr=config.trainable_mask.mask_lr)
            elif config.trainable_mask.initial_value == "DiffuserCam_gray":
                # TODO convert to grayscale
                mask = mask_class(
                    dataset.psf[:, :, :, 0, None],
                    optimizer="Adam",
                    lr=config.trainable_mask.mask_lr,
                    is_rgb=not config.simulation.grayscale,
                )

            # plot initial PSF
            psf_np = mask.get_psf().detach().cpu().numpy()[0, ...]
            save_image(psf_np, os.path.join(save, "psf_initial.png"))
            plot_image(psf_np, gamma=config.display.gamma)
            plt.savefig(os.path.join(save, "psf_initial_plot.png"))

    else:
        # Use a simulated dataset
        if config.trainable_mask.use_mask_in_dataset:
            train_set, mask = simulate_dataset(config)
            # the mask use will differ from the one in the benchmark dataset
            print("Trainable Mask will be used in the test dataset")
            test_set = None
        else:
            # TODO handlge case where finetuning PSF
            train_set, mask = simulate_dataset(config)

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
            dataset.psf,
            n_iter=config.reconstruction.unrolled_fista.n_iter,
            tk=config.reconstruction.unrolled_fista.tk,
            pad=True,
            learn_tk=config.reconstruction.unrolled_fista.learn_tk,
            pre_process=pre_process,
            post_process=post_process,
        ).to(device)
    elif config.reconstruction.method == "unrolled_admm":
        recon = UnrolledADMM(
            dataset.psf,
            n_iter=config.reconstruction.unrolled_admm.n_iter,
            mu1=config.reconstruction.unrolled_admm.mu1,
            mu2=config.reconstruction.unrolled_admm.mu2,
            mu3=config.reconstruction.unrolled_admm.mu3,
            tau=config.reconstruction.unrolled_admm.tau,
            pre_process=pre_process,
            post_process=post_process,
        ).to(device)
    else:
        raise ValueError(f"{config.reconstruction.method} is not a supported algorithm")

    # constructing algorithm name by appending pre and post process
    algorithm_name = config.reconstruction.method
    if config.reconstruction.pre_process.network is not None:
        algorithm_name = pre_process_name + "_" + algorithm_name
    if config.reconstruction.post_process.network is not None:
        algorithm_name += "_" + post_process_name

    # print number of parameters
    n_param = sum(p.numel() for p in recon.parameters())
    if mask is not None:
        n_param += sum(p.numel() for p in mask.parameters())
    log.info(f"Training model with {n_param} parameters")

    print(f"Setup time : {time.time() - start_time} s")
    print(f"PSF shape : {dataset.psf.shape}")
    trainer = Trainer(
        recon,
        train_set,
        test_set,
        mask=mask,
        batch_size=config.training.batch_size,
        loss=config.loss,
        lpips=config.lpips,
        l1_mask=config.trainable_mask.L1_strength,
        optimizer=config.optimizer.type,
        optimizer_lr=config.optimizer.lr,
        slow_start=config.training.slow_start,
        skip_NAN=config.training.skip_NAN,
        algorithm_name=algorithm_name,
        metric_for_best_model=config.training.metric_for_best_model,
        save_every=config.training.save_every,
        gamma=config.display.gamma,
    )

    trainer.train(n_epoch=config.training.epoch, save_pt=save, disp=disp)


if __name__ == "__main__":
    train_unrolled()
