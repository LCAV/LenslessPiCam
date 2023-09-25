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

By default it uses the configuration from the file `configs/train_unrolledADMM.yaml`.

To train pre- and post-processing networks, use the following command:
```
python scripts/recon/train_unrolled.py -cn train_pre-post-processing
```

To fine-tune the DiffuserCam PSF, use the following command:
```
python scripts/recon/train_unrolled.py -cn fine-tune_PSF
```

To train a PSF from scratch with a simulated dataset, use the following command:
```
python scripts/recon/train_unrolled.py -cn train_psf_from_scratch
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
from lensless.utils.image import rgb2gray, is_grayscale
from lensless.utils.simulation import FarFieldSimulator
from lensless.recon.utils import Trainer, device_checks
import torch
from torchvision import transforms, datasets
from lensless.utils.io import load_psf
from lensless.utils.io import save_image
from lensless.utils.plot import plot_image
import matplotlib.pyplot as plt

# A logger for this file
log = logging.getLogger(__name__)


class MyDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def simulate_dataset(config):

    # if config.torch_device == "cuda" and torch.cuda.is_available():
    #     device = "cuda"
    # else:
    #     device = "cpu"
    device, use_cuda, multi_gpu, device_ids = device_checks(config.torch_device, config.multi_gpu)

    import pudb

    pudb.set_trace()

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

    # drop depth dimension
    psf = psf.to(device)

    # load dataset
    transforms_list = [transforms.ToTensor()]
    data_path = os.path.join(get_original_cwd(), "data")
    if config.simulation.grayscale:
        transforms_list.append(transforms.Grayscale())
    transform = transforms.Compose(transforms_list)
    if config.files.dataset == "mnist":
        train_ds = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
        test_ds = datasets.MNIST(root=data_path, train=False, download=True, transform=transform)
    elif config.files.dataset == "fashion_mnist":
        train_ds = datasets.FashionMNIST(
            root=data_path, train=True, download=True, transform=transform
        )
        test_ds = datasets.FashionMNIST(
            root=data_path, train=False, download=True, transform=transform
        )
    elif config.files.dataset == "cifar10":
        train_ds = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
        test_ds = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)
    elif config.files.dataset == "CelebA":
        root = config.files.celeba_root
        data_path = os.path.join(root, "celeba")
        assert os.path.isdir(
            data_path
        ), f"Data path {data_path} does not exist. Make sure you download the CelebA dataset and provide the parent directory as 'config.files.celeba_root'. Download link: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html"
        train_ds = datasets.CelebA(root=root, split="train", download=False, transform=transform)
        test_ds = datasets.CelebA(root=root, split="test", download=False, transform=transform)
    else:
        raise NotImplementedError(f"Dataset {config.files.dataset} not implemented.")

    # convert PSF
    if config.simulation.grayscale and not is_grayscale(psf):
        psf = rgb2gray(psf)

    # prepare mask
    mask = prep_trainable_mask(config, psf, grayscale=config.simulation.grayscale)

    # check if gpu is available
    device_conv = config.torch_device
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
    n_files = config.files.n_files
    if n_files is not None:
        train_ds = torch.utils.data.Subset(train_ds, np.arange(n_files))
        test_ds = torch.utils.data.Subset(test_ds, np.arange(n_files))
    if mask is None:
        train_ds_prop = SimulatedFarFieldDataset(
            dataset=train_ds,
            simulator=simulator,
            dataset_is_CHW=True,
            device_conv=device_conv,
            flip=config.simulation.flip,
        )
        test_ds_prop = SimulatedFarFieldDataset(
            dataset=test_ds,
            simulator=simulator,
            dataset_is_CHW=True,
            device_conv=device_conv,
            flip=config.simulation.flip,
        )
    else:
        train_ds_prop = SimulatedDatasetTrainableMask(
            dataset=train_ds,
            mask=mask,
            simulator=simulator,
            dataset_is_CHW=True,
            device_conv=device_conv,
            flip=config.simulation.flip,
        )
        test_ds_prop = SimulatedDatasetTrainableMask(
            dataset=test_ds,
            mask=mask,
            simulator=simulator,
            dataset_is_CHW=True,
            device_conv=device_conv,
            flip=config.simulation.flip,
        )

    return train_ds_prop, test_ds_prop, mask


def prep_trainable_mask(config, psf, grayscale=False):
    mask = None
    if config.trainable_mask.mask_type is not None:
        mask_class = getattr(lensless.hardware.trainable_mask, config.trainable_mask.mask_type)

        if config.trainable_mask.initial_value == "random":
            initial_mask = torch.rand_like(psf)
        elif config.trainable_mask.initial_value == "psf":
            initial_mask = psf.clone()
        else:
            raise ValueError(
                f"Initial PSF value {config.trainable_mask.initial_value} not supported"
            )

        if config.trainable_mask.grayscale and not is_grayscale(initial_mask):
            initial_mask = rgb2gray(initial_mask)

        mask = mask_class(
            initial_mask, optimizer="Adam", lr=config.trainable_mask.mask_lr, grayscale=grayscale
        )

    return mask


@hydra.main(version_base=None, config_path="../../configs", config_name="train_unrolledADMM")
def train_unrolled(config):

    disp = config.display.disp
    if disp < 0:
        disp = None

    save = config.save
    if save:
        save = os.getcwd()

    device, use_cuda, multi_gpu, device_ids = device_checks(
        config.torch_device, config.multi_gpu, logger=log.info
    )

    # load dataset and create dataloader
    train_set = None
    test_set = None
    psf = None
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

        # -- if learning mask
        mask = prep_trainable_mask(config, dataset.psf)
        if mask is not None:
            # plot initial PSF
            psf_np = mask.get_psf().detach().cpu().numpy()[0, ...]
            if config.trainable_mask.grayscale:
                psf_np = psf_np[:, :, -1]

            save_image(psf_np, os.path.join(save, "psf_initial.png"))
            plot_image(psf_np, gamma=config.display.gamma)
            plt.savefig(os.path.join(save, "psf_initial_plot.png"))

        psf = dataset.psf

    else:

        train_set, test_set, mask = simulate_dataset(config)
        psf = train_set.psf

    assert train_set is not None
    assert psf is not None

    log.info(f"Train test size : {len(train_set)}")
    log.info(f"Test test size : {len(test_set)}")

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
        )
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
        )
    else:
        raise ValueError(f"{config.reconstruction.method} is not a supported algorithm")

    if multi_gpu:
        # recon = torch.nn.DataParallel(recon, device_ids=device_ids)
        recon = MyDataParallel(recon, device_ids=device_ids)
    if use_cuda:
        recon.to(device)

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

    log.info(f"Setup time : {time.time() - start_time} s")
    log.info(f"PSF shape : {psf.shape}")
    log.info(f"Results saved in {save}")
    trainer = Trainer(
        recon=recon,
        train_dataset=train_set,
        test_dataset=test_set,
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
        logger=log,
    )

    trainer.train(n_epoch=config.training.epoch, save_pt=save, disp=disp)

    log.info(f"Results saved in {save}")


if __name__ == "__main__":
    train_unrolled()
