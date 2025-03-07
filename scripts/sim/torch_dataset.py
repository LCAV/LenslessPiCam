from waveprop.dataset_util import SimulatedPytorchDataset
import hydra
from hydra.utils import to_absolute_path
import os
from lensless.utils.io import load_psf
from lensless.utils.image import rgb2gray
import torch
import time
import numpy as np
from torchvision import transforms, datasets
from tqdm import tqdm


@hydra.main(
    version_base=None, config_path="../../configs/simulate", config_name="simulate_torch_dataset"
)
def simulate(config):

    # load dataset
    transforms_list = [transforms.ToTensor()]
    if config.simulation.grayscale:
        transforms_list.append(transforms.Grayscale())
    transform = transforms.Compose(transforms_list)
    if config.files.dataset == "mnist":
        ds = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    elif config.files.dataset == "fashion_mnist":
        ds = datasets.FashionMNIST(root="data", train=True, download=True, transform=transform)
    elif config.files.dataset == "cifar10":
        ds = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
    else:
        raise NotImplementedError(f"Dataset {config.files.dataset} not implemented.")

    # load PSF
    if config.files.psf is not None:
        psf_fp = to_absolute_path(config.files.psf)
        assert os.path.exists(psf_fp), f"PSF {psf_fp} does not exist."
        psf = load_psf(psf_fp, downsample=config.simulation.downsample)
        if config.simulation.grayscale:
            psf = rgb2gray(psf)
        psf = transforms.ToTensor()(psf[0])  # first depth
    else:
        assert config.simulation.output_dim is not None
        psf = None

    # batch_size = config.files.batch_size
    batch_size = config.files.batch_size
    n_files = config.files.n_files
    device_conv = config.device_conv
    target = config.target

    # check if gpu is available
    if device_conv == "cuda" and torch.cuda.is_available():
        print("Using GPU for convolution.")
        device_conv = "cuda"
    else:
        print("Using CPU for convolution.")
        device_conv = "cpu"

    # create Pytorch dataset and dataloader
    if n_files is not None:
        ds = torch.utils.data.Subset(ds, np.arange(n_files))
    ds_prop = SimulatedPytorchDataset(
        dataset=ds, psf=psf, device_conv=device_conv, target=target, **config.simulation
    )
    ds_loader = torch.utils.data.DataLoader(dataset=ds_prop, batch_size=batch_size, shuffle=True)

    # loop over batches
    start_time = time.time()
    for i, (x, target) in enumerate(tqdm(ds_loader)):

        if i == 0:
            print("Batch shape : ", x.shape)
            print("Target shape : ", target.shape)
            print("Batch device : ", x.device)

    print("Time per batch : ", (time.time() - start_time) / len(ds_loader))
    print(f"Went through {len(ds_loader)} batches.")


if __name__ == "__main__":
    simulate()
