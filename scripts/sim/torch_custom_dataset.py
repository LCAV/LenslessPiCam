from waveprop.dataset_util import SimulatedDatasetFolder
import hydra
from hydra.utils import to_absolute_path
import os
from lensless.utils.io import load_psf
import torch
from torchvision.transforms import ToTensor
import time


@hydra.main(
    version_base=None,
    config_path="../../configs/simulate",
    config_name="simulate_torch_custom_dataset",
)
def simulate(config):

    dataset = to_absolute_path(config.files.dataset)
    if not os.path.isdir(dataset):
        print(f"No dataset found at {dataset}")
        try:
            from torchvision.datasets.utils import download_and_extract_archive, download_url
        except ImportError:
            exit()
        msg = "Do you want to download the sample CelebA dataset (764KB)?"

        # default to yes if no input is given
        valid = input("%s (Y/n) " % msg).lower() != "n"
        if valid:
            url = "https://drive.switch.ch/index.php/s/Q5OdDQMwhucIlt8/download"
            filename = "celeb_mini.zip"
            download_and_extract_archive(
                url, os.path.dirname(dataset), filename=filename, remove_finished=True
            )

    psf_fp = to_absolute_path(config.files.psf)
    assert os.path.exists(psf_fp), f"PSF {psf_fp} does not exist."

    image_ext = config.files.image_ext
    n_files = config.files.n_files
    batch_size = config.files.batch_size
    downsample = config.simulation.downsample
    device_conv = config.device_conv

    # check if gpu is available
    if device_conv == "cuda" and torch.cuda.is_available():
        print("Using GPU for convolution.")
        device_conv = "cuda"
    else:
        print("Using CPU for convolution.")
        device_conv = "cpu"

    # load PSF
    psf = load_psf(psf_fp, downsample=downsample)
    psf_tensor = ToTensor()(psf[0])  # first depth

    # create Pytorch dataset and dataloader
    ds = SimulatedDatasetFolder(
        path=dataset,
        iamge_ext=image_ext,
        n_files=n_files,
        psf=psf_tensor,
        device_conv=device_conv,
        **config.simulation,
    )
    ds_loader = torch.utils.data.DataLoader(dataset=ds, batch_size=batch_size, shuffle=True)

    # loop over batches
    start_time = time.time()
    for i, (x, target) in enumerate(ds_loader):

        if i == 0:
            print("Batch shape : ", x.shape)
            print("Target shape : ", target.shape)
            print("Batch device : ", x.device)

    print("Time per batch : ", (time.time() - start_time) / len(ds_loader))
    print(f"Went through {len(ds_loader)} batches.")


if __name__ == "__main__":
    simulate()
