import glob
import os
import pathlib as plib
from datetime import datetime
from lensless.io import load_psf
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import MSELoss, L1Loss
from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics.image import lpip, psnr
from tqdm import tqdm


class Real_Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(
        self,
        root_dir,
        n_files=False,
        background=None,
        transform_diffuser=None,
        transform_lensed=None,
    ):

        self.root_dir = root_dir
        self.diffuser_dir = os.path.join(root_dir, "diffuser")
        self.lensed_dir = os.path.join(root_dir, "lensed")
        files = glob.glob(self.diffuser_dir + "/*.npy")
        print(self.diffuser_dir)
        if n_files:
            files = files[:n_files]
        self.files = [os.path.basename(fn) for fn in files]

        self.background = background
        self.transform_diffuser = transform_diffuser
        self.transform_lensed = transform_lensed

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        diffuser_fp = os.path.join(self.diffuser_dir, self.files[idx])
        lensed_fp = os.path.join(self.lensed_dir, self.files[idx])
        diffuser = torch.from_numpy(np.load(diffuser_fp))
        lensed = torch.from_numpy(np.load(lensed_fp))

        if self.background is not None:
            diffuser = diffuser - self.background

        if self.transform_diffuser:
            diffuser = self.transform_diffuser(diffuser)

        if self.transform_lensed:
            lensed = self.transform_lensed(lensed)

        return diffuser, lensed


def benchmark(model, data, downsample=4, n_files=100, batchsize=1, **kwargs):

    assert isinstance(model._psf, torch.Tensor), "model need to be constructed with torch support"
    device = model._psf.device

    psf_fp = os.path.join(data, "psf.tiff")
    psf_float, background = load_psf(
        psf_fp,
        downsample=downsample,
        return_float=True,
        return_bg=True,
        bg_pix=(0, 15),
    )
    background = torch.from_numpy(background)

    dataset = Real_Dataset(data, n_files=n_files, background=background)
    dataloader = DataLoader(dataset, batch_size=batchsize, pin_memory=(device != "cpu"))

    metrics = {
        "MSE": MSELoss().to(device),
        "MAE": L1Loss().to(device),
        "LPIPS": lpip.LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(device),
        "PSNR": psnr.PeakSignalNoiseRatio().to(device),
        "SSIM": StructuralSimilarityIndexMeasure().to(device),
    }
    metrics_values = {key: 0.0 for key in metrics}

    for diffuser, lensed in tqdm(dataloader):
        diffuser = diffuser.to(device).squeeze()
        lensed = lensed.to(device).permute(0, 3, 1, 2)

        with torch.no_grad():
            if batchsize == 1:
                model.set_data(diffuser)
                prediction = model.apply(plot=False, save=False, **kwargs)[None, :, :, :].permute(
                    0, 3, 1, 2
                )
            else:
                prediction = model.batch_call(plot=False, save=False, **kwargs).permute(0, 3, 1, 2)

        for metric in metrics:
            metrics_values[metric] += metrics[metric](prediction, lensed).cpu().item()

    for metric in metrics:
        metrics_values[metric] /= len(dataloader)

    return metrics_values


if __name__ == "__main__":
    from lensless import ADMM

    downsample = 4
    device = "cpu"

    data = "data/DiffuserCam"
    psf_fp = os.path.join(data, "psf.tiff")
    psf_float, background = load_psf(
        psf_fp,
        downsample=downsample,
        return_float=True,
        return_bg=True,
        bg_pix=(0, 15),
    )
    psf = torch.from_numpy(psf_float).to(device)
    model = ADMM(psf)
    print(benchmark(model, data, n_files=10, downsample=downsample, n_iter=100))
