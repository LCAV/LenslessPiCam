# #############################################################################
# benchmark.py
# =========
# Authors :
# Yohann PERRON
# Eric BEZZAM [ebezzam@gmail.com]
# #############################################################################


import glob
import os
from lensless.io import load_psf, load_image
from lensless.util import resize
import numpy as np
from tqdm import tqdm

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torch.nn import MSELoss, L1Loss
    from torchmetrics import StructuralSimilarityIndexMeasure
    from torchmetrics.image import lpip, psnr
    from torchvision import transforms
except ImportError:
    raise ImportError("Torch and torchmetrics are needed to benchmark reconstruction algorithm")


class ParallelDataset(Dataset):
    """
    Dataset consisting of lensless and corresponding lensed image.

    It can be used with a PyTorch DataLoader to load a batch of lensless and corresponding lensed images.

    """

    def __init__(
        self,
        root_dir,
        n_files=False,
        background=None,
        downsample=4,
        flip=False,
        transform_lensless=None,
        transform_lensed=None,
        lensless_fn="diffuser",
        lensed_fn="lensed",
        image_ext="npy",
        **kwargs,
    ):
        """
        Dataset consisting of lensless and corresponding lensed image. Default parameters are for the DiffuserCam
        Lensless Mirflickr Dataset (DLMD).

        Parameters
        ----------

            root_dir : str
                Path to the test dataset. It is expected to contain two folders: ones of lensless images and one of lensed images.
            n_files : int or None, optional
                Metrics will be computed only on the first ``n_files`` images. If None, all images are used, by default False
            background : :py:class:`~torch.Tensor` or None, optional
                If not ``None``, background is removed from lensless images, by default ``None``.
            downsample : int, optional
                Downsample factor of the lensless images, by default 4.
            flip : bool, optional
                If ``True``, lensless images are flipped, by default ``False``.
            transform_lensless : PyTorch Transform or None, optional
                Transform to apply to the lensless images, by default None
            transform_lensed : PyTorch Transform or None, optional
                Transform to apply to the lensed images, by default None
            lensless_fn : str, optional
                Name of the folder containing the lensless images, by default "diffuser".
            lensed_fn : str, optional
                Name of the folder containing the lensed images, by default "lensed".
            image_ext : str, optional
                Extension of the images, by default "npy".
        """

        self.root_dir = root_dir
        self.lensless_dir = os.path.join(root_dir, lensless_fn)
        self.lensed_dir = os.path.join(root_dir, lensed_fn)
        self.image_ext = image_ext.lower()

        files = glob.glob(os.path.join(self.lensless_dir, "*." + image_ext))
        if n_files:
            files = files[:n_files]
        self.files = [os.path.basename(fn) for fn in files]

        if len(self.files) == 0:
            raise FileNotFoundError(
                f"No files found in {self.lensless_dir} with extension {image_ext}"
            )

        self.background = background
        self.downsample = downsample / 4
        self.flip = flip
        self.transform_lensless = transform_lensless
        self.transform_lensed = transform_lensed

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.image_ext == "npy":
            lensless_fp = os.path.join(self.lensless_dir, self.files[idx])
            lensed_fp = os.path.join(self.lensed_dir, self.files[idx])
            lensless = np.load(lensless_fp)
            lensed = np.load(lensed_fp)
        else:
            # more standard image formats: png, jpg, tiff, etc.
            lensless_fp = os.path.join(self.lensless_dir, self.files[idx])
            lensed_fp = os.path.join(self.lensed_dir, self.files[idx])
            lensless = load_image(lensless_fp)
            lensed = load_image(lensed_fp)

            # convert to float
            if lensless.dtype == np.uint8:
                lensless = lensless.astype(np.float32) / 255
                lensed = lensed.astype(np.float32) / 255
            else:
                # 16 bit
                lensless = lensless.astype(np.float32) / 65535
                lensed = lensed.astype(np.float32) / 65535

        if self.downsample != 1.0:
            lensless = resize(lensless, factor=1 / self.downsample)
            lensed = resize(lensed, factor=1 / self.downsample)

        lensless = torch.from_numpy(lensless)
        lensed = torch.from_numpy(lensed)

        if self.background is not None:
            lensless = lensless - self.background

        # flip image x and y if needed
        if self.flip:
            lensless = torch.rot90(lensless, dims=(-3, -2))
            lensed = torch.rot90(lensed, dims=(-3, -2))
        if self.transform_lensless:
            lensless = self.transform_lensless(lensless)

        if self.transform_lensed:
            lensed = self.transform_lensed(lensed)

        return lensless, lensed


class BenchmarkDataset(ParallelDataset):
    """
    Dataset consisting of lensless and corresponding lensed image. This is the standard dataset used for benchmarking.
    """

    def __init__(
        self,
        data_dir="data",
        n_files=200,
        downsample=8,
    ):
        """
        Dataset consisting of lensless and corresponding lensed image. Default parameters are for the test set of DiffuserCam
        Lensless Mirflickr Dataset (DLMD).

        Parameters
        ----------
        data_dir : str, optional
            The path to the folder containing the DiffuserCam_Test dataset, by default "data"
        n_files : int, optional
            Number of image pair to load in the dataset , by default 200
        downsample : int, optional
            Downsample factor of the lensless images, by default 8
        """
        # download dataset if necessary
        data_dir = os.path.join(data_dir, "DiffuserCam_Test")
        if not os.path.isdir(data_dir):
            print("No dataset found for benchmarking.")
            try:
                from torchvision.datasets.utils import download_and_extract_archive
            except ImportError:
                exit()
            msg = "Do you want to download the sample dataset (725MB)?"

            # default to yes if no input is given
            valid = input("%s (Y/n) " % msg).lower() != "n"
            if valid:
                url = "https://drive.switch.ch/index.php/s/vmAZzryGI8U8rcE/download"
                filename = "DiffuserCam_Mirflickr_200_3011302021_11h43_seed11.zip"
                download_and_extract_archive(url, "data/", filename=filename, remove_finished=True)

        psf_fp = os.path.join(data_dir, "psf.tiff")
        psf, background = load_psf(
            psf_fp,
            downsample=downsample,
            return_float=True,
            return_bg=True,
            bg_pix=(0, 15),
        )

        # transform from BGR to RGB
        transform_BRG2RGB = transforms.Lambda(lambda x: x[..., [2, 1, 0]])

        self.psf = transform_BRG2RGB(torch.from_numpy(psf))

        super().__init__(
            data_dir,
            n_files,
            background,
            downsample,
            flip=False,
            transform_lensless=transform_BRG2RGB,
            transform_lensed=transform_BRG2RGB,
            lensless_fn="diffuser",
            lensed_fn="lensed",
            image_ext="npy",
        )


def benchmark(model, dataset, batchsize=1, metrics=None, **kwargs):
    """
    Compute multiple metrics for a reconstruction algorithm.

    Parameters
    ----------
    model : :py:class:`~lensless.ReconstructionAlgorithm`
        Reconstruction algorithm to benchmark.
    dataset : :py:class:`~lensless.benchmark.ParallelDataset`
        Parallel dataset of lensless and lensed images.
    batchsize : int, optional
        Batch size for processing. For maximum compatibility use 1 (batchsize above 1 are not supported on all algorithm), by default 1
    metrics : dict, optional
        Dictionary of metrics to compute. If None, MSE, MAE, SSIM, LPIPS and PSNR are computed.

    Returns
    -------
    Dict[str, float]
        A dictionnary containing the metrics name and average value
    """
    assert isinstance(model._psf, torch.Tensor), "model need to be constructed with torch support"
    device = model._psf.device

    def reconstruction_error(prediction, lensless):
        convolver = model._convolver
        if convolver.pad:
            prediction = prediction.movedim(-3, -1)
        else:
            prediction = convolver._pad(prediction.movedim(-3, -1))
        Fx = convolver.convolve(prediction)
        Fy = lensless
        if not convolver.pad:
            Fx = convolver._crop(Fx)
        return torch.norm(Fx - Fy)

    if metrics is None:
        metrics = {
            "MSE": MSELoss().to(device),
            "MAE": L1Loss().to(device),
            "LPIPS": lpip.LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(device),
            "PSNR": psnr.PeakSignalNoiseRatio().to(device),
            "SSIM": StructuralSimilarityIndexMeasure().to(device),
            "ReconstructionError": reconstruction_error,
        }
    metrics_values = {key: 0.0 for key in metrics}

    # loop over batches
    dataloader = DataLoader(dataset, batch_size=batchsize, pin_memory=(device != "cpu"))
    model.reset()
    for lensless, lensed in tqdm(dataloader):
        lensless = lensless.to(device).squeeze()
        lensed = lensed.to(device).permute(0, 3, 1, 2)

        # compute predictions
        with torch.no_grad():
            if batchsize == 1:
                model.set_data(lensless)
                prediction = model.apply(plot=False, save=False, **kwargs)[None, :, :, :].permute(
                    0, 3, 1, 2
                )
            else:
                prediction = model.batch_call(lensless, **kwargs).permute(0, 3, 1, 2)

        # normalization
        prediction_max = torch.amax(prediction, dim=(1, 2, 3), keepdim=True)
        prediction = prediction / prediction_max
        lensed_max = torch.amax(lensed, dim=(1, 2, 3), keepdim=True)
        lensed = lensed / lensed_max
        # compute metrics
        for metric in metrics:
            if metric == "ReconstructionError":
                metrics_values[metric] += metrics[metric](prediction, lensless).cpu().item()
            else:
                metrics_values[metric] += metrics[metric](prediction, lensed).cpu().item()

        model.reset()

    # average metrics
    for metric in metrics:
        metrics_values[metric] /= len(dataloader)

    return metrics_values


if __name__ == "__main__":
    from lensless import ADMM

    downsample = 4
    batchsize = 1
    n_files = 10
    n_iter = 100

    # check if GPU is available
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # prepare dataset
    dataset = BenchmarkDataset(n_files=n_files, downsample=downsample)

    # prepare model
    psf = dataset.psf.to(device)
    model = ADMM(psf, max_iter=n_iter)

    # run benchmark
    print(benchmark(model, dataset, batchsize=batchsize))
