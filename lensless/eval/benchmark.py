# #############################################################################
# benchmark.py
# =================
# Authors :
# Yohann PERRON
# Eric BEZZAM [ebezzam@gmail.com]
# #############################################################################


from lensless.utils.dataset import DiffuserCamTestDataset
from tqdm import tqdm

from lensless.utils.io import load_image

try:
    import torch
    from torch.utils.data import DataLoader
    from torch.nn import MSELoss, L1Loss
    from torchmetrics import StructuralSimilarityIndexMeasure
    from torchmetrics.image import lpip, psnr
except ImportError:
    raise ImportError(
        "Torch, torchvision, and torchmetrics are needed to benchmark reconstruction algorithm."
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

    if metrics is None:
        metrics = {
            "MSE": MSELoss().to(device),
            "MAE": L1Loss().to(device),
            "LPIPS_Vgg": lpip.LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=True
            ).to(device),
            "LPIPS_Alex": lpip.LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(device),
            "PSNR": psnr.PeakSignalNoiseRatio().to(device),
            "SSIM": StructuralSimilarityIndexMeasure().to(device),
            "ReconstructionError": None,
        }
    metrics_values = {key: 0.0 for key in metrics}

    # loop over batches
    dataloader = DataLoader(dataset, batch_size=batchsize, pin_memory=(device != "cpu"))
    model.reset()
    for lensless, lensed in tqdm(dataloader):
        lensless = lensless.to(device)
        lensed = lensed.to(device)

        # compute predictions
        with torch.no_grad():
            if batchsize == 1:
                model.set_data(lensless)
                prediction = model.apply(plot=False, save=False, **kwargs)

            else:
                prediction = model.batch_call(lensless, **kwargs)

        # Convert to [N*D, C, H, W] for torchmetrics
        prediction = prediction.reshape(-1, *prediction.shape[-3:]).movedim(-1, -3)
        lensed = lensed.reshape(-1, *lensed.shape[-3:]).movedim(-1, -3)
        # normalization
        prediction_max = torch.amax(prediction, dim=(-1, -2, -3), keepdim=True)
        if torch.all(prediction_max != 0):
            prediction = prediction / prediction_max
        else:
            print("Warning: prediction is zero")
        lensed_max = torch.amax(lensed, dim=(1, 2, 3), keepdim=True)
        lensed = lensed / lensed_max
        # compute metrics
        for metric in metrics:
            if metric == "ReconstructionError":
                metrics_values[metric] += model.reconstruction_error().cpu().item()
            else:
                metrics_values[metric] += metrics[metric](prediction, lensed).cpu().item()

        model.reset()

    # average metrics
    for metric in metrics:
        metrics_values[metric] /= len(dataloader)

    return metrics_values


if __name__ == "__main__":
    from lensless import ADMM

    downsample = 1.0
    batchsize = 1
    n_files = 10
    n_iter = 100

    # check if GPU is available
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # prepare dataset
    dataset = DiffuserCamTestDataset(n_files=n_files, downsample=downsample)

    # prepare model
    psf = dataset.psf.to(device)
    model = ADMM(psf, n_iter=n_iter)

    # run benchmark
    print(benchmark(model, dataset, batchsize=batchsize))
