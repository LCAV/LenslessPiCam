# #############################################################################
# benchmark.py
# =================
# Authors :
# Yohann PERRON
# Eric BEZZAM [ebezzam@gmail.com]
# #############################################################################


from lensless.utils.dataset import DiffuserCamTestDataset
from lensless.utils.io import save_image
from waveprop.noise import add_shot_noise
from tqdm import tqdm
import os
import numpy as np
import wandb

try:
    import torch
    from torch.utils.data import DataLoader
    from torch.nn import MSELoss, L1Loss
    from torchmetrics import StructuralSimilarityIndexMeasure
    from torchmetrics.image import lpip, psnr
    from torch.optim import SGD
except ImportError:
    raise ImportError(
        "Torch, torchvision, and torchmetrics are needed to benchmark reconstruction algorithm."
    )


# TODO put Parameterize and Perturb functions in a separate file
def get_param(model):
    all_param = []
    for _, param in model.named_parameters():
        all_param.append(param)
    all_param = torch.cat([param.flatten() for param in all_param])
    return all_param


def pnp_loss(y_est, y, recon, mu, original_param):
    new_param = get_param(recon)
    loss = torch.mean((y_est - y) ** 2) + mu * torch.mean((new_param - original_param) ** 2)
    return loss


def swap_color_channels(tensor, channels):
    if len(channels) == 2:
        color1 = tensor[..., channels[0]].copy()
        tensor[..., channels[0]] = tensor[..., channels[1]]
        tensor[..., channels[1]] = color1
    elif len(channels) == 3:
        # reorder channels
        color1 = tensor[..., channels[0]].copy()
        color2 = tensor[..., channels[1]].copy()
        color3 = tensor[..., channels[2]].copy()
        tensor[..., 0] = color1
        tensor[..., 1] = color2
        tensor[..., 2] = color3
    return tensor


def benchmark(
    model,
    dataset,
    batchsize=1,
    metrics=None,
    crop=None,
    save_idx=None,
    save_intermediate=False,
    output_dir=None,
    unrolled_output_factor=False,
    pre_process_aux=False,
    return_average=True,
    snr=None,
    use_wandb=False,
    label=None,
    epoch=None,
    use_background=True,
    pnp=None,
    swap_channels=False,
    **kwargs,
):
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
    save_idx : list of int, optional
        List of indices to save the predictions, by default None (not to save any).
    output_dir : str, optional
        Directory to save the predictions, by default save in working directory if save_idx is provided.
    crop : dict, optional
        Dictionary of crop parameters (vertical: [start, end], horizontal: [start, end]), by default None (no crop).
    unrolled_output_factor : bool, optional
        If True, compute metrics for unrolled output, by default False.
    return_average : bool, optional
        If True, return the average value of the metrics, by default True.
    snr : float, optional
        Signal to noise ratio for adding shot noise. If None, no noise is added, by default None.
    use_background: bool, optional
        If dataset has background, use it for reconstruction, by default True.
    pnp : dict, optional
        Dictionary of parameters for (Parameterize and perturb) algorithm, by default None.
        Required keys: "mu" (distance from original parameters), "lr" (SGD learning rate), "n_iter" (number of iterations), "model_path" (original model path).

    Returns
    -------
    Dict[str, float]
        A dictionary containing the metrics name and average value
    """
    assert isinstance(model._psf, torch.Tensor), "model need to be constructed with torch support"
    device = model._psf.device

    if output_dir is None:
        output_dir = os.getcwd()
    else:
        output_dir = str(output_dir)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

    if pnp is not None:
        assert "mu" in pnp, "mu must be provided"
        assert "lr" in pnp, "lr must be provided"
        assert "n_iter" in pnp, "n_iter must be provided"
        assert "model_path" in pnp, "model_path must be provided"
        assert isinstance(pnp["mu"], float), "mu must be a float"
        assert isinstance(pnp["lr"], float), "lr must be a float"
        assert isinstance(pnp["n_iter"], int), "n_iter must be an int"
        assert isinstance(pnp["model_path"], str), "model_path must be a string"
        assert batchsize == 1, "batchsize must be 1 for parameterize and perturb"
        original_param = get_param(model)

    if metrics is None:
        metrics = {
            "MSE": MSELoss(reduction="mean").to(device),
            "LPIPS_Vgg": lpip.LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=True, reduction="sum"
            ).to(device),
            # "LPIPS_Alex": lpip.LearnedPerceptualImagePatchSimilarity(
            #     net_type="alex", normalize=True
            # ).to(device),
            "PSNR": psnr.PeakSignalNoiseRatio(reduction=None, dim=(1, 2, 3), data_range=(0, 1)).to(
                device
            ),
            "SSIM": StructuralSimilarityIndexMeasure(reduction=None, data_range=(0, 1)).to(device),
            "ReconstructionError": None,
        }

    metrics_values = {key: [] for key in metrics}
    if unrolled_output_factor:
        output_metrics = metrics.keys()
        for key in output_metrics:
            if key != "ReconstructionError":
                metrics_values[key + "_unrolled"] = []
    if pre_process_aux:
        metrics_values["ReconstructionError_PreProc"] = []
    output_intermediate = unrolled_output_factor or pre_process_aux or save_intermediate

    # loop over batches
    dataloader = DataLoader(dataset, batch_size=batchsize, pin_memory=(device != "cpu"))
    model.reset()
    idx = 0
    weights = []  # for averaging batches
    for batch in tqdm(dataloader):
        weights.append(len(batch[0]))

        flip_lr = None
        flip_ud = None
        background = None
        lensless = batch[0].to(device)
        lensed = batch[1].to(device)
        if dataset.measured_bg and use_background:
            background = batch[-1].to(device)
        if dataset.multimask or dataset.random_flip:
            psfs = batch[2]
            psfs = psfs.to(device)
        else:
            psfs = None
        if dataset.random_flip:
            flip_lr = batch[3]
            flip_ud = batch[4]

        # add shot noise
        if snr is not None:
            for i in range(lensless.shape[0]):
                lensless[i] = add_shot_noise(lensless[i], float(snr))

        # compute predictions
        if batchsize == 1:

            if pnp is not None:
                from lensless.recon.model_dict import load_model
                from lensless.recon.rfft_convolve import RealFFTConvolve2D

                psf = psfs[0].to(device)
                recon_pnp = load_model(pnp["model_path"], psf, device=device, verbose=False)

                # define optimizer
                optimizer = SGD(recon_pnp.parameters(), lr=pnp["lr"])

                # create forward model (which needs padding)
                forward_model_param = recon_pnp._convolver_param.copy()
                forward_model_param["pad"] = True
                forward_model = RealFFTConvolve2D(psf, **forward_model_param)

                # iterate / minimize loss
                for i in range(pnp["n_iter"]):
                    optimizer.zero_grad()
                    recon_pnp.set_data(lensless)
                    prediction = recon_pnp.apply(
                        disp_iter=-1,
                        save=False,
                        gamma=None,
                        plot=False,
                    )
                    # simulate measurement
                    y_est = forward_model.convolve(prediction)
                    # -- normalize
                    y_est = y_est - y_est.min()
                    y_est = y_est / y_est.max()
                    loss = pnp_loss(y_est, lensless, recon_pnp, pnp["mu"], original_param)
                    loss.backward()
                    optimizer.step()
                    # if i % 10 == 0:
                    #     print(f"Loss {i}: {loss}")
                # prediction = recon_pnp.apply(
                #     disp_iter=-1,
                #     save=False,
                #     gamma=None,
                #     plot=False,
                # ).detach()
                prediction = prediction.detach()

            else:
                with torch.no_grad():
                    if psfs is not None:
                        model._set_psf(psfs[0])
                    model.set_data(lensless)
                    prediction = model.apply(
                        plot=False,
                        save=False,
                        output_intermediate=output_intermediate,
                        background=background,
                        **kwargs,
                    )

        else:
            with torch.no_grad():
                prediction = model.forward(
                    batch=lensless, psfs=psfs, background=background, **kwargs
                )

        if output_intermediate:
            psfs_out = prediction[3]
            pre_process_out = prediction[2]
            unrolled_out = prediction[1]
            prediction = prediction[0]
        prediction_original = prediction.clone()

        # Convert to [N*D, C, H, W] for torchmetrics
        prediction = prediction.reshape(-1, *prediction.shape[-3:]).movedim(-1, -3)
        lensed = lensed.reshape(-1, *lensed.shape[-3:]).movedim(-1, -3)

        if hasattr(dataset, "alignment"):
            if dataset.alignment is not None:
                prediction = dataset.extract_roi(
                    prediction, axis=(-2, -1), flip_lr=flip_lr, flip_ud=flip_ud
                )
            else:
                prediction, lensed = dataset.extract_roi(
                    prediction, axis=(-2, -1), lensed=lensed, flip_lr=flip_lr, flip_ud=flip_ud
                )
        elif crop is not None:
            assert flip_lr is None and flip_ud is None
            prediction = prediction[
                ...,
                crop["vertical"][0] : crop["vertical"][1],
                crop["horizontal"][0] : crop["horizontal"][1],
            ]
            lensed = lensed[
                ...,
                crop["vertical"][0] : crop["vertical"][1],
                crop["horizontal"][0] : crop["horizontal"][1],
            ]

        if save_idx is not None:
            for i, _batch_idx in enumerate(np.arange(idx, idx + batchsize)):
                if _batch_idx in save_idx:
                    prediction_np = prediction.cpu().numpy()[i]
                    # switch to [H, W, C] for saving
                    prediction_np = np.moveaxis(prediction_np, 0, -1)
                    fp = os.path.join(output_dir, f"{_batch_idx}.png")
                    save_image(prediction_np, fp=fp)

                    if save_intermediate:
                        fp = os.path.join(output_dir, f"{_batch_idx}_inv.png")
                        unrolled_out_np = unrolled_out.cpu().numpy()[i].squeeze()
                        # -- swap red and green channels
                        if swap_channels:
                            unrolled_out_np = swap_color_channels(unrolled_out_np, swap_channels)
                        save_image(unrolled_out_np, fp=fp)

                        fp = os.path.join(output_dir, f"{_batch_idx}_preproc.png")
                        pre_process_out_np = pre_process_out.cpu().numpy()[i].squeeze()
                        # -- swap red and green channels
                        if swap_channels:
                            pre_process_out_np = swap_color_channels(
                                pre_process_out_np, swap_channels
                            )
                        save_image(pre_process_out_np, fp=fp)

                        if psfs_out is not None:
                            fp = os.path.join(output_dir, f"{_batch_idx}_psfs.png")
                            if psfs_out.shape[0] == 1:
                                psfs_out_np = psfs_out.cpu().numpy().squeeze()
                            else:
                                psfs_out_np = psfs_out.cpu().numpy()[i].squeeze()

                            # -- swap red and green channels
                            if swap_channels:
                                psfs_out_np = swap_color_channels(psfs_out_np, swap_channels)
                            save_image(psfs_out_np, fp=fp)

                    if use_wandb:
                        assert epoch is not None, "epoch must be provided for wandb logging"
                        log_key = f"{_batch_idx}_{label}" if label is not None else f"{_batch_idx}"
                        wandb.log({log_key: wandb.Image(fp)}, step=epoch)

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
                metrics_values[metric] += model.reconstruction_error(
                    prediction=prediction_original, lensless=lensless, psfs=psfs
                ).tolist()
            else:
                try:
                    if "LPIPS" in metric:
                        if prediction.shape[1] == 1:
                            # LPIPS needs 3 channels
                            metrics_values[metric].append(
                                metrics[metric](
                                    prediction.repeat(1, 3, 1, 1), lensed.repeat(1, 3, 1, 1)
                                )
                                .cpu()
                                .item()
                            )
                        else:
                            metrics_values[metric].append(
                                metrics[metric](prediction, lensed).cpu().item()
                            )
                    elif metric == "MSE":
                        metrics_values[metric].append(
                            metrics[metric](prediction, lensed).cpu().item() * len(batch[0])
                        )
                    else:
                        vals = metrics[metric](prediction, lensed).cpu()
                        if hasattr(vals.tolist(), "__len__"):
                            metrics_values[metric] += vals.tolist()
                        else:
                            metrics_values[metric].append(vals.item())
                except Exception as e:
                    print(f"Error in metric {metric}: {e}")

        # compute metrics for unrolled output
        if unrolled_output_factor:

            # -- convert to CHW and remove depth
            unrolled_out = unrolled_out.reshape(-1, *unrolled_out.shape[-3:]).movedim(-1, -3)

            # -- extraction region of interest
            if hasattr(dataset, "alignment"):
                if dataset.alignment is not None:
                    unrolled_out = dataset.extract_roi(unrolled_out, axis=(-2, -1))
                else:
                    unrolled_out = dataset.extract_roi(
                        unrolled_out,
                        axis=(-2, -1),
                        # lensed=lensed   # lensed already extracted before
                    )
                assert np.all(lensed.shape == unrolled_out.shape)
            elif crop is not None:
                unrolled_out = unrolled_out[
                    ...,
                    crop["vertical"][0] : crop["vertical"][1],
                    crop["horizontal"][0] : crop["horizontal"][1],
                ]

            # -- normalization
            unrolled_out_max = torch.amax(unrolled_out, dim=(-1, -2, -3), keepdim=True)
            if torch.all(unrolled_out_max != 0):
                unrolled_out = unrolled_out / unrolled_out_max

            # -- compute metrics
            for metric in metrics:
                if metric == "ReconstructionError":
                    # only have this for final output
                    continue
                else:
                    if "LPIPS" in metric:
                        if unrolled_out.shape[1] == 1:
                            # LPIPS needs 3 channels
                            metrics_values[metric + "_unrolled"].append(
                                metrics[metric](
                                    unrolled_out.repeat(1, 3, 1, 1), lensed.repeat(1, 3, 1, 1)
                                )
                                .cpu()
                                .item()
                            )
                        else:
                            metrics_values[metric + "_unrolled"].append(
                                metrics[metric](unrolled_out, lensed).cpu().item()
                            )
                    elif metric == "MSE":
                        metrics_values[metric + "_unrolled"].append(
                            metrics[metric](unrolled_out, lensed).cpu().item() * len(batch[0])
                        )
                    else:
                        vals = metrics[metric](unrolled_out, lensed).cpu()
                        if hasattr(vals.tolist(), "__len__"):
                            metrics_values[metric + "_unrolled"] += vals.tolist()
                        else:
                            metrics_values[metric + "_unrolled"].append(vals.item())

        # compute metrics for pre-processed output
        if pre_process_aux:
            metrics_values["ReconstructionError_PreProc"] += model.reconstruction_error(
                prediction=prediction_original, lensless=pre_process_out
            ).tolist()

        model.reset()
        idx += batchsize

    # average metrics
    if return_average:
        for metric in metrics_values.keys():
            if "MSE" in metric or "LPIPS" in metric:
                # differently because metrics are grouped into bathces
                metrics_values[metric] = np.sum(metrics_values[metric]) / len(dataset)
            else:
                metrics_values[metric] = np.mean(metrics_values[metric])

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
