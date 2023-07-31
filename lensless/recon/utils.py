import torch
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
