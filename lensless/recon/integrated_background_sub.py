# #############################################################################
# integrated_background_sub.py
# ============================
# Authors :
# Eric BEZZAM [ebezzam@gmail.com]
# Stefan PETERS
# #############################################################################


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lensless.recon.utils import convert_to_NCHW, convert_to_NDCHW


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.pool_conv(x)


class Down_concat(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # self.down = nn.Downsample(scale_factor=2, mode="bilinear", align_corners=True)
        # self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels * 2, out_channels),
        )

    def forward(self, x1, x2):
        # # don't need passing as should be same shape...
        # diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]
        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([x1, x2], dim=1)
        return self.pool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # or use ConvTranspose2d? https://github.com/milesial/Pytorch-UNet/blob/21d7850f2af30a9695bbeea75f3136aa538cfc4a/unet/unet_parts.py#L53
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))

    def forward(self, x):
        return self.conv(x)


class IntegratedBackgroundSub(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        input_shape,
        down_subtraction=False,  # TODO flag to do background subtraction while downsampling (instead of upsampling)
        nc=None,
    ):
        """
        Integrated Background Subtraction using a U-Net architecture.

        Parameters
        ----------
        in_channels : int TODO
            Number of input channels. RGB or grayscale, i.e. 3 and 1 respectively.
        out_channels : int
            Number of output channels. RGB or grayscale, i.e. 3 and 1 respectively.
        psf : :py:class:`~torch.Tensor`
            Point spread function (PSF) that models forward propagation.
        psf_channels : int
            Number of channels in the PSF. Default is 1.
        down_subtraction : bool
            Flag to do background subtraction while downsampling (instead of upsampling).
        nc : list
            Number of channels in the network. Default is [64, 128, 256, 512, 512].

        """
        assert in_channels == 1 or in_channels == 3, "in_channels must be 1 or 3"
        assert out_channels == 1 or out_channels == 3, "out_channels must be 1 or 3"
        assert in_channels >= out_channels
        if nc is None:
            nc = [64, 128, 256, 512, 512]

        super(IntegratedBackgroundSub, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.inc = DoubleConv(in_channels, nc[0])
        self.down_subtraction = down_subtraction

        if self.down_subtraction:
            self.down_layers = nn.ModuleList(
                [Down_concat(nc[i], nc[i + 1]) for i in range(len(nc) - 1)]
            )
        else:
            self.down_layers = nn.ModuleList([Down(nc[i], nc[i + 1]) for i in range(len(nc) - 1)])

        # background downsampling layers
        self.inc_back = DoubleConv(in_channels, nc[0])
        self.down_layers_bg = nn.ModuleList([Down(nc[i], nc[i + 1]) for i in range(len(nc) - 2)])
        self.subtraction_weight = nn.Parameter(torch.ones(len(nc) - 1))

        # upsampling layers
        self.up_layers = []
        n_prev = nc[-1]
        for i in range(len(nc) - 1):
            n_in = n_prev + nc[-i - 2]
            n_out = nc[-i - 2] // 2 if i < len(nc) - 2 else nc[0]
            self.up_layers.append(Up(n_in, n_out))
            n_prev = n_out
        self.up_layers = nn.ModuleList(self.up_layers)
        self.outc = OutConv(nc[0], out_channels)

        # padding H and W to next multiple of 8
        self.top = (8 - input_shape[0] % 8) // 2
        self.bottom = (8 - input_shape[0] % 8) - self.top
        self.left = (8 - input_shape[1] % 8) // 2
        self.right = (8 - input_shape[1] % 8) - self.left

    def forward(self, batch, background=None, **kwargs):
        assert batch.shape[-3:-1] == background.shape[-3:-1], "Data and background shape mismatch"
        assert len(batch.shape) >= 3, "Batch must have at least 3 dimensions"

        n_depth = batch[0].shape[-4]
        if n_depth > 1:
            raise NotImplementedError("3D not implemented yet.")

        # pad to multiple of 8
        batch = convert_to_NCHW(batch)
        batch = torch.nn.functional.pad(
            batch, (self.left, self.right, self.top, self.bottom), mode="constant", value=0
        )
        background = convert_to_NCHW(background)
        background = torch.nn.functional.pad(
            background, (self.left, self.right, self.top, self.bottom), mode="constant", value=0
        )

        # Select the images
        x_inter = [self.inc(batch)]

        # -- downscale background
        bg_inter = [self.inc_back(background)]
        for i in range(len(self.down_layers_bg)):
            bg_inter.append(self.down_layers_bg[i](bg_inter[-1]))

        if self.down_subtraction:
            # Concatenate background subtracted data during downsampling
            batch = x_inter[0]

            # -- downsample
            # for i in range(len(bg_inter)):
            #    x_inter.append(self.down_layers[i](batch - bg_inter[i], bg_inter[i]))
            # [(M-B), B]
            for i in range(len(bg_inter)):
                # -- [(M-B), M]
                # batch = self.down_layers[i](batch - self.subtraction_weight[i] * bg_inter[i], batch)
                # -- [(M-B), B]
                batch = self.down_layers[i](
                    batch - self.subtraction_weight[i] * bg_inter[i], bg_inter[i]
                )
                x_inter.append(batch)

            # upsample
            batch = self.up_layers[0](x_inter[-1], x_inter[-2])
            for i in range(len(self.up_layers) - 1):
                batch = self.up_layers[i + 1](batch, x_inter[-i - 3])

        else:
            # Concatenate background subtracted data during upsampling
            # -- downsample
            for i in range(len(self.down_layers)):
                x_inter.append(self.down_layers[i](x_inter[-1]))

            # simple subtraction between the downsapled backgrounds and the corresponding data
            for i in range(len(bg_inter)):
                x_inter[i] = x_inter[i] - self.subtraction_weight[i] * bg_inter[i]

            # upsample
            batch = self.up_layers[0](x_inter[-1], x_inter[-2])
            for i in range(len(self.up_layers) - 1):
                batch = self.up_layers[i + 1](batch, x_inter[-i - 3])

        batch = self.outc(batch)

        # back to original shape
        batch = batch[..., self.top : -self.bottom, self.left : -self.right]
        batch = convert_to_NDCHW(batch, n_depth)

        # normalize to [0,1], TODO use sigmoid instead?
        batch = (batch + 1) / 2
        batch = torch.clip(batch, min=0.0)

        return batch
