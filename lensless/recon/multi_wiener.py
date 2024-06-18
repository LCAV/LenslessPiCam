"""
Adapted from source code by KC Lee.

"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lensless.recon.utils import convert_to_NCHW, convert_to_NDCHW
from lensless.recon.rfft_convolve import RealFFTConvolve2D


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            # nn.Conv2d(in_channels, mid_channels, kernel_size=(3, 3), padding=1),
            # nn.BatchNorm2d(mid_channels),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(mid_channels, out_channels, kernel_size=(3, 3), padding=1),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True)
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
        self.avgpool_conv = nn.Sequential(
            # nn.AvgPool2d(2),
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.avgpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

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


class UpDec(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))

    def forward(self, x):
        return self.conv(x)


def WieNer(blur, psf, delta):
    blur_fft = torch.fft.rfft2(blur)
    psf_fft = torch.fft.rfft2(psf)
    psf_fft = torch.conj(psf_fft) / (torch.abs(psf_fft) ** 2 + delta)
    img = torch.fft.ifftshift(torch.fft.irfft2(psf_fft * blur_fft), (-2, -1))
    return img.real


class MultiWiener(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        psf,
        psf_channels=1,
        nc=None,
        pre_process=None,
        skip_pre=False,
    ):
        """
        Parameters
        ----------
        in_channels : int
            Number of input channels. RGB or grayscale, i.e. 3 and 1 respectively.
        out_channels : int
            Number of output channels. RGB or grayscale, i.e. 3 and 1 respectively.

        """
        assert in_channels == 1 or in_channels == 3, "in_channels must be 1 or 3"
        assert out_channels == 1 or out_channels == 3, "out_channels must be 1 or 3"
        if nc is None:
            nc = [64, 128, 256, 512, 512]
        # assert nc[-1] == nc[-2], "Last two channels must be the same"

        super(MultiWiener, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.inc = DoubleConv(in_channels, nc[0])
        self.down_layers = nn.ModuleList([Down(nc[i], nc[i + 1]) for i in range(len(nc) - 1)])
        # self.down1 = Down(64, 128)
        # self.down2 = Down(128, 256)
        # self.down3 = Down(256, 512)
        # self.down4 = Down(512, 512)

        self.up_layers = []
        n_prev = nc[-1]
        for i in range(len(nc) - 1):
            n_in = n_prev + nc[-i - 2]
            n_out = nc[-i - 2] // 2 if i < len(nc) - 2 else nc[0]
            self.up_layers.append(Up(n_in, n_out))
            n_prev = n_out
        self.up_layers = nn.ModuleList(self.up_layers)
        # self.up1 = Up(1024, 256)
        # self.up2 = Up(512, 128)
        # self.up3 = Up(256, 64)
        # self.up4 = Up(128, 64)
        self.outc = OutConv(nc[0], out_channels)

        self.delta = nn.Parameter(torch.tensor(np.ones(5) * 0.01, dtype=torch.float32))
        self.w = nn.Parameter(
            torch.tensor(np.ones((1, psf_channels, 1, 1)) * 0.001, dtype=torch.float32)
        )

        self.inc0 = DoubleConv(psf_channels, nc[0])
        self.psf_down = nn.ModuleList([Down(nc[i], nc[i + 1]) for i in range(len(nc) - 2)])
        # self.down11 = Down(64, 128)
        # self.down22 = Down(128, 256)
        # self.down33 = Down(256, 512)

        # padding H and W to next multiple of 8
        img_shape = psf.shape[-3:-1]
        self.top = (8 - img_shape[0] % 8) // 2
        self.bottom = (8 - img_shape[0] % 8) - self.top
        self.left = (8 - img_shape[1] % 8) // 2
        self.right = (8 - img_shape[1] % 8) - self.left

        self._psf = convert_to_NCHW(psf)
        self._psf = torch.nn.functional.pad(
            self._psf, (self.left, self.right, self.top, self.bottom), mode="constant", value=0
        )
        self._n_iter = 1
        self._convolver = RealFFTConvolve2D(psf, pad=True)

        self.set_pre_process(pre_process)
        self.skip_pre = skip_pre

    def _prepare_process_block(self, process):
        """
        Method for preparing the pre or post process block.
        Parameters
        ----------
            process : :py:class:`function` or :py:class:`~torch.nn.Module`, optional
                Pre or post process block to prepare.
        """
        if isinstance(process, torch.nn.Module):
            # If the post_process is a torch module, we assume it is a DruNet like network.
            from lensless.recon.utils import get_drunet_function_v2

            process_model = process
            process_function = get_drunet_function_v2(process_model, mode="train")
        elif process is not None:
            # Otherwise, we assume it is a function.
            assert callable(process), "pre_process must be a callable function"
            process_function = process
            process_model = None
        else:
            process_function = None
            process_model = None

        if process_function is not None:
            process_param = torch.nn.Parameter(torch.tensor([1.0], device=self._psf.device))
        else:
            process_param = None

        return process_function, process_model, process_param

    def set_pre_process(self, pre_process):
        (
            self.pre_process,
            self.pre_process_model,
            self.pre_process_param,
        ) = self._prepare_process_block(pre_process)

    def forward(self, batch, psfs=None):

        if psfs is None:
            psf = self._psf.to(batch.device)
        else:
            psf = convert_to_NCHW(psfs).to(batch.device)
            psf = torch.nn.functional.pad(
                psf, (self.left, self.right, self.top, self.bottom), mode="constant", value=0
            )
        n_depth = batch[0].shape[-4]
        if n_depth > 1:
            raise NotImplementedError("3D not implemented yet.")

        # pre process data
        if self.pre_process is not None and not self.skip_pre:
            device_before = batch.device
            batch = self.pre_process(batch, self.pre_process_param)
            batch = batch.to(device_before)

        # pad to multiple of 8
        batch = convert_to_NCHW(batch)
        batch = torch.nn.functional.pad(
            batch, (self.left, self.right, self.top, self.bottom), mode="constant", value=0
        )

        # -- downsample
        x_inter = [self.inc(batch)]
        for i in range(len(self.down_layers)):
            x_inter.append(self.down_layers[i](x_inter[-1]))
        # x1 = self.inc(x)
        # x2 = self.down1(x1)
        # x3 = self.down2(x2)
        # x4 = self.down3(x3)
        # x5 = self.down4(x4)

        # -- multi-scale Wiener filtering
        psf_multi = [self.inc0(self.w * psf)]
        for i in range(len(self.psf_down)):
            psf_multi.append(self.psf_down[i](psf_multi[-1]))
        # psf1 = self.inc0(self.w * psf)
        # psf2 = self.down11(psf1)
        # psf3 = self.down22(psf2)
        # psf4 = self.down33(psf3)
        for i in range(len(psf_multi)):
            x_inter[i] = WieNer(x_inter[i], psf_multi[i], self.delta[i])
        # x4 = WieNer(x4, psf4, self.delta[3])
        # x3 = WieNer(x3, psf3, self.delta[2])
        # x2 = WieNer(x2, psf2, self.delta[1])
        # x1 = WieNer(x1, psf1, self.delta[0])

        # upsample
        batch = self.up_layers[0](x_inter[-1], x_inter[-2])
        for i in range(len(self.up_layers) - 1):
            batch = self.up_layers[i + 1](batch, x_inter[-i - 3])
        # x = self.up1(x5, x4)
        # x = self.up2(x, x3)
        # x = self.up3(x, x2)
        # x = self.up4(x, x1)
        batch = self.outc(batch)

        # back to original shape
        batch = batch[..., self.top : -self.bottom, self.left : -self.right]
        batch = convert_to_NDCHW(batch, n_depth)

        # normalize to [0,1], TODO use sigmoid instead?
        batch = (batch + 1) / 2
        batch = torch.clip(batch, min=0.0)
        return batch

    def reset(self, batch_size=1):
        # no state variables
        return

    def reconstruction_error(self, prediction, lensless):
        convolver = self._convolver
        if not convolver.pad:
            prediction = convolver._pad(prediction)
        Fx = convolver.convolve(prediction)
        Fy = lensless

        if not convolver.pad:
            Fx = convolver._crop(Fx)

        # don't reduce batch dimension
        return torch.sum(torch.sqrt((Fx - Fy) ** 2), dim=(-1, -2, -3, -4)) / np.prod(
            prediction.shape[1:]
        )
