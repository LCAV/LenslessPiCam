import re
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F


def zero_padding_tensor(images, final_size):
    # -- pad rest with zeros
    # negative cropping = center crop

    padding = np.array(final_size[-2:]) - np.array(images[0].shape[-2:])
    left = padding[1] // 2
    right = padding[1] - left
    top = padding[0] // 2
    bottom = padding[0] - top
    padder = torch.nn.ConstantPad2d((left, right, top, bottom), 0.0)

    return padder(images)


def project_onto_l1_ball(x, eps):
    """
    See: https://gist.github.com/tonyduan/1329998205d88c566588e57e3e2c0c55
    """
    original_shape = x.shape
    x = x.view(x.shape[0], -1)
    mask = (torch.norm(x, p=1, dim=1) < eps).float().unsqueeze(1)
    mu, _ = torch.sort(torch.abs(x), dim=1, descending=True)
    cumsum = torch.cumsum(mu, dim=1)
    arange = torch.arange(1, x.shape[1] + 1, device=x.device)
    rho, _ = torch.max((mu * arange > (cumsum - eps)) * arange, dim=1)
    theta = (cumsum[torch.arange(x.shape[0]), rho.cpu() - 1] - eps) / rho
    proj = (torch.abs(x) - theta.unsqueeze(1)).clamp(min=0)
    x = mask * x + (1 - mask) * proj * torch.sign(x)
    return x.view(original_shape)


class BicubicDownSample(nn.Module):
    def bicubic_kernel(self, x, a=-0.50):
        """
        This equation is exactly copied from the website below:
        https://clouard.users.greyc.fr/Pantheon/experiments/rescaling/index-en.html#bicubic
        """
        abs_x = torch.abs(x)
        if abs_x <= 1.0:
            return (a + 2.0) * torch.pow(abs_x, 3.0) - (a + 3.0) * torch.pow(abs_x, 2.0) + 1
        elif 1.0 < abs_x < 2.0:
            return (
                a * torch.pow(abs_x, 3)
                - 5.0 * a * torch.pow(abs_x, 2.0)
                + 8.0 * a * abs_x
                - 4.0 * a
            )
        else:
            return 0.0

    def __init__(self, factor=4, cuda=True, padding="reflect"):
        super().__init__()
        self.factor = factor
        size = factor * 4
        k = torch.tensor(
            [
                self.bicubic_kernel((i - torch.floor(torch.tensor(size / 2)) + 0.5) / factor)
                for i in range(size)
            ],
            dtype=torch.float32,
        )
        k = k / torch.sum(k)
        # k = torch.einsum('i,j->ij', (k, k))
        k1 = torch.reshape(k, shape=(1, 1, size, 1))
        self.k1 = torch.cat([k1, k1, k1], dim=0)
        k2 = torch.reshape(k, shape=(1, 1, 1, size))
        self.k2 = torch.cat([k2, k2, k2], dim=0)
        self.cuda = ".cuda" if cuda else ""
        self.padding = padding
        # self.padding = 'constant'
        # self.padding = 'replicate'
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, nhwc=False, clip_round=False, byte_output=False):
        filter_height = self.factor * 4
        filter_width = self.factor * 4
        stride = self.factor
        pad_along_height = max(filter_height - stride, 0)
        pad_along_width = max(filter_width - stride, 0)
        filters1 = self.k1.type("torch{}.FloatTensor".format(self.cuda))
        filters2 = self.k2.type("torch{}.FloatTensor".format(self.cuda))
        # compute actual padding values for each side
        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left
        # apply mirror padding
        if nhwc:
            x = torch.transpose(torch.transpose(x, 2, 3), 1, 2)  # NHWC to NCHW
        # downscaling performed by 1-d convolution
        x = F.pad(x, (0, 0, pad_top, pad_bottom), self.padding)
        x = F.conv2d(input=x.float(), weight=filters1, stride=(stride, 1), groups=3)
        if clip_round:
            x = torch.clamp(torch.round(x), 0.0, 255.0)
        x = F.pad(x, (pad_left, pad_right, 0, 0), self.padding)
        x = F.conv2d(input=x, weight=filters2, stride=(1, stride), groups=3)
        if clip_round:
            x = torch.clamp(torch.round(x), 0.0, 255.0)
        if nhwc:
            x = torch.transpose(torch.transpose(x, 1, 3), 1, 2)
        if byte_output:
            return x.type("torch.ByteTensor")
        else:
            return x


# Utils
def get_array(file):
    img = np.array(Image.open(file).convert("RGB"))
    img = img / 255
    return img


def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval


def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    float regex comes from https://stackoverflow.com/a/12643073/190597
    """
    return [atof(c) for c in re.split(r"[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)", text)]
