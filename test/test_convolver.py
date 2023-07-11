from lensless.recon.rfft_convolve import RealFFTConvolve2D
import numpy as np
import torch

psf_fp = "data/psf/tape_rgb.png"
data_fp = "data/raw_data/thumbs_up_rgb.png"
downsample = 8


# test if padding and cropping works
def test_pad_np():
    psf = np.random.rand(5, 47, 29, 3)
    convolver = RealFFTConvolve2D(psf, pad=True)
    data = np.random.rand(12, 5, 47, 29, 3)
    padded = convolver._pad(data)
    np.testing.assert_array_equal(padded.shape[1:], convolver._padded_shape)
    cropped = convolver._crop(padded)
    np.testing.assert_array_equal(cropped, data)


def test_pad_torch():
    psf = torch.rand(5, 47, 29, 3)
    convolver = RealFFTConvolve2D(psf, pad=True)
    data = torch.rand(12, 5, 47, 29, 3)
    padded = convolver._pad(data)
    np.testing.assert_array_equal(padded.shape[1:], convolver._padded_shape)
    cropped = convolver._crop(padded)
    assert torch.equal(cropped, data)


# test if convolution works identically on each color channel
def test_conv_np():
    psf = np.random.rand(5, 47, 29, 3)
    convolver = RealFFTConvolve2D(psf, pad=True)
    data = np.random.rand(12, 5, 47, 29, 3)
    all_convolved = convolver.convolve(data)
    part_convolved = convolver.convolve(data[:1, :1, :, :, :1])
    np.testing.assert_almost_equal(
        all_convolved[:1, :1, :, :, :1], part_convolved[:1, :1, :, :, :1], decimal=5
    )


def test_conv_torch():
    psf = torch.rand(5, 47, 29, 3)
    convolver = RealFFTConvolve2D(psf, pad=True)
    data = torch.rand(12, 1, 47, 29, 3)
    all_convolved = convolver.convolve(data)
    part_convolved = convolver.convolve(data[:1, :1, :, :, :])
    torch.testing.assert_close(all_convolved[:1, :1, :, :, :], part_convolved[:1, :1, :, :, :])


def test_conv_torch_2():
    psf = torch.rand(1, 47, 29, 3)
    convolver = RealFFTConvolve2D(psf, pad=True)
    data = torch.rand(12, 1, 47, 29, 3)
    all_convolved = convolver.convolve(data)
    part_convolved = convolver.convolve(data[:1, :1, :, :, :])
    print(all_convolved.shape, part_convolved.shape)
    torch.testing.assert_close(all_convolved[:1, :1, :, :, :], part_convolved[:1, :1, :, :, :])
