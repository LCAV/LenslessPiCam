import numpy as np
from lensless.hardware.mask import CodedAperture, PhaseContour, FresnelZoneAperture, HeightVarying, MultiLensArray
from lensless.eval.metric import mse, psnr, ssim
from waveprop.fresnel import fresnel_conv
from matplotlib import pyplot as plt
from lensless.hardware.trainable_mask import TrainableMask
import torch 

resolution = np.array([380, 507])
d1 = 3e-6
dz = 4e-3


def test_flatcam():

    mask1 = CodedAperture(
        method="MURA",
        n_bits=25,
        resolution=resolution,
        feature_size=d1,
        distance_sensor=dz,
    )
    assert np.all(mask1.mask.shape == resolution)

    desired_psf_shape = np.array(tuple(resolution) + (len(mask1.psf_wavelength),))
    assert np.all(mask1.psf.shape == desired_psf_shape)

    mask2 = CodedAperture(
        method="MLS",
        n_bits=5,
        resolution=resolution,
        feature_size=d1,
        distance_sensor=dz,
    )
    assert np.all(mask2.mask.shape == resolution)
    desired_psf_shape = np.array(tuple(resolution) + (len(mask2.psf_wavelength),))
    assert np.all(mask2.psf.shape == desired_psf_shape)

"""
def test_phlatcam():

    mask = PhaseContour(
        noise_period=(8, 8),
        resolution=resolution,
        feature_size=d1,
        distance_sensor=dz,
    )
    assert np.all(mask.mask.shape == resolution)
    desired_psf_shape = np.array(tuple(resolution) + (len(mask.psf_wavelength),))
    assert np.all(mask.psf.shape == desired_psf_shape)

    Mp = np.sqrt(mask.target_psf) * np.exp(
        1j * np.angle(fresnel_conv(mask.mask, mask.design_wv, d1, dz, dtype=np.float32)[0])
    )
    assert mse(abs(Mp), np.sqrt(mask.target_psf)) < 0.1
    assert psnr(abs(Mp), np.sqrt(mask.target_psf)) > 30
    assert abs(1 - ssim(abs(Mp), np.sqrt(mask.target_psf), channel_axis=None)) < 0.1
"""

def test_fza():

    mask = FresnelZoneAperture(
        radius=30.0, resolution=resolution, feature_size=d1, distance_sensor=dz
    )
    assert np.all(mask.mask.shape == resolution)
    desired_psf_shape = np.array(tuple(resolution) + (len(mask.psf_wavelength),))
    assert np.all(mask.psf.shape == desired_psf_shape)


def test_classmethod():

    downsample = 8

    """mask1 = CodedAperture.from_sensor(
        sensor_name="rpi_hq", downsample=downsample, distance_sensor=dz
    )
    assert np.all(mask1.mask.shape == resolution)
    desired_psf_shape = np.array(tuple(resolution) + (len(mask1.psf_wavelength),))
    assert np.all(mask1.psf.shape == desired_psf_shape)"""
    """mask2 = PhaseContour.from_sensor(
        sensor_name="rpi_hq", downsample=downsample, distance_sensor=dz
    )
    assert np.all(mask2.mask.shape == resolution)
    desired_psf_shape = np.array(tuple(resolution) + (len(mask2.psf_wavelength),))
    assert np.all(mask2.psf.shape == desired_psf_shape)

    mask3 = FresnelZoneAperture.from_sensor(
        sensor_name="rpi_hq", downsample=downsample, distance_sensor=dz
    )
    assert np.all(mask3.mask.shape == resolution)
    desired_psf_shape = np.array(tuple(resolution) + (len(mask3.psf_wavelength),))
    assert np.all(mask3.psf.shape == desired_psf_shape)
    """
    mask4 = MultiLensArray.from_sensor(
        sensor_name="rpi_hq", downsample=downsample, distance_sensor=dz, N=10, is_Torch=True#radius=np.array([10, 25]), loc=np.array([[10.1, 11.3], [56.5, 89.2]])
    )

    phase = None
    if not mask4.is_torch:
        assert np.all(mask4.mask.shape == resolution)
        desired_psf_shape = np.array(tuple(resolution) + (len(mask4.psf_wavelength),))
        assert np.all(mask4.psf.shape == desired_psf_shape)
        phase = mask4.phi
    else:
        # PyTorch operations
        assert torch.equal(torch.tensor(mask4.mask.shape), torch.tensor(resolution))
        desired_psf_shape = torch.tensor(tuple(resolution) + (len(mask4.psf_wavelength),))
        assert torch.equal(torch.tensor(mask4.psf.shape), desired_psf_shape)
        angle=torch.angle(mask4.mask).cpu().detach().numpy()
    fig, ax = plt.subplots()
    im = ax.imshow(mask4.phi, cmap="gray")
    fig.colorbar(im, ax=ax, shrink=0.5, aspect=5)
    plt.show()
    """
    
    mask5 = HeightVarying.from_sensor(
        sensor_name="rpi_hq", downsample=downsample, distance_sensor=dz, is_Torch=False
    )
    #assert mask5.is_Torch
    if not mask5.is_torch:
        # NumPy operations
        assert np.all(mask5.mask.shape == resolution)
        desired_psf_shape = np.array(tuple(resolution) + (len(mask5.psf_wavelength),))
        assert np.all(mask5.psf.shape == desired_psf_shape)
        fig, ax = plt.subplots()
        im = ax.imshow(np.angle(mask5.mask), cmap="gray")
        fig.colorbar(im, ax=ax, shrink=0.5, aspect=5)
        plt.show()
    else:
        # PyTorch operations
        assert torch.equal(torch.tensor(mask5.mask.shape), torch.tensor(resolution))
        desired_psf_shape = torch.tensor(tuple(resolution) + (len(mask5.psf_wavelength),))
        assert torch.equal(torch.tensor(mask5.psf.shape), desired_psf_shape)
        fig, ax = plt.subplots()
        im = ax.imshow(torch.angle(mask5.mask), cmap="gray")
        fig.colorbar(im, ax=ax, shrink=0.5, aspect=5)
        plt.show()"""
    


if __name__ == "__main__":
##    test_flatcam()
##    test_phlatcam()
##    test_fza()
    test_classmethod()
