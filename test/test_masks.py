import numpy as np
from lensless.hardware.mask import CodedAperture, PhaseContour, FresnelZoneAperture
from lensless.eval.metric import mse, psnr, ssim
from waveprop.fresnel import fresnel_conv


resolution = np.array([380, 507])
d1 = 3e-6
dz = 4e-3


def test_flatcam():

    mask1 = CodedAperture(
        method="MURA",
        n_bits=23,
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


def test_phlatcam():

    mask = PhaseContour(
        noise_period=(8, 8),
        resolution=resolution,
        feature_size=d1,
        distance_sensor=dz,
    )
    assert np.all(mask.shape == resolution)
    desired_psf_shape = np.array(tuple(resolution) + (len(mask.psf_wavelength),))
    assert np.all(mask.psf.shape == desired_psf_shape)

    u1 = mask.height_map_to_field(wavelength=mask.design_wv)
    Mp = np.sqrt(mask.target_psf) * np.exp(
        1j * np.angle(fresnel_conv(u1, mask.design_wv, d1, dz, dtype=np.float32)[0])
    )
    assert mse(abs(Mp), np.sqrt(mask.target_psf)) < 0.1
    assert psnr(abs(Mp), np.sqrt(mask.target_psf)) > 30
    assert abs(1 - ssim(abs(Mp), np.sqrt(mask.target_psf), channel_axis=None)) < 0.1


def test_fza():

    mask = FresnelZoneAperture(
        radius=30.0, resolution=resolution, feature_size=d1, distance_sensor=dz
    )
    assert np.all(mask.mask.shape == resolution)
    desired_psf_shape = np.array(tuple(resolution) + (len(mask.psf_wavelength),))
    assert np.all(mask.psf.shape == desired_psf_shape)


def test_classmethod():

    downsample = 8

    mask1 = CodedAperture.from_sensor(
        sensor_name="rpi_hq", downsample=downsample, distance_sensor=dz
    )
    assert np.all(mask1.shape == resolution)
    desired_psf_shape = np.array(tuple(resolution) + (len(mask1.psf_wavelength),))
    assert np.all(mask1.psf.shape == desired_psf_shape)

    mask2 = PhaseContour.from_sensor(
        sensor_name="rpi_hq", downsample=downsample, distance_sensor=dz
    )
    assert np.all(mask2.shape == resolution)
    desired_psf_shape = np.array(tuple(resolution) + (len(mask2.psf_wavelength),))
    assert np.all(mask2.psf.shape == desired_psf_shape)

    mask3 = FresnelZoneAperture.from_sensor(
        sensor_name="rpi_hq", downsample=downsample, distance_sensor=dz
    )
    assert np.all(mask3.shape == resolution)
    desired_psf_shape = np.array(tuple(resolution) + (len(mask3.psf_wavelength),))
    assert np.all(mask3.psf.shape == desired_psf_shape)


if __name__ == "__main__":
    test_flatcam()
    test_phlatcam()
    test_fza()
    test_classmethod()
