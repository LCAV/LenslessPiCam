import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from lensless.mask import CodedAperture, PhaseContour, FresnelZoneAperture
from lensless.metric import mse, psnr, ssim
from waveprop.fresnel import fresnel_conv



lambd, sensor_size, nb_px, dz = 532e-9, 5e-3, 256, 0.5e-3
d1 = sensor_size / nb_px


def test_flatcam():

    mask1 = CodedAperture(method='MURA', 
                          n_bits=25, 
                          sensor_size_px=(101,101), 
                          sensor_size_m=None, 
                          feature_size=d1, 
                          distance_sensor=dz, 
                          wavelength=lambd)
    assert mask1.mask.shape == (101, 101)

    mask2 = CodedAperture(method='MLS', 
                          n_bits=5, 
                          sensor_size_px=(62,62), 
                          sensor_size_m=None, 
                          feature_size=d1, 
                          distance_sensor=dz, 
                          wavelength=lambd)
    assert mask2.mask.shape == (62, 62)


def test_phlatcam():
    
    mask = PhaseContour(noise_period=(8,8),  
                         sensor_size_px=(256,256), 
                         sensor_size_m=None, 
                         feature_size=d1, 
                         distance_sensor=dz, 
                         wavelength=lambd)
    assert mask.mask.shape == (256, 256)

    Mp = np.sqrt(mask.target_psf) * np.exp(1j * np.angle(fresnel_conv(mask.mask, lambd, d1, dz, dtype=np.float32)[0]))
    assert mse(abs(Mp), np.sqrt(mask.target_psf)) < 0.1
    assert psnr(abs(Mp), np.sqrt(mask.target_psf)) > 30
    assert abs(1 - ssim(abs(Mp), np.sqrt(mask.target_psf), channel_axis=None)) < 0.1


def test_fza():

    mask = FresnelZoneAperture(radius=30.,  
                                sensor_size_px=(512,512), 
                                sensor_size_m=None, 
                                feature_size=d1, 
                                distance_sensor=dz, 
                                wavelength=lambd)
    assert mask.mask.shape == (512, 512)


test_flatcam()
test_phlatcam()
test_fza()