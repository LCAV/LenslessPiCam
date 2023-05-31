import abc
import progressbar
import numpy as np
import cv2 as cv
from math import sqrt
from perlin_numpy import generate_perlin_noise_2d
from sympy.ntheory import quadratic_residues
from scipy.signal import max_len_seq
from scipy.ndimage import zoom
from waveprop.fresnel import fresnel_conv
from waveprop.rs import angular_spectrum


class Mask(abc.ABC):
    """
    Parent Mask class
    """

    def __init__(self, sensor_size_px: tuple, 
                 sensor_size_m: tuple, 
                 feature_size: tuple, 
                 distance_sensor: float, 
                 wavelength=532e-9):
        """
        Parent mask contructor.
        Attributes common to each type of mask.

        Parameters
        ----------
        sensor_size_px: tuple (dim=2)
            size of the sensor (px)
        sensor_size_m: tuple (dim=2)
            size of the sensor (m)
        feature_size: float (tuple ?)
            TODO, make it work for tuples (cf fresnel_conv)
            size of the feature (m)
        distance_sensor: float
            distance between the mask and the sensor (m)
        wavelength: float, optional
            wavelength to simulate PSF (m)
        """
        
        self.mask = None
        self.psf = None
        self.phase_mask = None
        self.sensor_size_px = sensor_size_px
        self.sensor_size_m = sensor_size_m
        self.feature_size = feature_size
        self.distance_sensor = distance_sensor
        self.wavelength = wavelength
        self.create_mask()
        self.compute_psf()
    

    @abc.abstractmethod
    def create_mask(self):
        """
        Abstract mask creation method.
        Creating mask with subclass-specific function.
        """
        pass


    def compute_psf(self):
        """
        Computing the PSF.
        Common to all types of masks.
        """
        self.psf, _, _ = angular_spectrum(
            u_in=self.mask,
            wv=self.wavelength,
            d1=self.feature_size,
            dz=self.distance_sensor,
            dtype=np.float32,
            bandlimit=True
        )


class CodedAperture(Mask):
    """
    Coded aperture subclass of the Mask class
    From the FlatCam article https://arxiv.org/abs/1509.00116
    """

    def __init__(self, method='MLS', n_bits=8, **kwargs):
        """
        Coded aperture mask contructor (FlatCam).

        Parameters
        ----------
        method: str
            pattern generation method (MURA or MLS)
            default value: MLS
        n_bits: int
            characteristic number for pattern generation
            size = 4*n_bits + 1 for MURA
                   2^n - 1 for MLS
            default value: 8 (for a 255x255 MLS mask)
        **kwargs: 
            sensor_size_px, 
            sensor_size_m, ``
            feature_size, 
            distance_sensor, 
            wavelength (optional)
        """
        
        self.row = None
        self.column = None
        self.method = method
        self.n_bits = n_bits

        super().__init__(**kwargs)

    def create_mask(self):
        """
        Creating coded aperture mask using either the MURA of MLS method
        """
        assert self.method in ['MURA', 'MLS'], "Method should be either 'MLS' or 'MURA'"
        
        if self.method == 'MURA':
            self.mask = self.squarepattern(4*self.n_bits+1)[1:,1:]
            self.row = 2 * self.mask[1,1:] - 1
            self.column = 2 * self.mask[1:,1] - 1
        else:
            seq = max_len_seq(self.n_bits)[0] * 2 - 1
            h_r = np.r_[seq, seq]
            self.row = h_r
            self.column = h_r
            self.mask = (np.outer(h_r, h_r) + 1) / 2

        if self.sensor_size_px != self.mask.shape:
            upscale_factor_height = self.sensor_size_px[0] / self.mask.shape[0]
            upscale_factor_width = self.sensor_size_px[1] / self.mask.shape[1]
            upscaled_mask = zoom(self.mask, (upscale_factor_height, upscale_factor_width))
            self.mask = np.clip(upscaled_mask, 0, 1)
        


    def is_prime(self, n):
        """
        Assess whether a number is prime or not

        Parameters
        ----------
        n: int
            the number we want to check
        """
        if n % 2 == 0 and n > 2: 
            return False
        return all(n % i for i in range(3, int(sqrt(n)) + 1, 2))
    

    def squarepattern(self, p):
        """
        Generate MURA square pattern

        Parameters
        ----------
        p: int
            number of bits
        """
        if not self.is_prime(p):
            raise ValueError("p is not a valid length. It must be prime.")
        A = np.zeros((p, p), dtype=int)
        q = quadratic_residues(p)
        A[1:, 0] = 1
        for j in range(1, p):
            for i in range(1, p):
                if not ((i - 1 in q) != (j - 1 in q)):
                    A[i, j] = 1
        return A


class PhaseContour(Mask):
    """
    Phase contour subclass of the Mask class
    From the PhlatCam article https://ieeexplore.ieee.org/document/9076617
    """

    def __init__(self, noise_period=(8,8), refractive_index=1.2, n_iter=10, **kwargs):
        """
        Phase contour mask contructor (PhlatCam).

        Parameters
        ----------
        noise_period: tuple (dim=2)
            noise period of the Perlin noise (px)
            default value: (8,8)
        **kwargs: 
            sensor_size_px, 
            sensor_size_m, ``
            feature_size, 
            distance_sensor, 
            wavelength (optional)
        """

        self.target_psf = None
        self.phase_pattern = None
        self.height_map = None
        self.noise_period = noise_period
        self.refractive_index = refractive_index
        self.n_iter = n_iter

        super().__init__(**kwargs)
    

    def create_mask(self):
        """
        Creating phase contour from Perlin noise
        """
        noise = generate_perlin_noise_2d(self.sensor_size_px, self.noise_period)
        sqrt_noise = abs(noise) ** 0.5 * np.sign(noise)
        sqrt_noise_as_img = np.interp(sqrt_noise, (-1,1), (0,255)).astype(np.uint8)
        self.target_psf = cv.Canny(sqrt_noise_as_img, 0, 255)
        phase_mask, height_map = phase_retrieval(
            target_psf=self.target_psf,
            lambd=self.wavelength, 
            d1=self.feature_size,
            dz=self.distance_sensor, 
            n=self.refractive_index, 
            n_iter=self.n_iter, 
            height_map=True
        )
        self.height_map = height_map
        self.phase_pattern = phase_mask
        self.mask = np.exp(1j * phase_mask)


class FresnelZoneAperture(Mask):
    """
    Fresnel Zone Aperture subclass of the Mask class
    From the FZA article https://www.nature.com/articles/s41377-020-0289-9
    """

    def __init__(self, radius=30., **kwargs):
        """
        Fresnel Zone Aperture mask contructor.

        Parameters
        ----------
        radius: float
            characteristic radius of the FZA (px)
            default value: 30
        **kwargs: 
            sensor_size_px, 
            sensor_size_m, ``
            feature_size, 
            distance_sensor, 
            wavelength (optional)
        """

        self.radius = radius
        
        super().__init__(**kwargs)
        

    def create_mask(self):
        """
        Creating Fresnel Zone Aperture mask using either the MURA of MLS method
        """
        dim = self.sensor_size_px
        x, y = np.meshgrid(np.linspace(-dim[0]/2, dim[0]/2-1, dim[0]), np.linspace(-dim[1]/2, dim[1]/2-1, dim[1]))
        self.mask = 0.5 * (1 + np.cos(np.pi * (x**2 + y**2) / self.radius**2))


def phase_retrieval(target_psf, lambd, d1, dz, n=1.2, n_iter=10, height_map=False, pbar=False):
    """
    Iterative phase retrieval algorithm from the PhlatCam article (https://ieeexplore.ieee.org/document/9076617)

    Parameters
    ----------
    lambd: float
        wavelength (m)
    d1: float
        sample period on the sensor i.e. pixel size (m)
    dz: float
        propagation distance between the mask and the sensor
    n: float
        refractive index of the mask substrate
        default value: 1.2
    n_iter: int
        number of iterations
        default value: 10
    """
    M_p = np.sqrt(target_psf)
    if pbar:
        iterator = progressbar.ProgressBar()(range(n_iter))
    else:
        iterator = range(n_iter)
    for _ in iterator:
            # back propagate from sensor to mask
        M_phi = fresnel_conv(M_p, lambd, d1, -dz, dtype=np.float32)[0]
            # constrain amplitude at mask to be unity, i.e. phase pattern
        M_phi = np.exp(1j * np.angle(M_phi))
            # forward propagate from mask to sensor
        M_p = fresnel_conv(M_phi, lambd, d1, dz, dtype=np.float32)[0]
            # constrain amplitude to be sqrt(PSF)
        M_p = np.sqrt(target_psf) * np.exp(1j * np.angle(M_p))

    phi = (np.angle(M_phi) + 2 * np.pi) % (2 * np.pi)

    if height_map:
        return phi, lambd * phi / (2 * np.pi * (n-1))
    else:
        return phi