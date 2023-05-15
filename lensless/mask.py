import numpy as np
import cv2 as cv
import abc
from math import sqrt
from sympy.ntheory import isprime, quadratic_residues
from scipy.signal import max_len_seq
from perlin_numpy import generate_perlin_noise_2d


class Mask(abc.ABC):
    """
    Parent Mask class
    """

    def __init__(self, sensor_size_px: tuple, 
                 sensor_size_m: tuple, 
                 feature_size: tuple, 
                 distance_sensor: float) -> None:
        """
        Parent mask contructor.
        Attributes common to each type of mask.

        Parameters
        ----------
        sensor_size_px: tuple (dim=2)
            size of the sensor in pixels
        sensor_size_m: tuple (dim=2)
            size of the sensor in meters
        feature_size: tuple (dim=2)
            size of the feature in meters
        distance_sensor: float
            distance between the mask and the sensor
        """
        
        self.mask = None
        self.psf = None
        self.sensor_size_px = sensor_size_px
        self.sensor_size_m = sensor_size_m
        self.feature_size = feature_size
        self.distance_sensor = distance_sensor
        self.create_mask()
    
    @abc.abstractmethod
    def create_mask(self):
        """
        Abstract mask creation method.
        Creating mask with subclass-specific function.
        """
        pass

    def compute_psf():
        pass
    
    def shape(self):
        """
        Shape of the mask.
        """
        return self.mask.shape



class CodedAperture(Mask):
    """
    https://arxiv.org/abs/1509.00116
    """

    def __init__(self, 
                 sensor_size_px: tuple, 
                 sensor_size_m: tuple, 
                 feature_size: tuple, 
                 distance_sensor: float, 
                 method: str, 
                 n_bits: int) -> None:
        """
        Coded aperture mask contructor (FlatCam).

        Parameters
        ----------
        sensor_size_px: tuple (dim=2)
            size of the sensor in pixels
        sensor_size_m: tuple (dim=2)
            size of the sensor in meters
        feature_size: tuple (dim=2)
            size of the feature in meters
        distance_sensor: float
            distance between the mask and the sensor
        method: str
            pattern generation method (MURA or MLS)
        n_bits: int
            characteristic number for pattern generation
            size = 4*n_bits + 1 for MURA
                   2^n - 1 for MLS
        """
        
        self.method = method
        self.n_bits = n_bits

        super().__init__(sensor_size_px, sensor_size_m, feature_size, distance_sensor)

    def create_mask(self):
        """
        Creating coded aperture mask using either the MURA of MLS method
        """
        assert self.method in ['MURA', 'MLS']
        if self.method == 'MURA':
            self.mask = self.squarepattern(4*self.n_bits+1)
        else:
            seq = max_len_seq(self.n_bits)[0] * 2 - 1
            h_r = np.r_[seq, seq]
            self.mask = (np.outer(h_r, h_r) + 1) / 2


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
    https://ieeexplore.ieee.org/document/9076617
    """

    def __init__(self, 
                 sensor_size_px: tuple, 
                 sensor_size_m: tuple, 
                 feature_size: tuple, 
                 distance_sensor: float, 
                 noise_period: tuple) -> None:
        """
        Coded aperture mask contructor (FlatCam).

        Parameters
        ----------
        sensor_size_px: tuple (dim=2)
            size of the sensor in pixels
        sensor_size_m: tuple (dim=2)
            size of the sensor in meters
        feature_size: tuple (dim=2)
            size of the feature in meters
        distance_sensor: float
            distance between the mask and the sensor
        noise_period: tuple (dim=2)
            noise period of the Perlin noise
        """

        self.noise_period = noise_period

        super().__init__(sensor_size_px, sensor_size_m, feature_size, distance_sensor)
    
    def create_mask(self):
        """
        Creating coded aperture mask using either the MURA of MLS method
        """
        noise = generate_perlin_noise_2d(self.sensor_size_px, self.noise_period)
        sqrt_noise = abs(noise) ** 0.5 * np.sign(noise)
        sqrt_noise_as_img = np.interp(sqrt_noise, (-1,1), (0,255)).astype(np.uint8)
        self.mask = cv.Canny(sqrt_noise_as_img,0,255)
        pass



class FresnelZoneAperture(Mask):
    """
    https://www.nature.com/articles/s41377-020-0289-9
    """

    def __init__(self, 
                 sensor_size_px: tuple, 
                 sensor_size_m: tuple, 
                 feature_size: tuple, 
                 distance_sensor: float, 
                 radius: float) -> None:
        """
        Fresnel Zone Aperture mask contructor.

        Parameters
        ----------
        sensor_size_px: tuple (dim=2)
            size of the sensor in pixels
        sensor_size_m: tuple (dim=2)
            size of the sensor in meters
        feature_size: tuple (dim=2)
            size of the feature in meters
        distance_sensor: float
            distance between the mask and the sensor
        radius: float
            characteristic radius of the FZA
        """

        self.radius = radius
        
        super().__init__(sensor_size_px, sensor_size_m, feature_size, distance_sensor)
        
    def create_mask(self):
        """
        Creating Fresnel Zone Aperture mask using either the MURA of MLS method
        """
        self.mask = self.FZA(self.sensor_size_px, self.radius)
        pass


    def FZA(self, dim, r):
        """
        Fresnel Zone Aperture function

        Parameters
        ----------
        dim: tuple (dim=2)
            mask dimension
        r: float
            characteristic radius of the FZA
        """
        x, y = np.meshgrid(np.linspace(-dim[0]/2, dim[0]/2-1, dim[0]), np.linspace(-dim[1]/2, dim[1]/2-1, dim[1]))
        mask = 0.5 * (1 + np.cos(np.pi * (x**2 + y**2) / r**2))
        return mask