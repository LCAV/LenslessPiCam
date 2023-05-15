import abc
import progressbar
import numpy as np
import cv2 as cv
from math import sqrt
from perlin_numpy import generate_perlin_noise_2d
from sympy.ntheory import quadratic_residues
from scipy.signal import max_len_seq
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
                 wavelength: float) -> None:
        """
        Parent mask contructor.
        Attributes common to each type of mask.

        Parameters
        ----------
        sensor_size_px: tuple (dim=2)
            size of the sensor (px)
        sensor_size_m: tuple (dim=2)
            size of the sensor (m)
        feature_size: float
            size of the feature (m)
        distance_sensor: float
            distance between the mask and the sensor (m)
        wavelength: float
            wavelength (m)
        """
        
        self.mask = None
        self.psf = None
        self.phase_mask = None
        self.height_map = None
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
            dtype=np.float32
        )
        pass
    

    def shape(self):
        """
        Shape of the mask.
        """
        return self.mask.shape
    

    def phase_retrieval(self, lambd, d1, dz, n=1.5, n_iter=10):
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
        n_iter: int
            number of iterations
        """
        M_p = np.sqrt(self.psf)
        for _ in progressbar.ProgressBar()(range(n_iter)):
            M_phi = np.exp(1j * np.angle(fresnel_conv(M_p, lambd, d1, -dz, dtype=np.float32)[0]))
            M_p = np.sqrt(self.psf) * np.exp(1j * fresnel_conv(M_phi, lambd, d1, dz, dtype=np.float32)[0])    
        phi = np.angle(M_phi)

        self.phase_mask = phi
        self.height_map = lambd * phi / (2 * np.pi * (n-1))
        pass





class CodedAperture(Mask):
    """
    Coded aperture subclass of the Mask class
    From the FlatCam article https://arxiv.org/abs/1509.00116
    """

    def __init__(self, 
                 sensor_size_px: tuple, 
                 sensor_size_m: tuple, 
                 feature_size: tuple, 
                 distance_sensor: float,
                 wavelength: float, 
                 method: str, 
                 n_bits: int) -> None:
        """
        Coded aperture mask contructor (FlatCam).

        Parameters
        ----------
        sensor_size_px: tuple (dim=2)
            size of the sensor (px)
        sensor_size_m: tuple (dim=2)
            size of the sensor (m)
        feature_size: float
            size of the feature (m)
        distance_sensor: float
            distance between the mask and the sensor (m)
        wavelength: float
            wavelength (m)
        method: str
            pattern generation method (MURA or MLS)
        n_bits: int
            characteristic number for pattern generation
            size = 4*n_bits + 1 for MURA
                   2^n - 1 for MLS
        """
        
        self.method = method
        self.n_bits = n_bits

        super().__init__(sensor_size_px, sensor_size_m, feature_size, distance_sensor, wavelength)


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
    Phase contour subclass of the Mask class
    From the PhlatCam article https://ieeexplore.ieee.org/document/9076617
    """

    def __init__(self, 
                 sensor_size_px: tuple, 
                 sensor_size_m: tuple, 
                 feature_size: tuple, 
                 distance_sensor: float,
                 wavelength: float, 
                 noise_period: tuple) -> None:
        """
        Coded aperture mask contructor (FlatCam).

        Parameters
        ----------
        sensor_size_px: tuple (dim=2)
            size of the sensor (px)
        sensor_size_m: tuple (dim=2)
            size of the sensor (m)
        feature_size: float
            size of the feature (m)
        distance_sensor: float
            distance between the mask and the sensor (m)
        wavelength: float
            wavelength (m)
        noise_period: tuple (dim=2)
            noise period of the Perlin noise (px)
        """

        self.noise_period = noise_period

        super().__init__(sensor_size_px, sensor_size_m, feature_size, distance_sensor, wavelength)
    

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
    Fresnel Zone Aperture subclass of the Mask class
    From the FZA article https://www.nature.com/articles/s41377-020-0289-9
    """

    def __init__(self, 
                 sensor_size_px: tuple, 
                 sensor_size_m: tuple, 
                 feature_size: tuple, 
                 distance_sensor: float, 
                 wavelength: float, 
                 radius: float) -> None:
        """
        Fresnel Zone Aperture mask contructor.

        Parameters
        ----------
        sensor_size_px: tuple (dim=2)
            size of the sensor (px)
        sensor_size_m: tuple (dim=2)
            size of the sensor (m)
        feature_size: float
            size of the feature (m)
        distance_sensor: float
            distance between the mask and the sensor (m)
        wavelength: float
            wavelength (m)
        radius: float
            characteristic radius of the FZA (px)
        """

        self.radius = radius
        
        super().__init__(sensor_size_px, sensor_size_m, feature_size, distance_sensor, wavelength)
        

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
            mask dimension (px)
        r: float
            characteristic radius of the FZA (px)
        """
        x, y = np.meshgrid(np.linspace(-dim[0]/2, dim[0]/2-1, dim[0]), np.linspace(-dim[1]/2, dim[1]/2-1, dim[1]))
        mask = 0.5 * (1 + np.cos(np.pi * (x**2 + y**2) / r**2))
        return mask