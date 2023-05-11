import numpy as np
import cv2 as cv
from math import sqrt
from sympy.ntheory import isprime, quadratic_residues
from scipy.signal import max_len_seq
from perlin_numpy import generate_perlin_noise_2d


class Mask():

    def __init__(self, sensor_size_px: tuple, 
                 sensor_size_m: tuple, 
                 feature_size: tuple, 
                 distance_sensor: float) -> None:
        
        self.mask = None
        self.sensor_size_px = sensor_size_px
        self.sensor_size_m = sensor_size_m
        self.feature_size = feature_size
        self.distance_sensor = distance_sensor
    
    def shape(self):
        return self.mask.shape



class FlatCam_Mask(Mask):

    def __init__(self, 
                 sensor_size_px: tuple, 
                 sensor_size_m: tuple, 
                 feature_size: tuple, 
                 distance_sensor: float, 
                 method: str, 
                 n_bits: int) -> None:
        
        super().__init__(sensor_size_px, sensor_size_m, feature_size, distance_sensor)
        self.method = method
        self.n_bits = n_bits

        assert method in ['MURA', 'MLS']
        if method == 'MURA':
            self.mask = self.squarepattern(4*n_bits+1)
        else:
            seq = max_len_seq(n_bits)[0] * 2 - 1
            h_r = np.r_[seq, seq]
            self.mask = (np.outer(h_r, h_r) + 1) / 2


    def is_prime(self, n):
        if n % 2 == 0 and n > 2: 
            return False
        return all(n % i for i in range(3, int(sqrt(n)) + 1, 2))
    

    def squarepattern(self, p):
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



class PhlatCam_Mask(Mask):

    def __init__(self, 
                 sensor_size_px: tuple, 
                 sensor_size_m: tuple, 
                 feature_size: tuple, 
                 distance_sensor: float, 
                 noise_period: tuple) -> None:
        
        super().__init__(sensor_size_px, sensor_size_m, feature_size, distance_sensor)
        self.noise_period = noise_period

        noise = generate_perlin_noise_2d(sensor_size_px, noise_period)
        sqrt_noise = abs(noise) ** 0.5 * np.sign(noise)
        sqrt_noise_as_img = np.interp(sqrt_noise, (-1,1), (0,255)).astype(np.uint8)
        self.mask = cv.Canny(sqrt_noise_as_img,0,255)



class FZA_Mask(Mask):

    def __init__(self, 
                 sensor_size_px: tuple, 
                 sensor_size_m: tuple, 
                 feature_size: tuple, 
                 distance_sensor: float, 
                 radius: tuple) -> None:
        
        super().__init__(sensor_size_px, sensor_size_m, feature_size, distance_sensor)
        self.radius = radius
        self.mask = self.FZA(sensor_size_px, radius)


    def FZA(self, dim, r):
        x, y = np.meshgrid(np.linspace(-dim[0]/2, dim[0]/2-1, dim[0]), np.linspace(-dim[1]/2, dim[1]/2-1, dim[1]))
        mask = 0.5 * (1 + np.cos(np.pi * (x**2 + y**2) / r**2))
        return mask