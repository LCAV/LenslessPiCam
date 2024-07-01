# #############################################################################
# mask.py
# =================
# Authors :
# Aaron FARGEON [aa.fargeon@gmail.com]
# Eric BEZZAM [ebezzam@gmail.com]
# #############################################################################

"""
Mask Design
===========

This module provides utilities to create different types of masks (:py:class:`~lensless.hardware.mask.CodedAperture`,
:py:class:`~lensless.hardware.mask.PhaseContour`,
:py:class:`~lensless.hardware.mask.FresnelZoneAperture`) and simulate the corresponding PSF.

"""


import abc
import warnings
import numpy as np
import cv2 as cv
from math import sqrt
from perlin_numpy import generate_perlin_noise_2d
from sympy.ntheory import quadratic_residues
from scipy.signal import max_len_seq
from scipy.linalg import circulant
from numpy.linalg import multi_dot
from waveprop.fresnel import fresnel_conv
from waveprop.rs import angular_spectrum
from waveprop.noise import add_shot_noise
from lensless.hardware.sensor import VirtualSensor
from lensless.utils.image import resize
import matplotlib.pyplot as plt

try:
    import torch

    torch_available = True
except ImportError:
    torch_available = False


class Mask(abc.ABC):
    """
    Parent ``Mask`` class. Attributes common to each type of mask.
    """

    def __init__(
        self,
        resolution,
        distance_sensor=None,
        size=None,
        feature_size=None,
        psf_wavelength=[460e-9, 550e-9, 640e-9],
        use_torch=False,
        torch_device="cpu",
        centered=True,
        **kwargs,
    ):
        """
        Constructor from parameters of the user's choice.

        Parameters
        ----------
        resolution: array_like
            Resolution of the  mask (px).
        distance_sensor: float
            Distance between the mask and the sensor (m). Needed to simulate PSF.
        size: array_like
            Size of the mask (m). Only one of ``size`` or ``feature_size`` needs to be specified.
        feature_size: float or array_like
            Size of the feature (m). Only one of ``size`` or ``feature_size`` needs to be specified.
        psf_wavelength: list, optional
            List of wavelengths to simulate PSF (m). Default is [460e-9, 550e-9, 640e-9] nm (blue, green, red).
        use_torch : bool, optional
            If True, the mask is created as a torch tensor. Default is False.
        torch_device : str, optional
            Device to use for torch tensor. Default is 'cpu'.
        centered: bool, optional
            If True, the mask is centered. Default is True.
        """

        resolution = np.array(resolution)
        assert len(resolution) == 2, "Sensor resolution should be of length 2"

        assert (
            size is not None or feature_size is not None
        ), "Either sensor_size or feature_size should be specified"
        if size is None:
            size = np.array(resolution * feature_size)
        else:
            size = np.array(size)
            assert len(size) == 2, "Sensor size should be of length 2"
        if feature_size is None:
            feature_size = np.array(size) / np.array(resolution)
        else:
            if isinstance(feature_size, float):
                feature_size = np.array([feature_size, feature_size])
            else:
                assert len(feature_size) == 2, "Feature size should be of length 2"
                feature_size = np.array(feature_size)
            assert np.all(feature_size > 0), "Feature size should be positive"
        assert np.all(resolution * feature_size <= size)

        self.resolution = resolution
        self.resolution = (int(self.resolution[0]), int(self.resolution[1]))
        self.size = size
        self.centered = centered
        if feature_size is None:
            self.feature_size = self.size / self.resolution
        else:
            self.feature_size = feature_size
        self.distance_sensor = distance_sensor

        if use_torch:
            assert torch_available, "PyTorch is not available"
        self.use_torch = use_torch
        self.torch_device = torch_device

        # create mask
        self.height_map = None  # for phase masks
        self.create_mask()  # creates self.mask
        self.shape = self.height_map.shape if self.height_map is not None else self.mask.shape

        # PSF
        assert hasattr(psf_wavelength, "__len__"), "psf_wavelength should be a list"
        self.psf_wavelength = psf_wavelength
        self.psf = None
        if self.distance_sensor is not None:
            self.compute_psf()

    @classmethod
    def from_sensor(cls, sensor_name, downsample=None, **kwargs):
        """
        Constructor from an existing virtual sensor that copies over the sensor parameters
        (sensor resolution, sensor size, feature size).

        Parameters
        ----------
        sensor_name: str
            Name of the sensor. See :py:class:`~lensless.hardware.sensor.SensorOptions`.
        downsample: float, optional
            Downsampling factor.
        **kwargs:
            Additional arguments for the mask constructor. See the abstract class :py:class:`~lensless.hardware.mask.Mask`
            and the corresponding subclass for more details.

        Example
        -------

        .. code-block:: python

            mask = CodedAperture.from_sensor(sensor_name=SensorOptions.RPI_HQ, downsample=8, ...)
        """
        sensor = VirtualSensor.from_name(sensor_name, downsample)
        return cls(
            resolution=tuple(sensor.resolution.copy()),
            size=tuple(sensor.size.copy()),
            feature_size=sensor.pixel_size.copy(),
            **kwargs,
        )

    @abc.abstractmethod
    def create_mask(self):
        """
        Abstract mask creation method that creates mask with subclass-specific function.
        """
        pass

    def height_map_to_field(self, wavelength, return_phase=False):
        """
        Compute phase from height map.

        Parameters
        ----------
        height_map: :py:class:`~numpy.ndarray`
            Height map.
        wavelength: float
            Wavelength of the light (m).
        return_phase: bool, optional
            If True, return the phase instead of the field. Default is False.
        """
        assert self.height_map is not None, "Height map should be computed first."
        assert self.refractive_index is not None, "Refractive index should be specified."

        phase_pattern = self.height_map * (self.refractive_index - 1) * 2 * np.pi / wavelength
        if return_phase:
            return phase_pattern
        else:
            return (
                np.exp(1j * phase_pattern) if not self.use_torch else torch.exp(1j * phase_pattern)
            )

    def compute_psf(self, distance_sensor=None, wavelength=None, intensity=True):
        """
        Compute the intensity PSF with bandlimited angular spectrum (BLAS) for each wavelength.
        Common to all types of masks.

        Parameters
        ----------
        distance_sensor: float, optional
            Distance between mask and sensor (m). Default is the distance specified at initialization.
        wavelength: float or array_like, optional
            Wavelength(s) to compute the PSF (m). Default is the list of wavelengths specified at initialization.
        """
        if distance_sensor is not None:
            self.distance_sensor = distance_sensor
        assert (
            self.distance_sensor is not None
        ), "Distance between mask and sensor should be specified."

        if wavelength is None:
            wavelength = self.psf_wavelength
        else:
            if not hasattr(wavelength, "__len__"):
                wavelength = [wavelength]

        if self.use_torch:
            psf = torch.zeros(
                tuple(self.resolution) + (len(self.psf_wavelength),),
                dtype=torch.complex64,
                device=self.torch_device,
            )
        else:
            psf = np.zeros(tuple(self.resolution) + (len(wavelength),), dtype=np.complex64)
        for i, wv in enumerate(wavelength):
            psf[:, :, i] = angular_spectrum(
                u_in=self.mask if self.height_map is None else self.height_map_to_field(wv),
                wv=wv,
                d1=self.feature_size,
                dz=self.distance_sensor,
                dtype=np.float32 if not self.use_torch else torch.float32,
                bandlimit=True,
                device=self.torch_device if self.use_torch else None,
            )[0]

        # intensity PSF
        if intensity:
            self.psf = np.abs(psf) ** 2 if not self.use_torch else torch.abs(psf) ** 2
        else:
            self.psf = psf

        return self.psf

    def plot(self, ax=None, **kwargs):
        """
        Plot the mask.

        Parameters
        ----------
        ax: :py:class:`~matplotlib.axes.Axes`, optional
            Axes to plot the mask on. Default is None.
        **kwargs:
            Additional arguments for the plot function.
        """

        if ax is None:
            _, ax = plt.subplots()

        if self.height_map is not None:
            mask = self.height_map
            title = "Height map"
        else:
            mask = self.mask
            title = "Amplitude mask"
        if self.use_torch:
            mask = mask.cpu().numpy()

        if self.centered:
            extent = (
                -self.size[1] / 2 * 1e3,
                self.size[1] / 2 * 1e3,
                self.size[0] / 2 * 1e3,
                -self.size[0] / 2 * 1e3,
            )
        else:
            extent = (0, self.size[1] * 1e3, self.size[0] * 1e3, 0)

        ax.imshow(mask, extent=extent, cmap="gray", **kwargs)
        ax.set_title(title)
        ax.set_xlabel("[mm]")
        ax.set_ylabel("[mm]")
        return ax


class CodedAperture(Mask):
    """
    Coded aperture mask as in `FlatCam <https://arxiv.org/abs/1509.00116>`_.
    """

    def __init__(self, method="MLS", n_bits=8, **kwargs):
        """
        Coded aperture mask contructor (FlatCam).

        Parameters
        ----------
        method: str
            Pattern generation method (MURA or MLS). Default is ``MLS``.
        n_bits: int, optional
            Number of bits for pattern generation, must be prime for MURA.
            Results in ``2^n - 1``x``2^n - 1`` for MLS.
            Default is 8 (for a 255x255 MLS mask).
        **kwargs:
            The keyword arguments are passed to the parent class :py:class:`~lensless.hardware.mask.Mask`.
        """

        self.row = None
        self.col = None
        self.method = method
        self.n_bits = n_bits

        assert self.method.upper() in ["MURA", "MLS"], "Method should be either 'MLS' or 'MURA'"
        # TODO? use: https://github.com/bpops/codedapertures

        # initialize parameters
        if self.method.upper() == "MURA":
            self.mask = self.generate_mura(self.n_bits)
            self.row = None
            self.col = None
        else:
            seq = max_len_seq(self.n_bits)[0] * 2 - 1
            self.row = seq
            self.col = seq

        if "is_torch" in kwargs and kwargs["is_torch"]:
            torch_device = kwargs["torch_device"] if "torch_device" in kwargs else "cpu"
            if self.row is not None and self.col is not None:
                self.row = torch.from_numpy(self.row).float().to(torch_device)
                self.col = torch.from_numpy(self.col).float().to(torch_device)
            else:
                self.mask = torch.from_numpy(self.mask).float().to(torch_device)

        # needs to be done at the end as it calls create_mask
        super().__init__(**kwargs)

    def create_mask(self, row=None, col=None, mask=None):
        """
        Creating coded aperture mask.
        """

        if mask is not None:
            self.mask = mask
            assert row is None and col is None, "Row and col should not be specified"

        elif row is not None:
            assert col is not None, "Both row and col should be specified"
            self.row = row
            self.col = col

        # output product if necessary
        if self.row is not None:
            if self.use_torch:
                self.mask = torch.outer(self.row, self.col)
                self.mask = torch.round((self.mask + 1) / 2).to(torch.uint8)
            else:
                self.mask = np.outer(self.row, self.col)
                self.mask = np.round((self.mask + 1) / 2).astype(np.uint8)
        assert self.mask is not None, "Mask should be specified"

        # resize to sensor shape
        if np.any(self.resolution != self.mask.shape):

            if self.use_torch:
                self.mask = self.mask.unsqueeze(0).unsqueeze(0)
                self.mask = torch.nn.functional.interpolate(
                    self.mask, size=tuple(self.resolution), mode="nearest"
                ).squeeze()
            else:
                # self.mask = resize(self.mask[:, :, np.newaxis], shape=tuple(self.resolution) + (1,))
                self.mask = resize(
                    self.mask[:, :, np.newaxis],
                    shape=tuple(self.resolution) + (1,),
                    interpolation=cv.INTER_NEAREST,
                ).squeeze()

    def is_prime(self, n):
        """
        Assess whether a number is prime or not.

        Parameters
        ----------
        n: int
            The number we want to check.
        """
        if n % 2 == 0 and n > 2:
            return False
        return all(n % i for i in range(3, int(sqrt(n)) + 1, 2))

    def generate_mura(self, p):
        """
        Generate MURA square pattern.

        Parameters
        ----------
        p: int
            Number of bits.
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

    def get_conv_matrices(self, img_shape):
        """
        Get theoretical left and right convolution matrices for the separable mask.

        Such that measurement model is given ``P @ img @ Q.T``.

        Parameters
        ----------
        img_shape: tuple
            Shape of the image to being convolved.

        Returns
        -------
        P: :py:class:`~numpy.ndarray`
            Left convolution matrix.
        Q: :py:class:`~numpy.ndarray`
            Right convolution matrix.

        """

        P = circulant(np.resize(self.col, self.resolution[0]))[:, : img_shape[0]]
        Q = circulant(np.resize(self.row, self.resolution[1]))[:, : img_shape[1]]

        return P, Q

    def simulate(self, obj, snr_db=20):
        """
        Simulate the mask measurement of an image. Apply left and right convolution matrices,
        add noise and return the measurement.

        Parameters
        ----------
        obj: :py:class:`~numpy.ndarray`
            Image to simulate.
        snr_db: float, optional
            Signal-to-noise ratio (dB) of the simulated measurement. Default is 20 dB.
        """
        assert len(obj.shape) == 3, "Object should be a 3D array (HxWxC) even if grayscale."

        # Get convolution matrices
        P, Q = self.get_conv_matrices(obj.shape)

        # Convolve image
        n_channels = obj.shape[-1]

        if torch_available and isinstance(obj, torch.Tensor):
            P = torch.from_numpy(P).float()
            Q = torch.from_numpy(Q).float()
            meas = torch.dstack(
                [torch.linalg.multi_dot([P, obj[:, :, c], Q.T]) for c in range(n_channels)]
            ).float()
        else:
            meas = np.dstack([multi_dot([P, obj[:, :, c], Q.T]) for c in range(n_channels)])

        # Add noise
        if snr_db is not None:
            meas = add_shot_noise(meas, snr_db=snr_db)

        if torch_available and isinstance(obj, torch.Tensor):
            meas = meas.to(obj)

        return meas


class MultiLensArray(Mask):
    """
    Multi-lens array mask.
    """

    def __init__(
        self,
        N=None,
        radius=None,
        loc=None,
        refractive_index=1.2,
        seed=0,
        min_height=1e-5,
        radius_range=(1e-4, 4e-4),
        min_separation=1e-4,
        focal_range=None,
        verbose=False,
        **kwargs,
    ):
        """
        Multi-lens array mask constructor.

        Parameters
        ----------
        N: int
            Number of micro-lenses.
        radius: array_like
            Radius of the lenses (m).
        loc: array_like of tuples
            Location of the lenses (m).
        refractive_index: float
            Refractive index of the mask substrate. Default is 1.2.
        seed: int
            Seed for the random number generator. Default is 0.
        min_height: float
            Minimum height of the lenses (m). Default is 1e-3.
        radius_range: array_like
            Range of the radius of the lenses (m). Default is (1e-4, 4e-4) m.
        focal_range: array_like
            Range of the focal length of the lenses (m). Default is None. Overrides the radius_range.
        min_separation: float
            Minimum separation between lenses (m). Default is 1e-4.
        verbose: bool
            If True, print lens placement information. Default is False.
        """
        self.N = N
        self.radius = radius
        self.loc = loc
        self.refractive_index = refractive_index
        self.seed = seed
        self.min_height = min_height
        self.min_separation = min_separation
        self.verbose = verbose

        self.radius_range = radius_range
        if focal_range is not None:
            self.radius_range = [
                focal_range[0] / (self.refractive_index - 1),
                focal_range[1] / (self.refractive_index - 1),
            ]

        super().__init__(**kwargs)

    def check_asserts(self):
        """
        Check the validity of the parameters.

        Generate the locations and radii of the lenses if not specified.
        """
        assert (
            self.radius_range[0] < self.radius_range[1]
        ), "Minimum radius should be smaller than maximum radius"
        if self.radius is not None:
            if self.use_torch:
                assert torch.all(self.radius >= 0)
            else:
                assert np.all(self.radius >= 0)
            assert (
                self.loc is not None
            ), "Location of the lenses should be specified if their radius is specified"
            assert len(self.radius) == len(
                self.loc
            ), "Number of radius should be equal to the number of locations"
            self.N = len(self.radius)
            circles = (
                np.array([(self.loc[i][0], self.loc[i][1], self.radius[i]) for i in range(self.N)])
                if not self.use_torch
                else torch.tensor(
                    [(self.loc[i][0], self.loc[i][1], self.radius[i]) for i in range(self.N)]
                ).to(self.torch_device)
            )
            assert self.no_circle_overlap(circles), "lenses should not overlap"
        else:
            # generate random locations and radii
            assert (
                self.N is not None
            ), "If positions are not specified, the number of lenses should be specified"

            np.random.seed(self.seed)
            self.radius = np.random.uniform(self.radius_range[0], self.radius_range[1], self.N)
            # radius get sorted in descending order
            self.loc, self.radius = self.place_spheres_on_plane(self.radius)
            if self.centered:
                self.loc = self.loc - np.array(self.size) / 2
            if self.use_torch:
                self.radius = torch.tensor(self.radius).to(self.torch_device)
                self.loc = torch.tensor(self.loc).to(self.torch_device)

    def no_circle_overlap(self, circles):
        """
        Check if any circle in the list overlaps with another.

        Parameters
        ----------
        circles: array_like
            List of circles, each represented by a tuple (x, y, r) with location (x, y) and radius r.
        """
        for i in range(len(circles)):
            if self.does_circle_overlap(
                circles[i + 1 :], circles[i][0], circles[i][1], circles[i][2]
            ):
                return False
        return True

    def does_circle_overlap(self, circles, x, y, r):
        """Check if a circle overlaps with any in the list."""
        for (cx, cy, cr) in circles:
            if sqrt((x - cx) ** 2 + (y - cy) ** 2) <= (r + cr + self.min_separation):
                return True, (cx, cy, cr)
        return False

    def place_spheres_on_plane(self, radius, max_attempts=1000):
        """Try to place circles of given radius on a 2D plane."""
        placed_circles = []
        rad_sorted = sorted(radius, reverse=True)  # sort the radius in descending order
        loc = []
        r_placed = []
        for r in rad_sorted:
            placed = False
            for _ in range(max_attempts):
                x = np.random.uniform(r, self.size[1] - r)
                y = np.random.uniform(r, self.size[0] - r)
                if not self.does_circle_overlap(placed_circles, x, y, r):
                    placed_circles.append((x, y, r))
                    loc.append([x, y])
                    r_placed.append(r)
                    placed = True
                    if self.verbose:
                        print(f"Placed circle with rad {r}, and center ({x}, {y})")
                    break
            if not placed:
                if self.verbose:
                    print(f"Failed to place circle with rad {r}")
                continue
        if len(r_placed) < self.N:
            warnings.warn(f"Could not place {self.N - len(r_placed)} lenses")
        return np.array(loc, dtype=np.float32), np.array(r_placed, dtype=np.float32)

    def create_mask(self, loc=None, radius=None):
        """
        Creating multi-lens array mask.

        Parameters
        ----------
        loc: array_like of tuples, optional
            Location of the lenses (m).
        radius: array_like, optional
            Radius of the lenses (m).
        """
        if radius is not None:
            self.radius = radius
        if loc is not None:
            self.loc = loc
        self.check_asserts()

        # convert to pixels (assume same size for x and y)
        locs_pix = self.loc * (1 / self.feature_size[0])
        radius_pix = self.radius * (1 / self.feature_size[0])
        self.height_map = self.create_height_map(radius_pix, locs_pix)

    def create_height_map(self, radius, locs):
        height = (
            np.full((self.resolution[0], self.resolution[1]), self.min_height).astype(np.float32)
            if not self.use_torch
            else torch.full((self.resolution[0], self.resolution[1]), self.min_height).to(
                self.torch_device, dtype=torch.float32
            )
        )
        x = (
            np.arange(self.resolution[0]).astype(np.float32)
            if not self.use_torch
            else torch.arange(self.resolution[0]).to(self.torch_device)
        )
        y = (
            np.arange(self.resolution[1]).astype(np.float32)
            if not self.use_torch
            else torch.arange(self.resolution[1]).to(self.torch_device)
        )
        if self.centered:
            x = x - self.resolution[1] / 2
            y = y - self.resolution[0] / 2
        X, Y = (
            np.meshgrid(x, y, indexing="ij")
            if not self.use_torch
            else torch.meshgrid(x, y, indexing="ij")
        )
        for idx, rad in enumerate(radius):
            contribution = self.lens_contribution(X, Y, rad, locs[idx]) * self.feature_size[0]
            contribution[(X - locs[idx][1]) ** 2 + (Y - locs[idx][0]) ** 2 > rad**2] = 0
            height = height + contribution
        height[height < self.min_height] = self.min_height
        return height

    def lens_contribution(self, x, y, radius, loc):
        return (
            np.sqrt(radius**2 - (x - loc[1]) ** 2 - (y - loc[0]) ** 2)
            if not self.use_torch
            else torch.sqrt(radius**2 - (x - loc[1]) ** 2 - (y - loc[0]) ** 2)
        )

    @property
    def focal_length(self):
        """
        Focal length of the lenses.

        As we have a plano-convex lens: 1/f = (n-1) / R -> f = R / (n-1)
        """
        return self.radius / (self.refractive_index - 1)


class PhaseContour(Mask):
    """
    Phase contour mask as in `PhlatCam <https://ieeexplore.ieee.org/document/9076617>`_.
    """

    def __init__(
        self, noise_period=(16, 16), refractive_index=1.2, n_iter=10, design_wv=532e-9, **kwargs
    ):
        """
        Phase contour mask contructor (PhlatCam).

        Parameters
        ----------
        noise_period: array_like, optional
            Noise period of the Perlin noise (px). Default is (8, 8).
        refractive_index: float, optional
            Refractive index of the mask substrate. Default is 1.2.
        n_iter: int, optional
            Number of iterations for the phase retrieval algorithm. Default is 10.
        design_wv: float, optional
            Wavelength used to design the mask (m). Default is 532e-9, as in the PhlatCam paper.
        **kwargs:
            The keyword arguments are passed to the parent class :py:class:`~lensless.hardware.mask.Mask`.
        """

        self.target_psf = None
        self.phase_pattern = None
        self.height_map = None
        self.noise_period = noise_period
        self.refractive_index = refractive_index
        self.n_iter = n_iter
        self.design_wv = design_wv

        super().__init__(**kwargs)

    def create_mask(self):
        """
        Creating phase contour from edges of Perlin noise.
        """

        # Creating Perlin noise
        proper_dim_1 = (self.resolution[0] // self.noise_period[0]) * self.noise_period[0]
        proper_dim_2 = (self.resolution[1] // self.noise_period[1]) * self.noise_period[1]
        noise = generate_perlin_noise_2d((proper_dim_1, proper_dim_2), self.noise_period)

        # Upscaling to correspond to sensor size
        if np.any(self.resolution != noise.shape):
            noise = resize(noise[:, :, np.newaxis], shape=tuple(self.resolution) + (1,)).squeeze()

        # Edge detection
        binary = np.clip(np.round(np.interp(noise, (-1, 1), (0, 1))), a_min=0, a_max=1)
        self.target_psf = cv.Canny(np.interp(binary, (-1, 1), (0, 255)).astype(np.uint8), 0, 255)

        # Computing mask and height map
        assert (
            self.distance_sensor is not None
        ), "Distance between mask and sensor should be specified."
        _, height_map = phase_retrieval(
            target_psf=self.target_psf,
            wv=self.design_wv,
            d1=self.feature_size,
            dz=self.distance_sensor,
            n=self.refractive_index,
            n_iter=self.n_iter,
            height_map=True,
        )
        self.height_map = height_map


def phase_retrieval(target_psf, wv, d1, dz, n=1.2, n_iter=10, height_map=False):
    """
    Iterative phase retrieval algorithm similar to `PhlatCam <https://ieeexplore.ieee.org/document/9076617>`_,
    using Fresnel propagation.

    Parameters
    ----------
    target_psf: array_like
        Target PSF to optimize the phase mask for.
    wv: float
        Wavelength (m).
    d1: float
        Sample period on the sensor i.e. pixel size (m).
    dz: float
        Propagation distance between the mask and the sensor.
    n: float
        Refractive index of the mask substrate. Default is 1.2.
    n_iter: int
        Number of iterations. Default value is 10.
    """
    M_p = np.sqrt(target_psf)

    if hasattr(d1, "__len__"):
        if d1[0] != d1[1]:
            warnings.warn("Non-square pixel, first dimension taken as feature size.")
        d1 = d1[0]

    for _ in range(n_iter):
        # back propagate from sensor to mask
        M_phi = fresnel_conv(M_p, wv, d1, -dz, dtype=np.float32)[0]
        # constrain amplitude at mask to be unity, i.e. phase pattern
        M_phi = np.exp(1j * np.angle(M_phi))
        # forward propagate from mask to sensor
        M_p = fresnel_conv(M_phi, wv, d1, dz, dtype=np.float32)[0]
        # constrain amplitude to be sqrt(PSF)
        M_p = np.sqrt(target_psf) * np.exp(1j * np.angle(M_p))

    phi = (np.angle(M_phi) + 2 * np.pi) % (2 * np.pi)

    if height_map:
        return phi, wv * phi / (2 * np.pi * (n - 1))
    else:
        return phi


class FresnelZoneAperture(Mask):
    """
    Fresnel Zone Aperture (FZA) mask as in `this work <https://www.nature.com/articles/s41377-020-0289-9>`_,
    namely binarized cosine function.
    """

    def __init__(self, radius=0.56e-3, **kwargs):
        """
        Fresnel Zone Aperture mask contructor.

        Parameters
        ----------
        radius: float
            Radius of the FZA (m). Default value is 0.56e-3 (largest in the paper, others are 0.32e-3 and 0.25e-3).
        **kwargs:
            The keyword arguments are passed to the parent class :py:class:`~lensless.hardware.mask.Mask`.
        """

        self.radius = radius

        super().__init__(**kwargs)

    def create_mask(self):
        """
        Creating binary Fresnel Zone Aperture mask.
        """
        dim = self.resolution
        x, y = np.meshgrid(
            np.linspace(-dim[1] / 2, dim[1] / 2 - 1, dim[1]),
            np.linspace(-dim[0] / 2, dim[0] / 2 - 1, dim[0]),
        )
        radius_px = self.radius / self.feature_size[0]
        mask = 0.5 * (1 + np.cos(np.pi * (x**2 + y**2) / radius_px**2))
        self.mask = np.round(mask)
