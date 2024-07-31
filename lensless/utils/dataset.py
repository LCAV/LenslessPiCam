# #############################################################################
# dataset.py
# =================
# Authors :
# Yohann PERRON [yohann.perron@gmail.com]
# Eric BEZZAM [ebezzam@gmail.com]
# #############################################################################

from hydra.utils import get_original_cwd
import numpy as np
import glob
import os
import torch
from abc import abstractmethod
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms
from torchvision.transforms import functional as F
from lensless.hardware.trainable_mask import prep_trainable_mask, AdafruitLCD
from lensless.utils.simulation import FarFieldSimulator
from lensless.utils.io import load_image, load_psf, save_image
from lensless.utils.image import is_grayscale, resize, rgb2gray
import re
from lensless.hardware.utils import capture
from lensless.hardware.utils import display
from lensless.hardware.slm import set_programmable_mask, adafruit_sub2full
from datasets import load_dataset
from lensless.recon.rfft_convolve import RealFFTConvolve2D
from huggingface_hub import hf_hub_download
import cv2
from lensless.hardware.sensor import sensor_dict, SensorParam
from scipy.ndimage import rotate
import warnings
from waveprop.noise import add_shot_noise
from lensless.utils.image import shift_with_pad
from PIL import Image


def convert(text):
    return int(text) if text.isdigit() else text.lower()


def alphanum_key(key):
    return [convert(c) for c in re.split("([0-9]+)", key)]


def natural_sort(arr):
    return sorted(arr, key=alphanum_key)


class DualDataset(Dataset):
    """
    Abstract class for defining a dataset of paired lensed and lensless images.
    """

    def __init__(
        self,
        indices=None,
        # psf_path=None,
        background=None,
        # background_pix=(0, 15),
        downsample=1,
        flip=False,
        flip_ud=False,
        flip_lr=False,
        transform_lensless=None,
        transform_lensed=None,
        input_snr=None,
        **kwargs,
    ):
        """
        Dataset consisting of lensless and corresponding lensed image.

        Parameters
        ----------
        indices : range or int or None
            Indices of the images to use in the dataset (if integer, it should be interpreted as range(indices)), by default None.
        psf_path : str
            Path to the PSF of the imaging system, by default None.
        background : :py:class:`~torch.Tensor` or None, optional
            If not ``None``, background is removed from lensless images, by default ``None``. If PSF is provided, background is estimated from the PSF.
        background_pix : tuple, optional
            Pixels to use for background estimation, by default (0, 15).
        downsample : int, optional
            Downsample factor of the lensless images, by default 1.
        flip : bool, optional
            If ``True``, lensless images are flipped, by default ``False``.
        transform_lensless : PyTorch Transform or None, optional
            Transform to apply to the lensless images, by default ``None``. Note that this transform is applied on HWC images (different from torchvision).
        transform_lensed : PyTorch Transform or None, optional
            Transform to apply to the lensed images, by default ``None``. Note that this transform is applied on HWC images (different from torchvision).
        input_snr : float, optional
            If not ``None``, Poisson noise is added to the lensless images to match the given SNR.
        """
        if isinstance(indices, int):
            indices = range(indices)
        self.indices = indices
        self.background = background
        self.input_snr = input_snr
        self.downsample = downsample
        self.flip = flip
        self.flip_ud = flip_ud
        self.flip_lr = flip_lr
        self.transform_lensless = transform_lensless
        self.transform_lensed = transform_lensed

        # self.psf = None
        # if psf_path is not None:
        #     psf, background = load_psf(
        #         psf_path,
        #         downsample=downsample,
        #         return_float=True,
        #         return_bg=True,
        #         bg_pix=background_pix,
        #     )
        #     if self.background is None:
        #         self.background = background
        #     self.psf = torch.from_numpy(psf)
        #     if self.transform_lensless is not None:
        #         self.psf = self.transform_lensless(self.psf)

    @abstractmethod
    def __len__(self):
        """
        Abstract method to get the length of the dataset. It should take into account the indices parameter.
        """
        raise NotImplementedError

    @abstractmethod
    def _get_images_pair(self, idx):
        """
        Abstract method to get the lensed and lensless images. Should return a pair (lensless, lensed) of numpy arrays with values in [0,1].

        Parameters
        ----------
        idx : int
            images index
        """
        raise NotImplementedError

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        if self.indices is not None:
            idx = self.indices[idx]
        lensless, lensed = self._get_images_pair(idx)

        if isinstance(lensless, np.ndarray):
            # expected case
            if self.downsample != 1.0:
                lensless = resize(lensless, factor=1 / self.downsample)
                lensed = resize(lensed, factor=1 / self.downsample)

            lensless = torch.from_numpy(lensless)
            lensed = torch.from_numpy(lensed)
        else:
            # torch tensor
            # This mean get_images_pair returned a torch tensor. This isn't recommended, if possible get_images_pair should return a numpy array
            # In this case it should also have applied the downsampling
            pass

        # If [H, W, C] -> [D, H, W, C]
        if len(lensless.shape) == 3:
            lensless = lensless.unsqueeze(0)
        if len(lensed.shape) == 3:
            lensed = lensed.unsqueeze(0)

        if self.background is not None:
            lensless = lensless - self.background
            lensless = torch.clamp(lensless, min=0)

        # add noise
        if self.input_snr is not None:
            lensless = add_shot_noise(lensless, self.input_snr)

        # flip image x and y if needed
        if self.flip:
            lensless = torch.rot90(lensless, dims=(-3, -2), k=2)
            lensed = torch.rot90(lensed, dims=(-3, -2), k=2)
        if self.flip_ud:
            lensless = torch.flip(lensless, dims=(-4, -3))
            lensed = torch.flip(lensed, dims=(-4, -3))
        if self.flip_lr:
            lensless = torch.flip(lensless, dims=(-4, -2))
            lensed = torch.flip(lensed, dims=(-4, -2))
        if self.transform_lensless:
            lensless = self.transform_lensless(lensless)
        if self.transform_lensed:
            lensed = self.transform_lensed(lensed)

        return lensless, lensed


class SimulatedFarFieldDataset(DualDataset):
    """
    Dataset of propagated images (through simulation) from a Torch Dataset. :py:class:`lensless.utils.simulation.FarFieldSimulator` is used for simulation,
    assuming a far-field propagation and a shift-invariant system with a single point spread function (PSF).

    """

    def __init__(
        self,
        dataset,
        simulator,
        pre_transform=None,
        dataset_is_CHW=False,
        flip=False,
        vertical_shift=None,
        horizontal_shift=None,
        crop=None,
        downsample=1,
        **kwargs,
    ):
        """
        Parameters
        ----------

        dataset : :py:class:`torch.utils.data.Dataset`
            Dataset to propagate. Should output images with shape [H, W, C] unless ``dataset_is_CHW`` is ``True`` (and therefore images have the dimension ordering of [C, H, W]).
        simulator : :py:class:`lensless.utils.simulation.FarFieldSimulator`
            Simulator object used on images from ``dataset``. Waveprop simulator to use for the simulation. It is expected to have ``is_torch = True``.
        pre_transform : PyTorch Transform or None, optional
            Transform to apply to the images before simulation, by default ``None``. Note that this transform is applied on HCW images (different from torchvision).
        dataset_is_CHW : bool, optional
            If True, the input dataset is expected to output images with shape [C, H, W], by default ``False``.
        flip : bool, optional
            If True, images are flipped beffore the simulation, by default ``False``.
        """

        # we do the flipping before the simualtion
        super(SimulatedFarFieldDataset, self).__init__(flip=False, **kwargs)

        assert isinstance(dataset, Dataset)
        self.dataset = dataset
        self.n_files = len(dataset)
        self.dataset_is_CHW = dataset_is_CHW
        self._pre_transform = pre_transform
        self.flip_pre_sim = flip

        self.vertical_shift = vertical_shift
        self.horizontal_shift = horizontal_shift
        self.crop = crop.copy() if crop is not None else None
        if downsample != 1:
            if self.vertical_shift is not None:
                self.vertical_shift = int(self.vertical_shift // downsample)
            if self.horizontal_shift is not None:
                self.horizontal_shift = int(self.horizontal_shift // downsample)

            if crop is not None:
                self.crop["vertical"][0] = int(self.crop["vertical"][0] // downsample)
                self.crop["vertical"][1] = int(self.crop["vertical"][1] // downsample)
                self.crop["horizontal"][0] = int(self.crop["horizontal"][0] // downsample)
                self.crop["horizontal"][1] = int(self.crop["horizontal"][1] // downsample)

        # check simulator
        assert isinstance(simulator, FarFieldSimulator), "Simulator should be a FarFieldSimulator"
        assert simulator.is_torch, "Simulator should be a pytorch simulator"
        assert simulator.fft_shape is not None, "Simulator should have a psf"
        self.sim = simulator

    @property
    def psf(self):
        return self.sim.get_psf()

    def get_image(self, index):
        return self.dataset[index]

    def _get_images_pair(self, index):
        # load image
        img, _ = self.get_image(index)
        # convert to HWC for simulator and transform
        if self.dataset_is_CHW:
            img = img.moveaxis(-3, -1)
        if self.flip_pre_sim:
            img = torch.rot90(img, dims=(-3, -2))
        if self._pre_transform is not None:
            img = self._pre_transform(img)

        lensless, lensed = self.sim.propagate_image(img, return_object_plane=True)

        if self.vertical_shift is not None:
            lensed = torch.roll(lensed, self.vertical_shift, dims=-3)
        if self.horizontal_shift is not None:
            lensed = torch.roll(lensed, self.horizontal_shift, dims=-2)

        if lensed.shape[-1] == 1 and lensless.shape[-1] == 3:
            # copy to 3 channels
            lensed = lensed.repeat(1, 1, 3)
        assert (
            lensed.shape[-1] == lensless.shape[-1]
        ), "Lensed and lensless should have same number of channels"

        return lensless, lensed

    def __len__(self):
        if self.indices is None:
            return self.n_files
        else:
            return len([x for x in self.indices if x < self.n_files])


class MeasuredDatasetSimulatedOriginal(DualDataset):
    """
    Abstract class for defining a dataset of paired lensed and lensless images.

    Dataset consisting of lensless image captured from a screen and the corresponding image shown on the screen.
    Unlike :py:class:`lensless.utils.dataset.MeasuredDataset`, the ground-truth lensed image is simulated using a :py:class:`lensless.utils.simulation.FarFieldSimulator`
    object rather than measured with a lensed camera.

    The class assumes that the ``measured_dir`` and ``original_dir`` have file names that match.

    The method ``_get_images_pair`` must be defined.
    """

    def __init__(
        self,
        measured_dir,
        original_dir,
        simulator,
        measurement_ext="png",
        original_ext="jpg",
        downsample=1,
        background=None,
        flip=False,
        **kwargs,
    ):
        """
        Dataset consisting of lensless image captured from a screen and the corresponding image shown on screen.

        Parameters
        ----------
        """
        super(MeasuredDatasetSimulatedOriginal, self).__init__(
            downsample=1, background=background, flip=flip, **kwargs
        )
        self.pre_downsample = downsample

        self.measured_dir = measured_dir
        self.original_dir = original_dir
        assert os.path.isdir(self.measured_dir)
        assert os.path.isdir(self.original_dir)

        self.measurement_ext = measurement_ext.lower()
        self.original_ext = original_ext.lower()

        files = natural_sort(glob.glob(os.path.join(self.measured_dir, "*." + measurement_ext)))

        self.files = [os.path.basename(fn) for fn in files]

        if len(self.files) == 0:
            raise FileNotFoundError(
                f"No files found in {self.measured_dir} with extension {self.measurement_ext }"
            )

        # check that corresponding files exist
        for fn in self.files:
            original_fp = os.path.join(self.original_dir, fn[:-3] + self.original_ext)
            assert os.path.exists(original_fp), f"File {original_fp} does not exist"

        # check simulator
        assert isinstance(simulator, FarFieldSimulator), "Simulator should be a FarFieldSimulator"
        assert simulator.is_torch, "Simulator should be a pytorch simulator"
        assert simulator.fft_shape is None, "Simulator should not have a psf"
        self.sim = simulator

    def __len__(self):
        if self.indices is None:
            return len(self.files)
        else:
            return len([i for i in self.indices if i < len(self.files)])

    # def _get_images_pair(self, idx):
    #     if self.image_ext == "npy" or self.image_ext == "npz":
    #         lensless_fp = os.path.join(self.lensless_dir, self.files[idx])
    #         original_fp = os.path.join(self.original_dir, self.files[idx])
    #         lensless = np.load(lensless_fp)
    #         lensless = resize(lensless, factor=1 / self.downsample)
    #         original = np.load(original_fp[:-3] + self.original_ext)
    #     else:
    #         # more standard image formats: png, jpg, tiff, etc.
    #         lensless_fp = os.path.join(self.lensless_dir, self.files[idx])
    #         original_fp = os.path.join(self.original_dir, self.files[idx])
    #         lensless = load_image(lensless_fp, downsample=self.pre_downsample)
    #         original = load_image(
    #             original_fp[:-3] + self.original_ext, downsample=self.pre_downsample
    #         )

    #         # convert to float
    #         if lensless.dtype == np.uint8:
    #             lensless = lensless.astype(np.float32) / 255
    #             original = original.astype(np.float32) / 255
    #         else:
    #             # 16 bit
    #             lensless = lensless.astype(np.float32) / 65535
    #             original = original.astype(np.float32) / 65535

    #     # convert to torch
    #     lensless = torch.from_numpy(lensless)
    #     original = torch.from_numpy(original)

    #     # project original image to lensed space
    #     with torch.no_grad():
    #         lensed = self.sim.propagate_image()

    #     return lensless, lensed


class DigiCamCelebA(MeasuredDatasetSimulatedOriginal):
    def __init__(
        self,
        celeba_root,
        data_dir=None,
        psf_path=None,
        downsample=1,
        flip=True,
        vertical_shift=None,
        horizontal_shift=None,
        crop=None,
        simulation_config=None,
        **kwargs,
    ):
        """

        Some parameters default to work for the ``celeba_adafruit_random_2mm_20230720_10K`` dataset,
        namely: flip, vertical_shift, horizontal_shift, crop, simulation_config.

        Parameters
        ----------
        celeba_root : str
            Path to the CelebA dataset.
        data_dir : str, optional
            Path to the lensless images, by default looks inside the ``data`` folder. Can download if not available.
        psf_path : str, optional
            Path to the PSF of the imaging system, by default looks inside the ``data/psf`` folder. Can download if not available.
        downsample : int, optional
            Downsample factor of the lensless images, by default 1.
        flip : bool, optional
            If True, measurements are flipped, by default ``True``. Does not get applied to the original images.
        vertical_shift : int, optional
            Vertical shift (in pixels) of the lensed images to align.
        horizontal_shift : int, optional
            Horizontal shift (in pixels) of the lensed images to align.
        crop : dict, optional
            Dictionary of crop parameters (vertical: [start, end], horizontal: [start, end]) to select region of interest.
        """

        if vertical_shift is None:
            # default to (no downsampling) of celeba_adafruit_random_2mm_20230720_10K
            vertical_shift = -85
            horizontal_shift = -5

        if crop is None:
            crop = {"vertical": [30, 560], "horizontal": [285, 720]}
        self.crop = crop

        self.vertical_shift = vertical_shift
        self.horizontal_shift = horizontal_shift
        if downsample != 1:
            self.vertical_shift = int(self.vertical_shift // downsample)
            self.horizontal_shift = int(self.horizontal_shift // downsample)

            self.crop["vertical"][0] = int(self.crop["vertical"][0] // downsample)
            self.crop["vertical"][1] = int(self.crop["vertical"][1] // downsample)
            self.crop["horizontal"][0] = int(self.crop["horizontal"][0] // downsample)
            self.crop["horizontal"][1] = int(self.crop["horizontal"][1] // downsample)

        # download dataset if necessary
        if data_dir is None:
            data_dir = os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "data",
                "celeba_adafruit_random_2mm_20230720_10K",
            )
        if not os.path.isdir(data_dir):
            main_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
            print("DigiCam CelebA dataset not found.")
            try:
                from torchvision.datasets.utils import download_and_extract_archive
            except ImportError:
                exit()
            msg = "Do you want to download this dataset of 10K examples (12.2GB)?"

            # default to yes if no input is given
            valid = input("%s (Y/n) " % msg).lower() != "n"
            if valid:
                url = "https://drive.switch.ch/index.php/s/9NNGCJs3DoBDGlY/download"
                filename = "celeba_adafruit_random_2mm_20230720_10K.zip"
                download_and_extract_archive(url, main_dir, filename=filename, remove_finished=True)

        # download PSF if necessary
        if psf_path is None:
            psf_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "data",
                "psf",
                "adafruit_random_2mm_20231907.png",
            )
        if not os.path.exists(psf_path):
            try:
                from torchvision.datasets.utils import download_url
            except ImportError:
                exit()
            msg = "Do you want to download the PSF (38.8MB)?"

            # default to yes if no input is given
            valid = input("%s (Y/n) " % msg).lower() != "n"
            output_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "psf")
            if valid:
                url = "https://drive.switch.ch/index.php/s/kfN5vOqvVkNyHmc/download"
                filename = "adafruit_random_2mm_20231907.png"
                download_url(url, output_path, filename=filename)

        # load PSF
        self.flip_measurement = flip
        psf, background = load_psf(
            psf_path,
            downsample=downsample * 4,  # PSF is 4x the resolution of the images
            return_float=True,
            return_bg=True,
            flip=flip,
            bg_pix=(0, 15),
        )
        self.psf = torch.from_numpy(psf)

        # create simulator
        simulation_config["output_dim"] = tuple(self.psf.shape[-3:-1])
        simulator = FarFieldSimulator(
            is_torch=True,
            **simulation_config,
        )

        super().__init__(
            measured_dir=data_dir,
            original_dir=os.path.join(celeba_root, "celeba", "img_align_celeba"),
            simulator=simulator,
            measurement_ext="png",
            original_ext="jpg",
            downsample=downsample,
            background=background,
            flip=False,  # will do flipping only on measurement
            **kwargs,
        )

    def _get_images_pair(self, idx):

        # more standard image formats: png, jpg, tiff, etc.
        lensless_fp = os.path.join(self.measured_dir, self.files[idx])
        original_fp = os.path.join(self.original_dir, self.files[idx][:-3] + self.original_ext)
        lensless = load_image(
            lensless_fp, downsample=self.pre_downsample, flip=self.flip_measurement
        )
        original = load_image(original_fp[:-3] + self.original_ext)

        # convert to float
        if lensless.dtype == np.uint8:
            lensless = lensless.astype(np.float32) / 255
            original = original.astype(np.float32) / 255
        else:
            # 16 bit
            lensless = lensless.astype(np.float32) / 65535
            original = original.astype(np.float32) / 65535

        # convert to torch
        lensless = torch.from_numpy(lensless)
        original = torch.from_numpy(original)

        # project original image to lensed space
        with torch.no_grad():
            lensed = self.sim.propagate_image(original, return_object_plane=True)

        if self.vertical_shift is not None:
            lensed = torch.roll(lensed, self.vertical_shift, dims=-3)
        if self.horizontal_shift is not None:
            lensed = torch.roll(lensed, self.horizontal_shift, dims=-2)

        return lensless, lensed


class MeasuredDataset(DualDataset):
    """
    Dataset consisting of lensless and corresponding lensed image.
    It can be used with a PyTorch DataLoader to load a batch of lensless and corresponding lensed images.
    Unless the setup is perfectly calibrated, one should expect to have to use ``transform_lensed`` to adjust the alignment and rotation.
    """

    def __init__(
        self,
        root_dir,
        lensless_fn="diffuser",
        lensed_fn="lensed",
        image_ext="npy",
        **kwargs,
    ):
        """
        Dataset consisting of lensless and corresponding lensed image. Default parameters are for the
        `DiffuserCam Lensless Mirflickr Dataset (DLMD) <https://waller-lab.github.io/LenslessLearning/dataset.html>`_.

        Parameters
        ----------
        root_dir : str
            Path to the test dataset. It is expected to contain two folders: ones of lensless images and one of lensed images.
        lensless_fn : str, optional
            Name of the folder containing the lensless images, by default "diffuser".
        lensed_fn : str, optional
            Name of the folder containing the lensed images, by default "lensed".
        image_ext : str, optional
            Extension of the images, by default "npy".
        """

        super(MeasuredDataset, self).__init__(**kwargs)

        self.root_dir = root_dir
        self.lensless_dir = os.path.join(root_dir, lensless_fn)
        self.lensed_dir = os.path.join(root_dir, lensed_fn)
        assert os.path.isdir(self.lensless_dir)
        assert os.path.isdir(self.lensed_dir)

        self.image_ext = image_ext.lower()

        files = natural_sort(glob.glob(os.path.join(self.lensless_dir, "*." + image_ext)))
        self.files = [os.path.basename(fn) for fn in files]

        if len(self.files) == 0:
            raise FileNotFoundError(
                f"No files found in {self.lensless_dir} with extension {image_ext}"
            )

    def __len__(self):
        if self.indices is None:
            return len(self.files)
        else:
            return len([i for i in self.indices if i < len(self.files)])

    def _get_images_pair(self, idx):
        if self.image_ext == "npy" or self.image_ext == "npz":
            lensless_fp = os.path.join(self.lensless_dir, self.files[idx])
            lensed_fp = os.path.join(self.lensed_dir, self.files[idx])
            lensless = np.load(lensless_fp)
            lensed = np.load(lensed_fp)

        else:
            # more standard image formats: png, jpg, tiff, etc.
            lensless_fp = os.path.join(self.lensless_dir, self.files[idx])
            lensed_fp = os.path.join(self.lensed_dir, self.files[idx])
            lensless = load_image(lensless_fp)
            lensed = load_image(lensed_fp)

            # convert to float
            if lensless.dtype == np.uint8:
                lensless = lensless.astype(np.float32) / 255
                lensed = lensed.astype(np.float32) / 255
            else:
                # 16 bit
                lensless = lensless.astype(np.float32) / 65535
                lensed = lensed.astype(np.float32) / 65535

        return lensless, lensed


class DiffuserCamMirflickr(MeasuredDataset):
    """
    Helper class for DiffuserCam Mirflickr dataset.

    Note that image colors are in BGR format: https://github.com/Waller-Lab/LenslessLearning/blob/master/utils.py#L432
    """

    def __init__(
        self,
        dataset_dir,
        psf_path,
        downsample=2,
        **kwargs,
    ):

        # check psf path exist
        if not os.path.exists(psf_path):
            psf_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "data", "psf", "diffusercam_psf.tiff"
            )

            try:
                from torchvision.datasets.utils import download_url
            except ImportError:
                exit()
            msg = "Do you want to download the DiffuserCam PSF (5.9MB)?"

            # default to yes if no input is given
            valid = input("%s (Y/n) " % msg).lower() != "n"
            output_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "psf")
            if valid:
                url = "https://drive.switch.ch/index.php/s/BteiuEcONmhmDSn/download"
                filename = "diffusercam_psf.tiff"
                download_url(url, output_path, filename=filename)

        psf, background = load_psf(
            psf_path,
            downsample=downsample * 4,  # PSF is 4x the resolution of the images
            return_float=True,
            return_bg=True,
            bg_pix=(0, 15),
        )
        transform_BRG2RGB = transforms.Lambda(lambda x: x[..., [2, 1, 0]])
        self.psf = transform_BRG2RGB(torch.from_numpy(psf))
        self.allowed_idx = np.arange(2, 25001)

        assert os.path.isdir(os.path.join(dataset_dir, "diffuser_images")) and os.path.isdir(
            os.path.join(dataset_dir, "ground_truth_lensed")
        ), "Dataset should contain 'diffuser_images' and 'ground_truth_lensed' folders. It can be downloaded from https://waller-lab.github.io/LenslessLearning/dataset.html"

        super().__init__(
            root_dir=dataset_dir,
            background=background,
            downsample=downsample,
            flip=False,
            transform_lensless=transform_BRG2RGB,
            transform_lensed=transform_BRG2RGB,
            lensless_fn="diffuser_images",
            lensed_fn="ground_truth_lensed",
            image_ext="npy",
            **kwargs,
        )

    def _get_images_pair(self, idx):

        assert idx >= self.allowed_idx.min(), f"idx should be >= {self.allowed_idx.min()}"
        assert idx <= self.allowed_idx.max(), f"idx should be <= {self.allowed_idx.max()}"

        fn = f"im{idx}.npy"
        lensless_fp = os.path.join(self.lensless_dir, fn)
        lensed_fp = os.path.join(self.lensed_dir, fn)
        lensless = np.load(lensless_fp)
        lensed = np.load(lensed_fp)

        return lensless, lensed


class DiffuserCamTestDataset(MeasuredDataset):
    """
    Dataset consisting of lensless and corresponding lensed image. This is the standard dataset used for benchmarking.
    """

    def __init__(
        self,
        data_dir=None,
        n_files=None,
        downsample=2,
    ):
        """
        Dataset consisting of lensless and corresponding lensed image. Default parameters are for the test set of
        `DiffuserCam Lensless Mirflickr Dataset (DLMD) <https://waller-lab.github.io/LenslessLearning/dataset.html>`_.

        Parameters
        ----------
        data_dir : str, optional
            The path to ``DiffuserCam_Test`` dataset, by default looks inside the ``data`` folder.
        n_files : int, optional
            Number of image pairs to load in the dataset , by default use all.
        downsample : int, optional
            Downsample factor of the lensless images, by default 2. Note that the PSF has a resolution of 4x of the images.
        """

        # download dataset if necessary
        if data_dir is None:
            data_dir = os.path.join(
                os.path.dirname(__file__), "..", "..", "data", "DiffuserCam_Test"
            )
        if not os.path.isdir(data_dir):
            main_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
            print("DiffuserCam test set not found for benchmarking.")
            try:
                from torchvision.datasets.utils import download_and_extract_archive
            except ImportError:
                exit()
            msg = "Do you want to download the dataset (3.5GB)?"

            # default to yes if no input is given
            valid = input("%s (Y/n) " % msg).lower() != "n"
            if valid:
                url = "https://drive.switch.ch/index.php/s/D3eRJ6PRljfHoH8/download"
                filename = "DiffuserCam_Test.zip"
                download_and_extract_archive(url, main_dir, filename=filename, remove_finished=True)

        psf_fp = os.path.join(data_dir, "psf.tiff")
        psf, background = load_psf(
            psf_fp,
            downsample=downsample * 4,  # PSF is 4x the resolution of the images
            return_float=True,
            return_bg=True,
            bg_pix=(0, 15),
            flip_ud=True,
            flip_lr=False,
        )

        # transform from BGR to RGB
        transform_BRG2RGB = transforms.Lambda(lambda x: x[..., [2, 1, 0]])

        self.psf = transform_BRG2RGB(torch.from_numpy(psf))

        if n_files is None:
            indices = None
        else:
            indices = range(n_files)

        super().__init__(
            root_dir=data_dir,
            indices=indices,
            background=background,
            downsample=downsample,
            flip=False,
            flip_ud=True,
            flip_lr=False,
            transform_lensless=transform_BRG2RGB,
            transform_lensed=transform_BRG2RGB,
            lensless_fn="diffuser",
            lensed_fn="lensed",
            image_ext="npy",
        )


class SimulatedDatasetTrainableMask(SimulatedFarFieldDataset):
    """
    Dataset of propagated images (through simulation) from a Torch Dataset with learnable mask.
    The `waveprop <https://github.com/ebezzam/waveprop/blob/master/waveprop/simulation.py>`_ package is used for the simulation,
    assuming a far-field propagation and a shift-invariant system with a single point spread function (PSF).
    To ensure autograd compatibility, the dataloader should have ``num_workers=0``.
    """

    def __init__(
        self,
        mask,
        dataset,
        simulator,
        **kwargs,
    ):
        """
        Parameters
        ----------

        mask : :py:class:`lensless.hardware.trainable_mask.TrainableMask`
            Mask to use for simulation. Should be a 4D tensor with shape [1, H, W, C]. Simulation of multi-depth data is not supported yet.
        dataset : :py:class:`torch.utils.data.Dataset`
            Dataset to propagate. Should output images with shape [H, W, C] unless ``dataset_is_CHW`` is ``True`` (and therefore images have the dimension ordering of [C, H, W]).
        simulator : :py:class:`lensless.utils.simulation.FarFieldSimulator`
            Simulator object used on images from ``dataset``. Waveprop simulator to use for the simulation. It is expected to have ``is_torch = True``.
        """

        self._mask = mask

        temp_psf = self._mask.get_psf()
        test_sim = FarFieldSimulator(psf=temp_psf, **simulator.params)
        assert (
            test_sim.conv_dim == simulator.conv_dim
        ).all(), "PSF shape should match simulator shape"
        assert (
            not simulator.quantize
        ), "Simulator should not perform quantization to maintain differentiability. Please set quantize=False"

        super(SimulatedDatasetTrainableMask, self).__init__(dataset, simulator, **kwargs)

    def set_psf(self, psf=None):
        """
        Set the PSF of the simulator.

        Parameters
        ----------
        psf : :py:class:`torch.Tensor`, optional
            PSF to use for the simulation. If ``None``, the PSF of the mask is used.
        """
        if psf is None:
            psf = self._mask.get_psf()
        self.sim.set_point_spread_function(psf)


class HITLDatasetTrainableMask(SimulatedDatasetTrainableMask):
    """
    Dataset of on-the-fly measurements and simulated ground-truth.
    """

    def __init__(
        self,
        rpi_username,
        rpi_hostname,
        celeba_root,
        display_config,
        capture_config,
        mask_center,
        **kwargs,
    ):
        self.rpi_username = rpi_username
        self.rpi_hostname = rpi_hostname
        self.celeba_root = celeba_root
        assert os.path.isdir(self.celeba_root)

        self.display_config = display_config
        self.capture_config = capture_config
        self.mask_center = mask_center

        super(HITLDatasetTrainableMask, self).__init__(**kwargs)

    def __getitem__(self, index):

        # propagate through mask in digital model
        _, lensed = super().__getitem__(index)

        ## measure lensless image
        # get image file path
        idx = self.dataset.indices[index]

        # twice nested as we do train-test split of subset of CelebA
        fn = self.dataset.dataset.dataset.filename[idx]
        fp = os.path.join(self.celeba_root, "celeba", "img_align_celeba", fn)

        # display on screen
        display(
            fp=fp,
            rpi_username=self.rpi_username,
            rpi_hostname=self.rpi_hostname,
            **self.display_config,
        )

        # set mask
        with torch.no_grad():
            subpattern = self._mask.get_vals()
            subpattern_np = subpattern.detach().cpu().numpy().copy()
            pattern = adafruit_sub2full(
                subpattern_np,
                center=self.mask_center,
            )
        set_programmable_mask(
            pattern,
            self._mask.device,
            self.rpi_username,
            self.rpi_hostname,
        )

        # take picture
        _, img = capture(
            rpi_username=self.rpi_username,
            rpi_hostname=self.rpi_hostname,
            verbose=False,
            **self.capture_config,
        )

        # -- normalize
        img = img.astype(np.float32) / img.max()

        # prep
        img = torch.from_numpy(img)
        # -- if [H, W, C] -> [D, H, W, C]
        if len(img.shape) == 3:
            img = img.unsqueeze(0)

        if self.background is not None:
            img = img - self.background

        # flip image x and y if needed
        if self.capture_config.flip:
            img = torch.rot90(img, dims=(-3, -2), k=2)

        # return simulated images (replace simulated with measured)
        return img, lensed


class DiffuserCamMirflickrHF(DualDataset):
    def __init__(
        self,
        split,
        repo_id="bezzam/DiffuserCam-Lensless-Mirflickr-Dataset",
        psf="psf.tiff",
        downsample=2,
        flip_ud=True,
        dtype="float32",
        **kwargs,
    ):
        """
        Parameters
        ----------
        split : str
            Split of the dataset to use: 'train', 'test', or 'all'.
        downsample : int, optional
            Downsample factor of the PSF, which is 4x the resolution of the images, by default 6 for resolution of (180, 320).
        flip_ud : bool, optional
            If True, data is flipped up-down, by default ``True``. Otherwise data is upside-down.
        """

        # get dataset
        self.dataset = load_dataset(repo_id, split=split)

        # get PSF
        psf_fp = hf_hub_download(repo_id=repo_id, filename=psf, repo_type="dataset")
        psf, bg = load_psf(
            psf_fp,
            verbose=False,
            downsample=downsample * 4,
            return_bg=True,
            flip_ud=flip_ud,
            dtype=dtype,
            bg_pix=(0, 15),
        )
        self.psf = torch.from_numpy(psf)

        super(DiffuserCamMirflickrHF, self).__init__(
            flip_ud=flip_ud, downsample=downsample, background=bg, **kwargs
        )

    def __len__(self):
        return len(self.dataset)

    def _get_images_pair(self, idx):
        lensless = np.array(self.dataset[idx]["lensless"])
        lensed = np.array(self.dataset[idx]["lensed"])

        # normalize
        lensless = lensless.astype(np.float32) / 255
        lensed = lensed.astype(np.float32) / 255

        return lensless, lensed


class HFSimulated(DualDataset):
    def __init__(
        self,
        huggingface_repo,
        split,
        n_files=None,
        psf=None,
        downsample=1,
        cache_dir=None,
        single_channel_psf=False,
        flipud=False,
        display_res=None,
        alignment=None,
        sensor="rpi_hq",
        slm="adafruit",
        simulation_config=dict(),
        snr_db=40,
        **kwargs,
    ):
        """
        Wrapper for Hugging Face datasets, where lensless images are simulated from lensed ones.

        This is used for seeing how simulated lensless images compare with real ones.
        """

        if isinstance(split, str):
            if n_files is not None:
                split = f"{split}[0:{n_files}]"
            self.dataset = load_dataset(huggingface_repo, split=split, cache_dir=cache_dir)
        elif isinstance(split, Dataset):
            self.dataset = split
        else:
            raise ValueError("split should be a string or a Dataset object")

        # deduce downsampling factor from the first image
        data_0 = self.dataset[0]
        self.downsample = downsample
        # -- use lensless data just for shape but using lensed data in simulation
        lensless = np.array(data_0["lensless"])
        self.lensless_shape = np.array(lensless.shape[:2]) // self.downsample

        # download PSF from huggingface
        # TODO : assuming psf is not None
        self.multimask = False
        self.convolver = None
        if psf is not None:
            psf_fp = hf_hub_download(repo_id=huggingface_repo, filename=psf, repo_type="dataset")
            psf, _ = load_psf(
                psf_fp,
                shape=self.lensless_shape,
                return_float=True,
                return_bg=True,
                flip_ud=flipud,
                bg_pix=(0, 15),
                single_psf=single_channel_psf,
            )
            self.psf = torch.from_numpy(psf)
            if single_channel_psf:
                # replicate across three channels
                self.psf = self.psf.repeat(1, 1, 1, 3)

            # create convolver object
            self.convolver = RealFFTConvolve2D(self.psf)

        elif "mask_label" in data_0:
            self.multimask = True
            mask_labels = []
            for i in range(len(self.dataset)):
                mask_labels.append(self.dataset[i]["mask_label"])
            mask_labels = list(set(mask_labels))

            # simulate all PSFs
            self.psf = dict()
            for label in mask_labels:
                mask_fp = hf_hub_download(
                    repo_id=huggingface_repo,
                    filename=f"masks/mask_{label}.npy",
                    repo_type="dataset",
                )
                mask_vals = np.load(mask_fp)

                if psf is None:
                    sensor_res = sensor_dict[sensor][SensorParam.RESOLUTION]
                    downsample_fact = min(sensor_res / lensless.shape[:2])
                else:
                    downsample_fact = 1

                mask = AdafruitLCD(
                    initial_vals=torch.from_numpy(mask_vals.astype(np.float32)),
                    sensor=sensor,
                    slm=slm,
                    downsample=downsample_fact,
                    flipud=rotate or flipud,  # TODO separate commands?
                    use_waveprop=simulation_config.get("use_waveprop", False),
                    scene2mask=simulation_config.get("scene2mask", None),
                    mask2sensor=simulation_config.get("mask2sensor", None),
                    deadspace=simulation_config.get("deadspace", True),
                )
                self.psf[label] = mask.get_psf().detach()

                assert (
                    self.psf[label].shape[-3:-1] == lensless.shape[:2]
                ), f"PSF shape should match lensless shape: PSF {self.psf[label].shape[-3:-1]} vs lensless {lensless.shape[:2]}"

            # create convolver object
            self.convolver = RealFFTConvolve2D(self.psf[label])
        assert self.convolver is not None

        self.crop = None
        self.random_flip = None
        self.flipud = flipud
        self.snr_db = snr_db

        self.display_res = display_res
        self.alignment = None
        self.cropped_lensed_shape = None
        if alignment is not None:
            self.alignment = dict(alignment.copy())
            self.alignment["top_left"] = (
                int(self.alignment["top_left"][0] / downsample),
                int(self.alignment["top_left"][1] / downsample),
            )
            self.alignment["height"] = int(self.alignment["height"] / downsample)

            original_aspect_ratio = display_res[1] / display_res[0]
            self.alignment["width"] = int(self.alignment["height"] * original_aspect_ratio)
            self.cropped_lensed_shape = (self.alignment["height"], self.alignment["width"], 3)

        super(HFSimulated, self).__init__(**kwargs)

    def __len__(self):
        return len(self.dataset)

    def _get_images_pair(self, idx):

        # load image
        lensed_np = np.array(self.dataset[idx]["lensed"])
        if self.flipud:
            lensed_np = np.flipud(lensed_np)

        # convert to float
        if lensed_np.dtype == np.uint8:
            lensed_np = lensed_np.astype(np.float32) / 255
        else:
            # 16 bit
            lensed_np = lensed_np.astype(np.float32) / 65535

        # resize if necessary
        if self.cropped_lensed_shape is not None:
            cropped_lensed_np = resize(
                lensed_np, shape=self.cropped_lensed_shape, interpolation=cv2.INTER_NEAREST
            )
            lensed_np = np.zeros(tuple(self.lensless_shape) + (3,), dtype=np.float32)
            lensed_np[
                self.alignment["top_left"][0] : self.alignment["top_left"][0]
                + self.alignment["height"],
                self.alignment["top_left"][1] : self.alignment["top_left"][1]
                + self.alignment["width"],
            ] = cropped_lensed_np

        elif (self.lensless_shape != np.array(lensed_np.shape[:2])).any():

            lensed_np = resize(
                lensed_np, shape=self.lensless_shape, interpolation=cv2.INTER_NEAREST
            )
        lensed = torch.from_numpy(lensed_np)

        # simulate lensless with convolution
        lensed = lensed.unsqueeze(0)  # add batch dimension

        if self.multimask:
            mask_label = self.dataset[idx]["mask_label"]
            self.convolver.set_psf(self.psf[mask_label])
        lensless = self.convolver.convolve(lensed)

        # add noise
        if self.snr_db is not None:
            lensless = add_shot_noise(lensless, self.snr_db)

        if lensless.max() > 1:
            print("CLIPPING!")
            lensless /= lensless.max()

        if self.cropped_lensed_shape:
            return lensless, torch.from_numpy(cropped_lensed_np)
        else:
            return lensless, lensed

    def __getitem__(self, idx):
        lensless, lensed = super().__getitem__(idx)
        if self.multimask:
            mask_label = self.dataset[idx]["mask_label"]
            return lensless, lensed, self.psf[mask_label]
        return lensless, lensed

    def extract_roi(self, reconstruction, lensed=None, axis=(1, 2), **kwargs):
        """
        Extract region of interest from lensless and lensed images.
        """

        n_dim = len(reconstruction.shape)
        assert max(axis) < n_dim, "Axis should be within the dimensions of the reconstruction."

        # add batch dimension
        if n_dim == 3:
            if isinstance(reconstruction, torch.Tensor):
                reconstruction = reconstruction.unsqueeze(0)
            else:
                reconstruction = reconstruction[np.newaxis]
            # increment axis
            axis = (axis[0] + 1, axis[1] + 1)

        # extract
        if self.alignment is not None:
            top_left = self.alignment["top_left"]
            height = self.alignment["height"]
            width = self.alignment["width"]

            # extract according to axis
            index = [slice(None)] * n_dim
            index[axis[0]] = slice(top_left[0], top_left[0] + height)
            index[axis[1]] = slice(top_left[1], top_left[1] + width)
            reconstruction = reconstruction[tuple(index)]

            # rotate if necessary
            angle = self.alignment.get("angle", 0)
            if isinstance(reconstruction, torch.Tensor) and angle:
                reconstruction = F.rotate(reconstruction, angle, expand=False)
            elif angle:
                reconstruction = rotate(reconstruction, angle, axes=axis, reshape=False)

        # remove batch dimension
        if n_dim == 3:
            if isinstance(reconstruction, torch.Tensor):
                reconstruction = reconstruction.squeeze(0)
            else:
                reconstruction = reconstruction[0]

        if lensed is None:
            return reconstruction
        return reconstruction, lensed


class HFDataset(DualDataset):
    def __init__(
        self,
        huggingface_repo,
        split,
        n_files=None,
        psf=None,
        rotate=False,  # just the lensless image
        flipud=False,
        flip_lensed=False,
        downsample=1,
        downsample_lensed=1,
        display_res=None,
        sensor="rpi_hq",
        slm="adafruit",
        alignment=None,
        return_mask_label=False,
        save_psf=False,
        simulation_config=dict(),
        simulate_lensless=False,
        force_rgb=False,
        cache_dir=None,
        single_channel_psf=False,
        random_flip=False,
        **kwargs,
    ):
        """
        Wrapper for lensless datasets on Hugging Face.

        Parameters
        ----------
        huggingface_repo : str
            Hugging Face repository ID.
        split : str or :py:class:`torch.utils.data.Dataset`
            Split of the dataset to use: 'train', 'test', or 'all'. If a Dataset object is given, it is used directly.
        n_files : int, optional
            Number of files to load from the dataset, by default None, namely all.
        psf : str, optional
            File name of the PSF at the repository. If None, it is assumed that there is a mask pattern from which the PSF is simulated, by default None.
        rotate : bool, optional
            If True, lensless images and PSF are rotated 180 degrees. Lensed/original image is not rotated! By default False.
        downsample : float, optional
            Downsample factor of the lensless images, by default 1.
        downsample_lensed : float, optional
            Downsample factor of the lensed images, by default 1.
        display_res : tuple, optional
            Resolution of images when displayed on screen during measurement.
        sensor : str, optional
            If `psf` not provided, the sensor to use for the PSF simulation, by default "rpi_hq".
        slm : str, optional
            If `psf` not provided, the SLM to use for the PSF simulation, by default "adafruit".
        alignment : dict, optional
            Alignment parameters between lensless and lensed data.
            If "top_left", "height", and "width" are provided, the region-of-interest from the reconstruction of ``lensless`` is extracted and ``lensed`` is reshaped to match.
            If "crop" is provided, the region-of-interest is extracted from the simulated lensed image, namely a ``simulation`` configuration should be provided within ``alignment``.
        return_mask_label : bool, optional
            If multimask dataset, return the mask label (True) or the corresponding PSF (False).
        save_psf : bool, optional
            If multimask dataset, save the simulated PSFs.
        random_flip : bool, optional
            If True, randomly flip the lensless images vertically and horizonally with equal probability. By default, no flipping.
        simulation_config : dict, optional
            Simulation parameters for PSF if using a mask pattern.

        """

        if isinstance(split, str):
            if n_files is not None:
                split = f"{split}[0:{n_files}]"
            self.dataset = load_dataset(huggingface_repo, split=split, cache_dir=cache_dir)
        elif isinstance(split, Dataset):
            self.dataset = split
        else:
            raise ValueError("split should be a string or a Dataset object")

        self.rotate = rotate
        self.flipud = flipud
        self.flip_lensed = flip_lensed
        self.display_res = display_res
        self.return_mask_label = return_mask_label
        self.force_rgb = force_rgb  # if some data is not 3D

        # augmentation
        self.random_flip = random_flip

        # deduce downsampling factor from the first image
        data_0 = self.dataset[0]
        self.downsample_lensless = downsample
        self.downsample_lensed = downsample_lensed
        lensless = np.array(data_0["lensless"])

        if self.downsample_lensless != 1.0:
            lensless = resize(lensless, factor=1 / self.downsample_lensless)
        if psf is None:
            sensor_res = sensor_dict[sensor][SensorParam.RESOLUTION]
            downsample_fact = min(sensor_res / lensless.shape[:2])
        else:
            downsample_fact = 1

        # deduce recon shape from original image
        self.alignment = None
        self.crop = None
        if alignment is not None:
            if alignment.get("top_left", None) is not None:
                self.alignment = dict(alignment.copy())
                self.alignment["top_left"] = (
                    int(self.alignment["top_left"][0] / downsample),
                    int(self.alignment["top_left"][1] / downsample),
                )
                self.alignment["height"] = int(self.alignment["height"] / downsample)

                original_aspect_ratio = display_res[1] / display_res[0]
                self.alignment["width"] = int(self.alignment["height"] * original_aspect_ratio)

            if alignment.get("topright", None) is not None:
                # typo in original configuration
                self.alignment = dict(alignment.copy())
                self.alignment["top_left"] = (
                    int(self.alignment["topright"][0] / downsample),
                    int(self.alignment["topright"][1] / downsample),
                )
                self.alignment["height"] = int(self.alignment["height"] / downsample)

                original_aspect_ratio = display_res[1] / display_res[0]
                self.alignment["width"] = int(self.alignment["height"] * original_aspect_ratio)

            elif alignment.get("crop", None) is not None:
                self.crop = dict(alignment["crop"].copy())
                self.crop["vertical"][0] = int(self.crop["vertical"][0] / downsample)
                self.crop["vertical"][1] = int(self.crop["vertical"][1] / downsample)
                self.crop["horizontal"][0] = int(self.crop["horizontal"][0] / downsample)
                self.crop["horizontal"][1] = int(self.crop["horizontal"][1] / downsample)

        # download all masks
        # TODO: reshape directly with lensless image shape
        self.multimask = False
        if psf is not None:
            # download PSF from huggingface
            psf_fp = hf_hub_download(repo_id=huggingface_repo, filename=psf, repo_type="dataset")
            psf, _ = load_psf(
                psf_fp,
                shape=lensless.shape,
                return_float=True,
                return_bg=True,
                flip=self.rotate,
                flip_ud=flipud,
                bg_pix=(0, 15),
                force_rgb=force_rgb,
                single_psf=single_channel_psf,
            )
            self.psf = torch.from_numpy(psf)
            if single_channel_psf:
                # replicate across three channels
                self.psf = self.psf.repeat(1, 1, 1, 3)

        elif "mask_label" in data_0:
            self.multimask = True
            mask_labels = []
            for i in range(len(self.dataset)):
                mask_labels.append(self.dataset[i]["mask_label"])
            mask_labels = list(set(mask_labels))

            # simulate all PSFs
            self.psf = dict()
            for label in mask_labels:
                mask_fp = hf_hub_download(
                    repo_id=huggingface_repo,
                    filename=f"masks/mask_{label}.npy",
                    repo_type="dataset",
                )
                mask_vals = np.load(mask_fp)
                mask = AdafruitLCD(
                    initial_vals=torch.from_numpy(mask_vals.astype(np.float32)),
                    sensor=sensor,
                    slm=slm,
                    downsample=downsample_fact,
                    flipud=self.rotate or flipud,  # TODO separate commands?
                    use_waveprop=simulation_config.get("use_waveprop", False),
                    scene2mask=simulation_config.get("scene2mask", None),
                    mask2sensor=simulation_config.get("mask2sensor", None),
                    deadspace=simulation_config.get("deadspace", True),
                )
                self.psf[label] = mask.get_psf().detach()

                assert (
                    self.psf[label].shape[-3:-1] == lensless.shape[:2]
                ), f"PSF shape should match lensless shape: PSF {self.psf[label].shape[-3:-1]} vs lensless {lensless.shape[:2]}"

                if save_psf:
                    # same viewable image of PSF
                    save_image(self.psf[label].squeeze().cpu().numpy(), f"psf_{label}.png")

        else:

            # single mask pattern
            mask_fp = hf_hub_download(
                repo_id=huggingface_repo, filename="mask_pattern.npy", repo_type="dataset"
            )
            mask_vals = np.load(mask_fp)
            mask = AdafruitLCD(
                initial_vals=torch.from_numpy(mask_vals.astype(np.float32)),
                sensor=sensor,
                slm=slm,
                downsample=downsample_fact,
                flipud=self.rotate or flipud,  # TODO separate commands?
                use_waveprop=simulation_config.get("use_waveprop", False),
                scene2mask=simulation_config.get("scene2mask", None),
                mask2sensor=simulation_config.get("mask2sensor", None),
                deadspace=simulation_config.get("deadspace", True),
            )
            self.psf = mask.get_psf().detach()
            assert (
                self.psf.shape[-3:-1] == lensless.shape[:2]
            ), "PSF shape should match lensless shape"

            if save_psf:
                # same viewable image of PSF
                save_image(self.psf.squeeze().cpu().numpy(), "psf.png")

        # create simulator
        self.simulate_lensless = simulate_lensless
        if simulate_lensless:
            assert (
                alignment is not None and alignment.get("simulation") is not None
            ), "Need simulation parameters for lensless images"
        self.simulator = None
        if alignment is not None and "simulation" in alignment:
            simulation_config = dict(alignment["simulation"].copy())
            simulation_config["output_dim"] = tuple(self.psf.shape[-3:-1])
            if simulation_config.get("vertical_shift", None) is not None:
                simulation_config["vertical_shift"] = int(
                    simulation_config["vertical_shift"] / downsample
                )
            if simulation_config.get("horizontal_shift", None) is not None:
                simulation_config["horizontal_shift"] = int(
                    simulation_config["horizontal_shift"] / downsample
                )
            simulator = FarFieldSimulator(
                psf=self.psf if self.simulate_lensless else None,
                is_torch=True,
                **simulation_config,
            )
            self.simulator = simulator

        super(HFDataset, self).__init__(**kwargs)

    def __len__(self):
        return len(self.dataset)

    def _get_images_pair(self, idx):

        # load image
        lensless_np = np.array(self.dataset[idx]["lensless"])
        lensed_np = np.array(self.dataset[idx]["lensed"])

        if self.force_rgb:
            if len(lensless_np.shape) == 2:
                warnings.warn(f"Converting lensless[{idx}] to RGB")
                lensless_np = np.stack([lensless_np] * 3, axis=2)
            elif len(lensless_np.shape) == 3:
                pass
            else:
                raise ValueError(f"lensless[{idx}] should be 2D or 3D")

            if len(lensed_np.shape) == 2:
                warnings.warn(f"Converting lensed[{idx}] to RGB")
                lensed_np = np.stack([lensed_np] * 3, axis=2)
            elif len(lensed_np.shape) == 3:
                pass

        # convert to float
        if lensless_np.dtype == np.uint8:
            lensless_np = lensless_np.astype(np.float32) / 255
            lensed_np = lensed_np.astype(np.float32) / 255
        else:
            # 16 bit
            lensless_np = lensless_np.astype(np.float32) / 65535
            lensed_np = lensed_np.astype(np.float32) / 65535

        # downsample if necessary
        if self.downsample_lensless != 1.0:
            lensless_np = resize(
                lensless_np, factor=1 / self.downsample_lensless, interpolation=cv2.INTER_NEAREST
            )

        lensless = lensless_np
        lensed = lensed_np

        if self.simulator is not None:
            # convert to torch
            lensless = torch.from_numpy(lensless_np)
            lensed = torch.from_numpy(lensed_np)

            # project original image to lensed space
            with torch.no_grad():

                if self.simulate_lensless:
                    lensless, lensed = self.simulator.propagate_image(
                        lensed, return_object_plane=True
                    )
                else:
                    lensed = self.simulator.propagate_image(lensed, return_object_plane=True)

        elif self.alignment is not None:
            lensed = resize(
                lensed_np,
                shape=(self.alignment["height"], self.alignment["width"], 3),
                interpolation=cv2.INTER_NEAREST,
            )
        elif self.display_res is not None:
            lensed = resize(
                lensed_np, shape=(*self.display_res, 3), interpolation=cv2.INTER_NEAREST
            )
        elif self.downsample_lensed != 1.0:
            lensed = resize(
                lensed_np,
                factor=1 / self.downsample_lensed,
                interpolation=cv2.INTER_NEAREST,
            )

        return lensless, lensed

    def __getitem__(self, idx):
        lensless, lensed = super().__getitem__(idx)
        if not self.simulate_lensless:
            if self.rotate:
                lensless = torch.rot90(lensless, dims=(-3, -2), k=2)
            if self.flipud:
                lensless = torch.flip(lensless, dims=(-3,))

        if self.flip_lensed:
            if self.rotate:
                lensed = torch.rot90(lensed, dims=(-3, -2), k=2)
            if self.flipud:
                lensed = torch.flip(lensed, dims=(-3,))

        if self.multimask:
            mask_label = self.dataset[idx]["mask_label"]

        flip_lr = False
        flip_ud = False
        if self.random_flip:
            flip_lr = torch.rand(1) > 0.5
            flip_ud = torch.rand(1) > 0.5

            if self.multimask:
                psf_aug = self.psf[mask_label].clone()
            else:
                psf_aug = self.psf.clone()

            if flip_lr:
                lensless = torch.flip(lensless, dims=(-2,))
                lensed = torch.flip(lensed, dims=(-2,))
                psf_aug = torch.flip(psf_aug, dims=(-2,))
            if flip_ud:
                lensless = torch.flip(lensless, dims=(-3,))
                lensed = torch.flip(lensed, dims=(-3,))
                psf_aug = torch.flip(psf_aug, dims=(-3,))

        # return corresponding PSF
        if self.multimask:
            if self.return_mask_label:
                return lensless, lensed, mask_label
            else:
                if not self.random_flip:
                    return lensless, lensed, self.psf[mask_label]
                else:
                    return lensless, lensed, psf_aug, flip_lr, flip_ud
        else:
            if not self.random_flip:
                return lensless, lensed
            else:
                return lensless, lensed, psf_aug, flip_lr, flip_ud

    def extract_roi(
        self,
        reconstruction,
        lensed=None,
        axis=(1, 2),
        flip_lr=None,
        flip_ud=None,
        rotate_aug=False,
        shift_aug=None,
    ):
        """
        Parameters
        ----------
        flip_lr : torch.Tensor, optional
            Tensor of booleans indicating whether to flip the reconstruction left-right, by default None.
        flip_ud : bool, optional
            Tensor of booleans indicating whether to flip the reconstruction up-down, by default None.
        """
        n_dim = len(reconstruction.shape)
        assert max(axis) < n_dim, "Axis should be within the dimensions of the reconstruction."

        # add batch dimension
        if n_dim == 3:
            if isinstance(reconstruction, torch.Tensor):
                reconstruction = reconstruction.unsqueeze(0)
                if lensed is not None:
                    lensed = lensed.unsqueeze(0)
            else:
                reconstruction = reconstruction[np.newaxis]
                if lensed is not None:
                    lensed = lensed[np.newaxis]
            # increment axis
            axis = (axis[0] + 1, axis[1] + 1)

        # flip/rotate before alignment, as alignment parameters are assuming no flip/rotate
        if flip_lr is not None:
            flip_lr = flip_lr.squeeze().tolist()
            if isinstance(reconstruction, torch.Tensor):
                reconstruction[flip_lr] = torch.flip(reconstruction[flip_lr], dims=(axis[1],))
                if lensed is not None:
                    lensed[flip_lr] = torch.flip(lensed[flip_lr], dims=(axis[1],))
            else:
                reconstruction[flip_lr] = np.flip(reconstruction[flip_lr], axis=axis[1])
                if lensed is not None:
                    lensed[flip_lr] = np.flip(lensed[flip_lr], axis=axis[1])
        if flip_ud is not None:
            flip_ud = flip_ud.squeeze().tolist()
            if isinstance(reconstruction, torch.Tensor):
                reconstruction[flip_ud] = torch.flip(reconstruction[flip_ud], dims=(axis[0],))
                if lensed is not None:
                    lensed[flip_ud] = torch.flip(lensed[flip_ud], dims=(axis[0],))
            else:
                reconstruction[flip_ud] = np.flip(reconstruction[flip_ud], axis=axis[0])
                if lensed is not None:
                    lensed[flip_ud] = np.flip(lensed[flip_ud], axis=axis[0])
        if rotate_aug:
            assert isinstance(rotate_aug, float)
            if isinstance(reconstruction, torch.Tensor):
                assert axis == (-2, -1), "Only ...HW rotation is supported for torch.Tensor"
                reconstruction = F.rotate(reconstruction, -rotate_aug, expand=False)
                if lensed is not None:
                    lensed = F.rotate(lensed, -rotate_aug, expand=False)
            else:
                reconstruction = rotate(reconstruction, angle=-rotate_aug, axes=axis, reshape=False)
                if lensed is not None:
                    lensed = rotate(lensed, angle=-rotate_aug, axes=axis, reshape=False)
        if shift_aug is not None:
            assert isinstance(shift_aug, tuple)
            neg_shift = (-shift_aug[0], -shift_aug[1])
            reconstruction = shift_with_pad(reconstruction, neg_shift, axis=axis)
            if lensed is not None:
                lensed = shift_with_pad(lensed, neg_shift, axis=axis)

        if self.alignment is not None:
            top_left = self.alignment["top_left"]
            height = self.alignment["height"]
            width = self.alignment["width"]

            # extract according to axis
            index = [slice(None)] * n_dim
            index[axis[0]] = slice(top_left[0], top_left[0] + height)
            index[axis[1]] = slice(top_left[1], top_left[1] + width)
            reconstruction = reconstruction[tuple(index)]

            # rotate if necessary
            angle = self.alignment.get("angle", 0)
            if isinstance(reconstruction, torch.Tensor) and angle:
                reconstruction = F.rotate(reconstruction, angle, expand=False)
            elif angle:
                reconstruction = rotate(reconstruction, angle, axes=axis, reshape=False)

        elif self.crop is not None:
            vertical = self.crop["vertical"]
            horizontal = self.crop["horizontal"]

            # extract according to axis
            index = [slice(None)] * n_dim
            index[axis[0]] = slice(vertical[0], vertical[1])
            index[axis[1]] = slice(horizontal[0], horizontal[1])
            reconstruction = reconstruction[tuple(index)]
            if lensed is not None:
                lensed = lensed[tuple(index)]

        # flip/rotate back
        if flip_lr is not None:
            if isinstance(reconstruction, torch.Tensor):
                reconstruction[flip_lr] = torch.flip(reconstruction[flip_lr], dims=(axis[1],))
                if lensed is not None:
                    lensed[flip_lr] = torch.flip(lensed[flip_lr], dims=(axis[1],))
            else:
                reconstruction[flip_lr] = np.flip(reconstruction[flip_lr], axis=axis[1])
                if lensed is not None:
                    lensed[flip_lr] = np.flip(lensed[flip_lr], axis=axis[1])
        if flip_ud is not None:
            if isinstance(reconstruction, torch.Tensor):
                reconstruction[flip_ud] = torch.flip(reconstruction[flip_ud], dims=(axis[0],))
                if lensed is not None:
                    lensed[flip_ud] = torch.flip(lensed[flip_ud], dims=(axis[0],))
            else:
                reconstruction[flip_ud] = np.flip(reconstruction[flip_ud], axis=axis[0])
                if lensed is not None:
                    lensed[flip_ud] = np.flip(lensed[flip_ud], axis=axis[0])
        if rotate_aug:
            if isinstance(reconstruction, torch.Tensor):
                assert axis == (-2, -1), "Only ...HW rotation is supported for torch.Tensor"
                reconstruction = F.rotate(reconstruction, rotate_aug, expand=False)
                if lensed is not None:
                    lensed = F.rotate(lensed, rotate_aug, expand=False)
            else:
                reconstruction = rotate(reconstruction, angle=rotate_aug, axes=axis, reshape=False)
                if lensed is not None:
                    lensed = rotate(lensed, angle=rotate_aug, axes=axis, reshape=False)

        if shift_aug is not None:
            reconstruction = shift_with_pad(reconstruction, shift_aug, axis=axis)
            if lensed is not None:
                lensed = shift_with_pad(lensed, shift_aug, axis=axis)

        # remove batch dimension
        if n_dim == 3:
            if isinstance(reconstruction, torch.Tensor):
                reconstruction = reconstruction.squeeze(0)
                if lensed is not None:
                    lensed = lensed.squeeze(0)
            else:
                reconstruction = reconstruction[0]
                if lensed is not None:
                    lensed = lensed[0]

        if self.alignment is None and lensed is not None:
            return reconstruction, lensed
        else:
            return reconstruction


def simulate_dataset(config, generator=None):
    """
    Prepare datasets for training and testing.

    Parameters
    ----------
    config : omegaconf.DictConfig
        Configuration, e.g. from Hydra. See ``scripts/recon/train_learning_based.py`` for an example that uses this function.
    generator : torch.Generator, optional
        Random number generator, by default ``None``.
    """

    if "cuda" in config.torch_device and torch.cuda.is_available():
        device = config.torch_device
    else:
        device = "cpu"

    # -- prepare PSF
    psf = None
    if config.trainable_mask.mask_type is None or config.trainable_mask.initial_value == "psf":
        psf_fp = os.path.join(get_original_cwd(), config.files.psf)
        psf, _ = load_psf(
            psf_fp,
            downsample=config.files.downsample,
            return_float=True,
            return_bg=True,
            bg_pix=(0, 15),
        )
        if config.files.diffusercam_psf:
            transform_BRG2RGB = transforms.Lambda(lambda x: x[..., [2, 1, 0]])
            psf = transform_BRG2RGB(torch.from_numpy(psf))

        # drop depth dimension
        psf = psf.to(device)

    else:
        # training mask / PSF
        mask = prep_trainable_mask(config, psf)
        psf = mask.get_psf().to(device)

    # -- load dataset
    pre_transform = None
    transforms_list = [transforms.ToTensor()]
    data_path = os.path.join(get_original_cwd(), "data")
    if config.simulation.grayscale:
        transforms_list.append(transforms.Grayscale())

    if config.files.dataset == "mnist":
        transform = transforms.Compose(transforms_list)
        train_ds = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
        test_ds = datasets.MNIST(root=data_path, train=False, download=True, transform=transform)

    elif config.files.dataset == "fashion_mnist":
        transform = transforms.Compose(transforms_list)
        train_ds = datasets.FashionMNIST(
            root=data_path, train=True, download=True, transform=transform
        )
        test_ds = datasets.FashionMNIST(
            root=data_path, train=False, download=True, transform=transform
        )
    elif config.files.dataset == "cifar10":
        transform = transforms.Compose(transforms_list)
        train_ds = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
        test_ds = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)

    elif config.files.dataset == "CelebA":
        root = config.files.celeba_root
        data_path = os.path.join(root, "celeba")
        assert os.path.isdir(
            data_path
        ), f"Data path {data_path} does not exist. Make sure you download the CelebA dataset and provide the parent directory as 'config.files.celeba_root'. Download link: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html"
        transform = transforms.Compose(transforms_list)
        if config.files.n_files is None:
            train_ds = datasets.CelebA(
                root=root, split="train", download=False, transform=transform
            )
            test_ds = datasets.CelebA(root=root, split="test", download=False, transform=transform)
        else:
            ds = datasets.CelebA(root=root, split="all", download=False, transform=transform)

            ds = Subset(ds, np.arange(config.files.n_files))

            train_size = int((1 - config.files.test_size) * len(ds))
            test_size = len(ds) - train_size
            train_ds, test_ds = torch.utils.data.random_split(
                ds, [train_size, test_size], generator=generator
            )
    else:
        raise NotImplementedError(f"Dataset {config.files.dataset} not implemented.")

    if config.files.dataset != "CelebA":
        if config.files.n_files is not None:
            train_size = int((1 - config.files.test_size) * config.files.n_files)
            test_size = config.files.n_files - train_size
            train_ds = Subset(train_ds, np.arange(train_size))
            test_ds = Subset(test_ds, np.arange(test_size))

    # convert PSF
    if config.simulation.grayscale and not is_grayscale(psf):
        psf = rgb2gray(psf)

    # check if gpu is available
    device_conv = config.torch_device
    if device_conv == "cuda" and torch.cuda.is_available():
        device_conv = "cuda"
    else:
        device_conv = "cpu"

    # create simulator
    simulator = FarFieldSimulator(
        psf=psf,
        is_torch=True,
        **config.simulation,
    )

    # create Pytorch dataset and dataloader
    crop = config.files.crop.copy() if config.files.crop is not None else None
    if mask is None:
        train_ds_prop = SimulatedFarFieldDataset(
            dataset=train_ds,
            simulator=simulator,
            dataset_is_CHW=True,
            device_conv=device_conv,
            flip=config.simulation.flip,
            vertical_shift=config.files.vertical_shift,
            horizontal_shift=config.files.horizontal_shift,
            crop=crop,
            downsample=config.files.downsample,
            pre_transform=pre_transform,
        )
        test_ds_prop = SimulatedFarFieldDataset(
            dataset=test_ds,
            simulator=simulator,
            dataset_is_CHW=True,
            device_conv=device_conv,
            flip=config.simulation.flip,
            vertical_shift=config.files.vertical_shift,
            horizontal_shift=config.files.horizontal_shift,
            crop=crop,
            downsample=config.files.downsample,
            pre_transform=pre_transform,
        )
    else:
        if config.measure is not None:

            train_ds_prop = HITLDatasetTrainableMask(
                rpi_username=config.measure.rpi_username,
                rpi_hostname=config.measure.rpi_hostname,
                celeba_root=config.files.celeba_root,
                display_config=config.measure.display,
                capture_config=config.measure.capture,
                mask_center=config.trainable_mask.ap_center,
                dataset=train_ds,
                mask=mask,
                simulator=simulator,
                dataset_is_CHW=True,
                device_conv=device_conv,
                flip=config.simulation.flip,
                vertical_shift=config.files.vertical_shift,
                horizontal_shift=config.files.horizontal_shift,
                crop=crop,
                downsample=config.files.downsample,
                pre_transform=pre_transform,
            )

            test_ds_prop = HITLDatasetTrainableMask(
                rpi_username=config.measure.rpi_username,
                rpi_hostname=config.measure.rpi_hostname,
                celeba_root=config.files.celeba_root,
                display_config=config.measure.display,
                capture_config=config.measure.capture,
                mask_center=config.trainable_mask.ap_center,
                dataset=test_ds,
                mask=mask,
                simulator=simulator,
                dataset_is_CHW=True,
                device_conv=device_conv,
                flip=config.simulation.flip,
                vertical_shift=config.files.vertical_shift,
                horizontal_shift=config.files.horizontal_shift,
                crop=crop,
                downsample=config.files.downsample,
                pre_transform=pre_transform,
            )

        else:

            train_ds_prop = SimulatedDatasetTrainableMask(
                dataset=train_ds,
                mask=mask,
                simulator=simulator,
                dataset_is_CHW=True,
                device_conv=device_conv,
                flip=config.simulation.flip,
                vertical_shift=config.files.vertical_shift,
                horizontal_shift=config.files.horizontal_shift,
                crop=crop,
                downsample=config.files.downsample,
                pre_transform=pre_transform,
            )
            test_ds_prop = SimulatedDatasetTrainableMask(
                dataset=test_ds,
                mask=mask,
                simulator=simulator,
                dataset_is_CHW=True,
                device_conv=device_conv,
                flip=config.simulation.flip,
                vertical_shift=config.files.vertical_shift,
                horizontal_shift=config.files.horizontal_shift,
                crop=crop,
                downsample=config.files.downsample,
                pre_transform=pre_transform,
            )

    return train_ds_prop, test_ds_prop, mask


class MyDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
