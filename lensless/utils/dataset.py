# #############################################################################
# dataset.py
# =================
# Authors :
# Yohann PERRON [yohann.perron@gmail.com]
# #############################################################################

import numpy as np
import glob
import os
import torch
from abc import abstractmethod
from torch.utils.data import Dataset
from torchvision import transforms
from waveprop.simulation import FarFieldSimulator

from lensless.utils.io import load_image, load_psf
from lensless.utils.image import resize


class DualDataset(Dataset):
    """
    Abstract class for defining a dataset of paired lensed and lensless images.
    """

    def __init__(
        self,
        indices=None,
        background=None,
        downsample=1,
        flip=False,
        transform_lensless=None,
        transform_lensed=None,
        **kwargs,
    ):
        """
        Dataset consisting of lensless and corresponding lensed image.

        Parameters
        ----------
            indexes : range or int or None
                Indexes of the images to use in the dataset (if integer, it should be interpreted as range(indexes)), by default None.
            background : :py:class:`~torch.Tensor` or None, optional
                If not ``None``, background is removed from lensless images, by default ``None``.
            downsample : int, optional
                Downsample factor of the lensless images, by default 1.
            flip : bool, optional
                If ``True``, lensless images are flipped, by default ``False``.
            transform_lensless : PyTorch Transform or None, optional
                Transform to apply to the lensless images, by default ``None``.
            transform_lensed : PyTorch Transform or None, optional
                Transform to apply to the lensed images, by default ``None``.
        """
        if isinstance(indices, int):
            indices = range(indices)
        self.indices = indices
        self.background = background
        self.downsample = downsample
        self.flip = flip
        self.transform_lensless = transform_lensless
        self.transform_lensed = transform_lensed

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

        # flip image x and y if needed
        if self.flip:
            lensless = torch.rot90(lensless, dims=(-3, -2))
            lensed = torch.rot90(lensed, dims=(-3, -2))
        if self.transform_lensless:
            lensless = self.transform_lensless(lensless)
        if self.transform_lensed:
            lensed = self.transform_lensed(lensed)

        return lensless, lensed


class SimulatedDataset(DualDataset):
    """
    Dataset of propagated images (through simulation) from a Torch Dataset.
    The `waveprop <https://github.com/ebezzam/waveprop/blob/master/waveprop/simulation.py>`_ package is used for the simulation,
    assuming a far-field propagation and a shift-invariant system with a single point spread function (PSF).

    """

    def __init__(
        self,
        psf,
        dataset,
        pre_transform=None,
        dataset_is_CHW=False,
        flip=False,
        **kwargs,
    ):
        """
        Parameters
        ----------

        psf : torch.Tensor
            PSF to use for simulate. Should be a 4D tensor with shape [1, H, W, C]. Simulation of multi-depth data is not supported yet.
        dataset : torch.utils.data.Dataset
            Dataset to propagate. Should output images with shape [H, W, C] unless ``dataset_is_CHW`` is ``True`` (and therefore images have the dimension ordering of [C, H, W]).
        pre_transform : PyTorch Transform or None, optional
            Transform to apply to the images before simulation, by default ``None``.
        dataset_is_CHW : bool, optional
            If True, the input dataset is expected to output images with shape [C, H, W], by default ``False``.
        flip : bool, optional
                If True,images are flipped beffore the simulation, by default ``False``..
        """

        # we do the flipping before the simualtion
        super(SimulatedDataset, self).__init__(flip=False, **kwargs)

        assert isinstance(dataset, Dataset)
        self.dataset = dataset
        self.n_files = len(dataset)

        self.dataset_is_CHW = dataset_is_CHW
        self._pre_transform = pre_transform
        self.flip = flip

        psf = psf.squeeze().movedim(-1, 0)

        # initialize simulator
        self.sim = FarFieldSimulator(psf=psf, is_torch=True, **kwargs)

    def get_image(self, index):
        return self.dataset[index]

    def _get_images_pair(self, index):
        # load image
        img, _ = self.get_image(index)
        if not self.dataset_is_CHW:
            img = img.movedim(-1, 0)
        if self.flip:
            img = torch.rot90(img, dims=(-3, -2))
        if self._pre_transform is not None:
            img = self._pre_transform(img)

        lensless, lensed = self.sim.propagate(img, return_object_plane=True)
        lensless = lensless.moveaxis(-3, -1)
        lensed = lensed.moveaxis(-3, -1)
        return lensless, lensed

    def __len__(self):
        if self.indices is None:
            return self.n_files
        else:
            return len([x for x in self.indices if x < self.n_files])


class LenslessDataset(DualDataset):
    """
    Dataset consisting of lensless image captured from a screen and the corresponding image shown on screen (not necessarily captured with a lens).
    It can be used with a PyTorch DataLoader to load a batch of lensless and corresponding lensed images.
    Unless the setup is perfectly calibrated, one should expect to have to use ``transform_lensed`` to adjust the alignement and rotation.
    """

    def __init__(
        self,
        root_dir,
        lensless_fn="diffuser",
        original_fn="lensed",
        image_ext="npy",
        original_ext=None,
        downsample=1,
        **kwargs,
    ):
        """
        Dataset consisting of lensless image captured from a screen and the corresponding image shown on screen.

        Parameters
        ----------
        root_dir : str
            Path to the test dataset. It is expected to contain two folders: one of lensless images and one of original images.
        lensless_fn : str, optional
            Name of the folder containing the lensless images, by default "diffuser".
        lensed_fn : str, optional
            Name of the folder containing the lensed images, by default "lensed".
        image_ext : str, optional
            Extension of the images, by default "npy".
        original_ext : str, optional
            Extension of the original image if different from lenless, by default None.
        downsample : int, optional
            Downsample factor of the lensless images, by default 1.
        """
        super(LenslessDataset, self).__init__(downsample=1, **kwargs)
        self.pre_downsample = downsample

        self.root_dir = root_dir
        self.lensless_dir = os.path.join(root_dir, lensless_fn)
        self.original_dir = os.path.join(root_dir, original_fn)
        self.image_ext = image_ext.lower()
        self.original_ext = original_ext.lower() if original_ext is not None else image_ext.lower()

        files = glob.glob(os.path.join(self.lensless_dir, "*." + image_ext))
        files.sort()
        self.files = [os.path.basename(fn) for fn in files]

        if len(self.files) == 0:
            raise FileNotFoundError(
                f"No files found in {self.lensless_dir} with extension {image_ext}"
            )

        # initialize simulator
        self.sim = FarFieldSimulator(is_torch=True, **kwargs)

    def __len__(self):
        if self.indices is None:
            return len(self.files)
        else:
            return len([i for i in self.indices if i < len(self.files)])

    def _get_images_pair(self, idx):
        if self.image_ext == "npy":
            lensless_fp = os.path.join(self.lensless_dir, self.files[idx])
            original_fp = os.path.join(self.original_dir, self.files[idx])
            lensless = np.load(lensless_fp)
            lensless = resize(lensless, factor=1 / self.downsample)
            original = np.load(original_fp[:-3] + self.original_ext)
        else:
            # more standard image formats: png, jpg, tiff, etc.
            lensless_fp = os.path.join(self.lensless_dir, self.files[idx])
            original_fp = os.path.join(self.original_dir, self.files[idx])
            lensless = load_image(lensless_fp, downsample=self.pre_downsample)
            original = load_image(
                original_fp[:-3] + self.original_ext, downsample=self.pre_downsample
            )

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
            lensed = self.sim.propagate(original.moveaxis(-1, -3)).moveaxis(-3, -1)

        return lensless, lensed


class ParallelDataset(DualDataset):
    """
    Dataset consisting of lensless and corresponding lensed image.
    It can be used with a PyTorch DataLoader to load a batch of lensless and corresponding lensed images.
    Unless the setup is perfectly calibrated, one should expect to have to use ``transform_lensed`` to adjust the alignement and rotation.
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

        super(ParallelDataset, self).__init__(**kwargs)

        self.root_dir = root_dir
        self.lensless_dir = os.path.join(root_dir, lensless_fn)
        self.lensed_dir = os.path.join(root_dir, lensed_fn)
        self.image_ext = image_ext.lower()

        files = glob.glob(os.path.join(self.lensless_dir, "*." + image_ext))
        files.sort()
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
        if self.image_ext == "npy":
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


class DiffuserCamTestDataset(ParallelDataset):
    """
    Dataset consisting of lensless and corresponding lensed image. This is the standard dataset used for benchmarking.
    """

    def __init__(
        self,
        data_dir="data",
        n_files=200,
        downsample=2,
    ):
        """
        Dataset consisting of lensless and corresponding lensed image. Default parameters are for the test set
        `DiffuserCam Lensless Mirflickr Dataset (DLMD) <https://waller-lab.github.io/LenslessLearning/dataset.html>`_.

        Parameters
        ----------
        data_dir : str, optional
            The path to the folder containing the DiffuserCam_Test dataset, by default "data".
        n_files : int, optional
            Number of image pair to load in the dataset , by default 200.
        downsample : int, optional
            Downsample factor of the lensless images, by default 8.
        """
        # download dataset if necessary
        main_dir = data_dir
        data_dir = os.path.join(data_dir, "DiffuserCam_Test")
        if not os.path.isdir(data_dir):
            print("No dataset found for benchmarking.")
            try:
                from torchvision.datasets.utils import download_and_extract_archive
            except ImportError:
                exit()
            msg = "Do you want to download the sample dataset (3.5GB)?"

            # default to yes if no input is given
            valid = input("%s (Y/n) " % msg).lower() != "n"
            if valid:
                url = "https://drive.switch.ch/index.php/s/D3eRJ6PRljfHoH8/download"
                filename = "DiffuserCam_Test.zip"
                download_and_extract_archive(url, main_dir, filename=filename, remove_finished=True)

        psf_fp = os.path.join(data_dir, "psf.tiff")
        psf, background = load_psf(
            psf_fp,
            downsample=downsample,
            return_float=True,
            return_bg=True,
            bg_pix=(0, 15),
        )

        # transform from BGR to RGB
        transform_BRG2RGB = transforms.Lambda(lambda x: x[..., [2, 1, 0]])

        self.psf = transform_BRG2RGB(torch.from_numpy(psf))

        super().__init__(
            root_dir=data_dir,
            indices=range(n_files),
            background=background,
            downsample=downsample / 4,
            flip=False,
            transform_lensless=transform_BRG2RGB,
            transform_lensed=transform_BRG2RGB,
            lensless_fn="diffuser",
            lensed_fn="lensed",
            image_ext="npy",
        )
