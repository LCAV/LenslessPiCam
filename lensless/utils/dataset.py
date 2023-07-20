import numpy as np
import glob
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from waveprop.simulation import FarFieldSimulator

from lensless.io import load_image, load_psf
from lensless.util import resize


class SimulatedDataset(Dataset):
    """
    Dataset of propagated images from a sigle image torch Dataset.
    """

    def __init__(
        self,
        psf,
        dataset,
        downsample=4,
        flip=False,
        pre_transform=None,
        transform_lensless=None,
        transform_lensed=None,
        dataset_is_CHW=False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        psf : torch.Tensor
            PSF to use for propagation. Should be a 4D tensor with shape 1 H W C.
            For no simated dataset don't support 3D reconstruction.
        dataset : torch.utils.data.Dataset
            Dataset to propagate. Should output images with shape H W C unless dataset_is_CHW is True.
        """

        super(Dataset, self).__init__()

        assert isinstance(dataset, Dataset)
        self.dataset = dataset
        self.n_files = len(dataset)

        self.dataset_is_CHW = dataset_is_CHW
        self.flip = flip
        self._pre_transform = pre_transform
        self._transform_lensless = transform_lensless
        self._transform_lensed = transform_lensed

        psf = psf.squeeze().movedim(-1, 0)

        # initialize simulator
        self.sim = FarFieldSimulator(psf=psf, is_torch=True, **kwargs)

    def get_image(self, index):
        return self.dataset[index]

    def __getitem__(self, index):
        # load image
        img, label = self.get_image(index)
        if not self.dataset_is_CHW:
            img = img.movedim(-1, 0)
        if self.flip:
            img = torch.rot90(img, dims=(-3, -2))
        if self._pre_transform is not None:
            img = self._pre_transform(img)

        lensless, lensed = self.sim.propagate(img, return_object_plane=True)
        if self._transform_lensless is not None:
            lensless = self._transform_lensless(lensless)
        if self._transform_lensed is not None:
            lensed = self._transform_lensed(lensed)

        # convert to DHWC
        lensless = lensless.movedim(0, -1)
        lensed = lensed.movedim(0, -1)
        lensless = lensless[None, ...]
        lensed = lensed[None, ...]
        return lensless, lensed

    def __len__(self):
        return self.n_files


class ParallelDataset(Dataset):
    """
    Dataset consisting of lensless and corresponding lensed image.
    It can be used with a PyTorch DataLoader to load a batch of lensless and corresponding lensed images.
    """

    def __init__(
        self,
        root_dir,
        n_files=False,
        background=None,
        downsample=4,
        flip=False,
        transform_lensless=None,
        transform_lensed=None,
        lensless_fn="diffuser",
        lensed_fn="lensed",
        image_ext="npy",
        **kwargs,
    ):
        """
        Dataset consisting of lensless and corresponding lensed image. Default parameters are for the DiffuserCam
        Lensless Mirflickr Dataset (DLMD).
        Parameters
        ----------
            root_dir : str
                Path to the test dataset. It is expected to contain two folders: ones of lensless images and one of lensed images.
            n_files : int or None, optional
                Metrics will be computed only on the first ``n_files`` images. If None, all images are used, by default False
            background : :py:class:`~torch.Tensor` or None, optional
                If not ``None``, background is removed from lensless images, by default ``None``.
            downsample : int, optional
                Downsample factor of the lensless images, by default 4.
            flip : bool, optional
                If ``True``, lensless images are flipped, by default ``False``.
            transform_lensless : PyTorch Transform or None, optional
                Transform to apply to the lensless images, by default None
            transform_lensed : PyTorch Transform or None, optional
                Transform to apply to the lensed images, by default None
            lensless_fn : str, optional
                Name of the folder containing the lensless images, by default "diffuser".
            lensed_fn : str, optional
                Name of the folder containing the lensed images, by default "lensed".
            image_ext : str, optional
                Extension of the images, by default "npy".
        """

        self.root_dir = root_dir
        self.lensless_dir = os.path.join(root_dir, lensless_fn)
        self.lensed_dir = os.path.join(root_dir, lensed_fn)
        self.image_ext = image_ext.lower()

        files = glob.glob(os.path.join(self.lensless_dir, "*." + image_ext))
        if n_files:
            files = files[:n_files]
        self.files = [os.path.basename(fn) for fn in files]

        if len(self.files) == 0:
            raise FileNotFoundError(
                f"No files found in {self.lensless_dir} with extension {image_ext}"
            )

        self.background = background
        self.downsample = downsample / 4
        self.flip = flip
        self.transform_lensless = transform_lensless
        self.transform_lensed = transform_lensed

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

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

        if self.downsample != 1.0:
            lensless = resize(lensless, factor=1 / self.downsample)
            lensed = resize(lensed, factor=1 / self.downsample)

        lensless = torch.from_numpy(lensless)
        lensed = torch.from_numpy(lensed)

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


class DiffuserCamTestDataset(ParallelDataset):
    """
    Dataset consisting of lensless and corresponding lensed image. This is the standard dataset used for benchmarking.
    """

    def __init__(
        self,
        data_dir="data",
        n_files=200,
        downsample=8,
    ):
        """
        Dataset consisting of lensless and corresponding lensed image. Default parameters are for the test set of DiffuserCam
        Lensless Mirflickr Dataset (DLMD).
        Parameters
        ----------
        data_dir : str, optional
            The path to the folder containing the DiffuserCam_Test dataset, by default "data"
        n_files : int, optional
            Number of image pair to load in the dataset , by default 200
        downsample : int, optional
            Downsample factor of the lensless images, by default 8
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
            data_dir,
            n_files,
            background,
            downsample,
            flip=False,
            transform_lensless=transform_BRG2RGB,
            transform_lensed=transform_BRG2RGB,
            lensless_fn="diffuser",
            lensed_fn="lensed",
            image_ext="npy",
        )
