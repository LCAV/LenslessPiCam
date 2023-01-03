from waveprop.dataset_util import Propagated
import numpy as np
from lensless.io import load_psf
import torch
from torchvision.transforms import ToTensor


batch_size = 4
ds_path = "data/celeba_mini"
psf_fp = "data/psf/tape_rgb.png"
image_ext = "jpg"
downsample = 8
random_vflip = 0.5
random_hflip = 0.5
random_rotate = 90
random_shift = True
object_height = [0.4, 0.4]


psf = load_psf(psf_fp, downsample=downsample)
psf = ToTensor()(psf)

ds = Propagated(
    path=ds_path,
    image_ext=image_ext,
    object_height=object_height,
    scene2mask=40e-2,
    mask2sensor=4e-3,
    sensor="rpi_hq",
    psf=psf,
    snr_db=40,
    max_val=255,
    target="original",  # "original" or "object_plane"
    random_vflip=random_vflip,
    random_hflip=random_hflip,
    random_rotate=random_rotate,
    random_shift=random_shift,
)

ds_loader = torch.utils.data.DataLoader(dataset=ds, batch_size=batch_size, shuffle=True)

for i, (x, target) in enumerate(ds_loader):

    if i == 0:
        print("Batch shape : ", x.shape)
        print("Target shape : ", target.shape)

print(f"Went through {len(ds_loader)} batches.")
