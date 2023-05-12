from lensless.io import load_data
import numpy as np

psf_fp = "data/psf/tape_rgb.png"
data_fp = "data/raw_data/thumbs_up_rgb.png"
downsample = 8


def test_load_data():
    for gray in [True, False]:
        for dtype in ["float32", "float64"]:
            psf, data = load_data(
                psf_fp=psf_fp,
                data_fp=data_fp,
                downsample=downsample,
                plot=False,
                gray=gray,
                dtype=dtype,
            )
            assert psf.shape[3] == (1 if gray else 3)
            assert len(psf.shape) == 4
            assert len(data.shape) == 4
            assert data.shape[0] == 1
            assert psf.shape[1:] == data.shape[1:]
            assert psf.dtype == dtype, dtype
            assert data.dtype == dtype, dtype


test_load_data()
