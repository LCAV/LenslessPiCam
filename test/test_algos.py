import numpy as np
from lensless.io import load_data

try:
    import pycsou
    from lensless import GradientDescent, NesterovGradientDescent, FISTA, ADMM, APGD

    pycsou_available = True
    algos = [GradientDescent, NesterovGradientDescent, FISTA, ADMM, APGD]

except ImportError:
    from lensless import GradientDescent, NesterovGradientDescent, FISTA, ADMM

    pycsou_available = False
    algos = [GradientDescent, NesterovGradientDescent, FISTA, ADMM]


try:
    import torch

    torch_vals = [True, False]
except ImportError:
    torch_vals = [False]


psf_fp = "data/psf/tape_rgb.png"
data_fp = "data/raw_data/thumbs_up_rgb.png"
downsample = 16
n_iter = 5
disp = None


def test_algo():
    for algo in algos:
        for gray in [True, False]:
            for dtype in ["float32", "float64"]:
                for torch_data in torch_vals:
                    if pycsou_available and algo == APGD and torch_data:
                        continue
                    psf, data = load_data(
                        psf_fp=psf_fp,
                        data_fp=data_fp,
                        downsample=downsample,
                        plot=False,
                        gray=gray,
                        dtype=dtype,
                        torch=torch_data,
                    )
                    recon = algo(psf, dtype=dtype)
                    recon.set_data(data)
                    res = recon.apply(n_iter=n_iter, disp_iter=None, plot=False)
                    if gray:
                        assert len(psf.shape) == 2
                    else:
                        assert len(psf.shape) == 3
                    assert res.dtype == psf.dtype, f"Got {res.dtype}, expected {dtype}"


test_algo()
