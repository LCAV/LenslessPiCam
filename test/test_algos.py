import numpy as np
from lensless.io import load_data
from lensless import GradientDescient, NesterovGradientDescent, FISTA, ADMM, APGD


psf_fp = "data/psf/tape_rgb.png"
data_fp = "data/raw_data/thumbs_up_rgb.png"
downsample = 16
n_iter = 5
disp = None


def test_algo():
    for algo in [GradientDescient, NesterovGradientDescent, FISTA, ADMM, APGD]:
        for gray in [True, False]:
            for dtype in [np.float32, np.float64]:
                psf, data = load_data(
                    psf_fp=psf_fp,
                    data_fp=data_fp,
                    downsample=downsample,
                    plot=False,
                    gray=gray,
                    dtype=dtype,
                )
                recon = algo(psf, dtype=dtype)
                recon.set_data(data)
                res = recon.apply(n_iter=n_iter, disp_iter=None, plot=False)
                if gray:
                    assert len(psf.shape) == 2
                else:
                    assert len(psf.shape) == 3
                assert res.dtype == dtype, f"Got {res.dtype}, expected {dtype}"


test_algo()
