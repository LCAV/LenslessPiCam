import numpy as np
from lensless.io import load_data

try:
    import pycsou
    from lensless import GradientDescent, NesterovGradientDescent, FISTA, ADMM, APGD

    pycsou_available = True

except ImportError:
    from lensless import GradientDescent, NesterovGradientDescent, FISTA, ADMM

    pycsou_available = False


try:
    import torch

    torch_vals = [True, False]
except ImportError:
    torch_vals = [False]


psf_fp = "data/psf/tape_rgb.png"
data_fp = "data/raw_data/thumbs_up_rgb.png"
dtypes = ["float32", "float64"]
downsample = 16
n_iter = 5
disp = None


def test_gradient_descent():
    for gray in [True, False]:
        for dtype in dtypes:
            for torch_data in torch_vals:
                psf, data = load_data(
                    psf_fp=psf_fp,
                    data_fp=data_fp,
                    downsample=downsample,
                    plot=False,
                    gray=gray,
                    dtype=dtype,
                    torch=torch_data,
                )
                recon = GradientDescent(psf, dtype=dtype)
                recon.set_data(data)
                res = recon.apply(n_iter=n_iter, disp_iter=None, plot=False)
                if gray:
                    assert len(psf.shape) == 2
                else:
                    assert len(psf.shape) == 3
                assert res.dtype == psf.dtype, f"Got {res.dtype}, expected {dtype}"


def test_admm():
    for gray in [True, False]:
        for dtype in dtypes:
            for torch_data in torch_vals:
                psf, data = load_data(
                    psf_fp=psf_fp,
                    data_fp=data_fp,
                    downsample=downsample,
                    plot=False,
                    gray=gray,
                    dtype=dtype,
                    torch=torch_data,
                )
                recon = ADMM(psf, dtype=dtype)
                recon.set_data(data)
                res = recon.apply(n_iter=n_iter, disp_iter=None, plot=False)
                if gray:
                    assert len(psf.shape) == 2
                else:
                    assert len(psf.shape) == 3
                assert res.dtype == psf.dtype, f"Got {res.dtype}, expected {dtype}"


def test_nesterov_gradient_descent():
    for gray in [True, False]:
        for dtype in dtypes:
            for torch_data in torch_vals:
                psf, data = load_data(
                    psf_fp=psf_fp,
                    data_fp=data_fp,
                    downsample=downsample,
                    plot=False,
                    gray=gray,
                    dtype=dtype,
                    torch=torch_data,
                )
                recon = NesterovGradientDescent(psf, dtype=dtype)
                recon.set_data(data)
                res = recon.apply(n_iter=n_iter, disp_iter=None, plot=False)
                if gray:
                    assert len(psf.shape) == 2
                else:
                    assert len(psf.shape) == 3
                assert res.dtype == psf.dtype, f"Got {res.dtype}, expected {dtype}"


def test_fista():
    for gray in [True, False]:
        for dtype in dtypes:
            for torch_data in torch_vals:
                psf, data = load_data(
                    psf_fp=psf_fp,
                    data_fp=data_fp,
                    downsample=downsample,
                    plot=False,
                    gray=gray,
                    dtype=dtype,
                    torch=torch_data,
                )
                recon = FISTA(psf, dtype=dtype)
                recon.set_data(data)
                res = recon.apply(n_iter=n_iter, disp_iter=None, plot=False)
                if gray:
                    assert len(psf.shape) == 2
                else:
                    assert len(psf.shape) == 3
                assert res.dtype == psf.dtype, f"Got {res.dtype}, expected {dtype}"


def test_apgd():
    if pycsou_available:
        for gray in [True, False]:
            for dtype in dtypes:
                psf, data = load_data(
                    psf_fp=psf_fp,
                    data_fp=data_fp,
                    downsample=downsample,
                    plot=False,
                    gray=gray,
                    dtype=dtype,
                    torch=False,
                )
                recon = APGD(psf, dtype=dtype)
                recon.set_data(data)
                res = recon.apply(n_iter=n_iter, disp_iter=None, plot=False)
                if gray:
                    assert len(psf.shape) == 2
                else:
                    assert len(psf.shape) == 3
                assert res.dtype == psf.dtype, f"Got {res.dtype}, expected {dtype}"
    else:
        print("Pycsou not installed. Skipping APGD test.")


if __name__ == "__main__":
    test_gradient_descent()
    test_admm()
    test_nesterov_gradient_descent()
    test_fista()
    test_apgd()
