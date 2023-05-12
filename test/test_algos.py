from lensless.io import load_data

# classical algorithm
try:
    import pycsou
    from lensless import GradientDescent, NesterovGradientDescent, FISTA, ADMM, APGD, APGDPriors

    pycsou_available = True

except ImportError:
    from lensless import GradientDescent, NesterovGradientDescent, FISTA, ADMM

    pycsou_available = False


try:
    import torch

    torch_is_available = True
    torch_vals = [True, False]
    torch.autograd.set_detect_anomaly(True)
except ImportError:
    torch_vals = [False]
    torch_is_available = False


psf_fp = "data/psf/tape_rgb.png"
data_fp = "data/raw_data/thumbs_up_rgb.png"
dtypes = ["float32", "float64"]
downsample = 16
n_iter = 5

# classical algorithms


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
                assert psf.shape[3] == (1 if gray else 3)
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
                assert psf.shape[3] == (1 if gray else 3)
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
                assert psf.shape[3] == (1 if gray else 3)
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
                assert psf.shape[3] == (1 if gray else 3)
                assert res.dtype == psf.dtype, f"Got {res.dtype}, expected {dtype}"


def test_apgd():
    if pycsou_available:
        for gray in [True, False]:
            for dtype in dtypes:
                for diff_penalty in [APGDPriors.L2, None]:
                    for prox_penalty in [APGDPriors.NONNEG, APGDPriors.L1]:
                        psf, data = load_data(
                            psf_fp=psf_fp,
                            data_fp=data_fp,
                            downsample=downsample,
                            plot=False,
                            gray=gray,
                            dtype=dtype,
                            torch=False,
                        )
                        recon = APGD(
                            psf,
                            dtype=dtype,
                            prox_penalty=prox_penalty,
                            diff_penalty=diff_penalty,
                            rel_error=None,
                        )
                        recon.set_data(data)
                        res = recon.apply(n_iter=n_iter, disp_iter=None, plot=False)
                        assert psf.shape[3] == (1 if gray else 3)
                        assert res.dtype == psf.dtype, f"Got {res.dtype}, expected {dtype}"
    else:
        print("Pycsou not installed. Skipping APGD test.")


#  trainable algorithms


def test_unrolled_fista():
    if torch_is_available:
        from lensless.unrolled_fista import unrolled_FISTA

        for dtype, torch_type in [("float32", torch.float32), ("float64", torch.float64)]:
            psf = torch.rand(32, 64, 3, dtype=torch_type)
            data = torch.rand(2, 32, 64, 3, dtype=torch_type)
            recon = unrolled_FISTA(psf, n_iter=n_iter, dtype=dtype)

            assert (
                next(recon.parameters(), None) is not None
            ), "unrolled FISTA has no trainable parameters"

            res = recon.batch_call(data)
            loss = torch.mean(res)
            loss.backward()

            assert (
                data.shape[0] == res.shape[0]
            ), f"Batch dimension changed: got {res.shape[0]} expected {data.shape[0]}"

            assert len(psf.shape) == 3
            assert res.shape[3] == 3, "Input in HWC format but output CHW format"

            # check support for CHW
            data = torch.rand(1, 3, 32, 64, dtype=torch_type)
            res = recon.batch_call(data)
            assert (
                data.shape[0] == res.shape[0]
            ), f"Batch dimension changed: got {res.shape[0]} expected {data.shape[0]}"
            assert res.shape[1] == 3, "Input in CHW format but output HWC format"
            assert res.dtype == psf.dtype, f"Got {res.dtype}, expected {dtype}"


def test_unrolled_admm():
    if torch_is_available:
        from lensless.unrolled_admm import unrolled_ADMM

        for dtype, torch_type in [("float32", torch.float32), ("float64", torch.float64)]:
            psf = torch.rand(32, 64, 3, dtype=torch_type)
            data = torch.rand(2, 32, 64, 3, dtype=torch_type)
            recon = unrolled_ADMM(psf, n_iter=n_iter, dtype=dtype)

            assert (
                next(recon.parameters(), None) is not None
            ), "unrolled ASMM has no trainable parameters"

            res = recon.batch_call(data)
            loss = torch.mean(res)
            loss.backward()

            assert (
                data.shape[0] == res.shape[0]
            ), f"Batch dimension changed: got {res.shape[0]} expected {data.shape[0]}"

            assert len(psf.shape) == 3
            assert res.shape[3] == 3, "Input in HWC format but output CHW format"

            # check support for CHW
            data = torch.rand(1, 3, 32, 64, dtype=torch_type)
            res = recon.batch_call(data)
            assert (
                data.shape[0] == res.shape[0]
            ), f"Batch dimension changed: got {res.shape[0]} expected {data.shape[0]}"
            assert res.shape[1] == 3, "Input in CHW format but output HWC format"
            assert res.dtype == psf.dtype, f"Got {res.dtype}, expected {dtype}"


if __name__ == "__main__":
    test_gradient_descent()
    test_admm()
    test_nesterov_gradient_descent()
    test_fista()
    test_apgd()
    test_unrolled_fista()
    test_unrolled_admm()
