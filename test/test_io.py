from lensless.utils.io import load_data, rgb2gray

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


def test_rgb2gray():
    for is_torch in [True, False]:
        psf, data = load_data(
            psf_fp=psf_fp,
            data_fp=data_fp,
            downsample=downsample,
            plot=False,
            dtype="float32",
            use_torch=is_torch,
        )
        data = data[0]  # drop first depth dimension

        # try with 4D
        psf_gray = rgb2gray(psf, keepchanneldim=False)
        assert len(psf_gray.shape) == 3
        psf_gray = rgb2gray(psf, keepchanneldim=True)
        assert len(psf_gray.shape) == 4

        # try with 3D
        data_gray = rgb2gray(data, keepchanneldim=False)
        assert len(data_gray.shape) == 2
        data_gray = rgb2gray(data, keepchanneldim=True)
        assert len(data_gray.shape) == 3


if __name__ == "__main__":
    test_load_data()
    test_rgb2gray()
