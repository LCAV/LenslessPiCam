import os.path
import rawpy
import cv2
import numpy as np
import warnings

def load_3d_psf(
    fp,
    downsample=1,
    return_float=True,
    bg_pix=(5, 25),
    return_bg=False,
    dtype=np.float32,
    single_psf=False,
    shape=None,
):

    # load image data
    assert os.path.isfile(fp)
    psf = np.load(fp)
    original_dtype = psf.dtype
    psf = np.array(psf, dtype=dtype)

    # check that all depths of the psf have the same shape.
    for i in range(psf.shape[0]):
        assert psf[0].shape == psf[i].shape

    # subtract background, assume black edges
    if bg_pix is None:
        bg = np.zeros(len(np.shape(psf[0])))

    else:
        # grayscale
        if len(np.shape(psf)) < 4:
            bg = np.mean(psf[:, bg_pix[0]: bg_pix[1], bg_pix[0]: bg_pix[1]])
            psf -= bg

        # rgb
        else:
            bg = []
            for i in range(3):
                bg_i = np.mean(psf[:, bg_pix[0]: bg_pix[1], bg_pix[0]: bg_pix[1], i])
                psf[:, :, :, i] -= bg_i
                bg.append(bg_i)

        psf = np.clip(psf, a_min=0, a_max=psf.max())
        bg = np.array(bg)

    # resize
    if shape:
        for i in range(psf.shape[0]):
            psf[i] = resize(psf[i], shape=shape)
    elif downsample != 1:
        for i in range(psf.shape[0]):
            psf[i] = resize(psf[i], factor=1 / downsample)

    if single_psf:
        if(len(psf.shape) == 4):
            # TODO : in Lensless Learning, they sum channels --> `psf_diffuser = np.sum(psf_diffuser,2)`
            # https://github.com/Waller-Lab/LenslessLearning/blob/master/pre-trained%20reconstructions.ipynb
            psf = np.sum(psf, 2)
            psf = psf[:, :, :, np.newaxis]
        else:
            warnings.warn("Notice : single_psf has no effect for grayscale psf")
            single_psf = False

    # normalize
    if return_float:
        # psf /= psf.max()
        psf /= np.linalg.norm(psf.ravel())
    else:
        psf = psf.astype(original_dtype)

    if return_bg:
        return psf, bg
    else:
        return psf


print(load_3d_psf("psf.npy").shape)