# #############################################################################
# mirflickr.py
# =================
# Authors :
# Eric BEZZAM [ebezzam@gmail.com]
# #############################################################################


import numpy as np
from lensless.recon.admm import ADMM
import cv2


class ADMM_MIRFLICKR(ADMM):
    """
    Object for applying ADMM (Alternating Direction Method of Multipliers) to
    the DiffuserCam Lensless Mirflickr Dataset.

    New class as a unique `_form_image` function has to be defined to crop the
    data at the suitable region.

    Original dataset page: https://waller-lab.github.io/LenslessLearning/dataset.html
    Subset (200 files) we've prepared: https://drive.switch.ch/index.php/s/vmAZzryGI8U8rcE

    """

    def __init__(self, *args, **kwargs):
        # call reset() to initialize matrices
        super(ADMM_MIRFLICKR, self).__init__(*args, **kwargs)

    def _form_image(self):
        image = self._crop(self._image_est)
        image[image < 0] = 0
        return postprocess(image)


def postprocess(image):
    # https://github.com/Waller-Lab/LenslessLearning/blob/eaab0fb694a4f51fdda382a53f92832af98fd692/utils.py#L429
    image = image.astype(np.float32)
    image_color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    out_image = np.flipud(np.clip(image_color, 0, 1))
    return out_image[60:, 62:-38, :]
