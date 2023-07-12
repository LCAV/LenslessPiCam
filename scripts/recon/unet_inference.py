# #############################################################################
# unet_inference.py
# =================
# Authors :
# Jonathan REYMOND [jonathan.reymond7@gmail.com]
# #############################################################################

"""
Load U-net model and run for reconstruction

```
python scripts/recon/unet_inference.py
```

"""

import hydra

import os
import sys
import numpy as np
import time
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/jreymond/LenslessPiCam_ml/')
from lensless.recon.unet_recon import *



@hydra.main(version_base=None, config_path="../../configs", config_name="unet_recon")
def main(config):

    psf = np.zeros(config.input_shape)

    UnetModel(input_shape=config.input_shape,
              output_shape=config.output_shape,
              perceptual_args=config.perceptual_args,
              camera_inversion_args=config.camera_inversion_args,
              model_weights_path=config.model_weights_path,
              psf=psf,
              name=config.name)
    



if __name__ == '__main__':
    main()
    print('done')
    







