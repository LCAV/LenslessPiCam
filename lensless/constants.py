import numpy as np

# estimated here: https://www.strollswithmydog.com/open-raspberry-pi-high-quality-camera-raw
RPI_HQ_CAMERA_CCM_MATRIX = np.array(
    [
        [2.0659, -0.93119, -0.13421],
        [-0.11615, 1.5593, -0.44314],
        [0.073694, -0.4368, 1.3636],
    ]
)
RPI_HQ_CAMERA_BLACK_LEVEL = 256.3
