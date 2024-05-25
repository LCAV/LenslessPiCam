import numpy as np
from lensless.hardware.utils import set_mask_sensor_distance
import hydra
import os
from datetime import datetime
from PIL import Image

SATURATION_THRESHOLD = 0.01


@hydra.main(version_base=None, config_path="../../configs", config_name="digicam_config")
def config_digicam(config):

    rpi_username = config.rpi.username
    rpi_hostname = config.rpi.hostname

    mask_sensor_distances = np.arange(9) * 0.1
    exposure_time = 5

    timestamp = datetime.now().strftime("%Y%m%d")

    for i in range(len(mask_sensor_distances)):

        print(f"Mask sensor distance: {mask_sensor_distances[i]}mm")
        mask_sensor_distance = mask_sensor_distances[i]

        # set the mask sensor distance
        set_mask_sensor_distance(mask_sensor_distance, rpi_username, rpi_hostname)

        good_exposure = False
        while not good_exposure:

            # measure PSF
            output_folder = f"adafruit_psf_{mask_sensor_distance}mm__{timestamp}"
            os.system(
                f"python scripts/remote_capture.py -cn capture_bayer output={output_folder} rpi.username={rpi_username} rpi.hostname={rpi_hostname} capture.exp={exposure_time}"
            )

            # check for saturation
            OUTPUT_FP = os.path.join(output_folder, "raw_data.png")
            # -- load picture to check for saturation
            img = np.array(Image.open(OUTPUT_FP))
            ratio = np.sum(img == 4095) / np.prod(img.shape)
            print(f"Saturation ratio: {ratio}")
            if ratio > SATURATION_THRESHOLD or ratio == 0:

                if ratio == 0:
                    print("Need to increase exposure time.")
                else:
                    print("Need to decrease exposure time.")

                # enter new exposure time from keyboard
                exposure_time = float(input("Enter new exposure time: "))

            else:
                good_exposure = True


if __name__ == "__main__":
    config_digicam()
