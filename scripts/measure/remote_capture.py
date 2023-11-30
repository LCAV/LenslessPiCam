"""

For Bayer data with RPI HQ sensor:
```
python scripts/measure/remote_capture.py \
rpi.username=USERNAME rpi.hostname=IP_ADDRESS
```

For Bayer data with RPI Global shutter sensor:
```
python scripts/measure/remote_capture.py -cn remote_capture_rpi_gs \
rpi.username=USERNAME rpi.hostname=IP_ADDRESS
```

For RGB data with RPI HQ RPI Global shutter sensor:
```
python scripts/measure/remote_capture.py -cn remote_capture_rpi_gs \
rpi.username=USERNAME rpi.hostname=IP_ADDRESS \
capture.bayer=False capture.down=2
```

Check out the `configs/demo.yaml` file for parameters, specifically:

- `rpi`: RPi parameters
- `capture`: parameters for taking pictures

"""

import hydra
import os
import matplotlib.pyplot as plt
from lensless.hardware.utils import check_username_hostname, capture
from lensless.utils.image import rgb2gray
from lensless.utils.plot import plot_image, pixel_histogram
from lensless.utils.io import save_image


@hydra.main(version_base=None, config_path="../../configs", config_name="demo")
def liveview(config):

    username = config.rpi.username
    hostname = config.rpi.hostname
    check_username_hostname(username, hostname)

    # black_level, ccm, _ = check_capture_config(config.capture)
    gray = config.capture.gray
    fn = config.capture.fn
    gamma = config.capture.gamma
    plot = config.plot

    if config.save:
        if config.output is not None:
            # make sure output directory exists
            os.makedirs(config.output, exist_ok=True)
            save = config.output
        else:
            save = os.getcwd()
    else:
        save = False

    # take picture
    _, img = capture(
        rpi_username=username,
        rpi_hostname=hostname,
        verbose=True,
        output_dir=save,
        **config.capture,
    )

    # save image as viewable 8 bit
    fp = os.path.join(save, f"{fn}_rgb_8bit.png")
    save_image(img, fp)

    # plot RGB
    if plot:
        if not gray:
            ax = plot_image(img, gamma=gamma)
            ax.set_title("RGB")
            if save:
                plt.savefig(os.path.join(save, f"{fn}_plot.png"))

            # plot grayscale
            img_1chan = rgb2gray(img[None, :, :, :])
            ax = plot_image(img_1chan)
            ax.set_title("Grayscale")
            if save:
                plt.savefig(os.path.join(save, f"{fn}_gray.png"))

            # plot histogram, useful for checking clipping
            pixel_histogram(img)
            if save:
                plt.savefig(os.path.join(save, f"{fn}_hist.png"))
            pixel_histogram(img_1chan)
            if save:
                plt.savefig(os.path.join(save, f"{fn}_gray_hist.png"))

        else:
            ax = plot_image(img, gamma=gamma)
            if save:
                plt.savefig(os.path.join(save, f"{fn}_plot.png"))
            pixel_histogram(img)
            if save:
                plt.savefig(os.path.join(save, f"{fn}_hist.png"))

        plt.show()

    if save:
        print(f"\nSaved images to: {save}")


if __name__ == "__main__":
    liveview()
