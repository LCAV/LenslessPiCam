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
import subprocess
import cv2
from pprint import pprint
import matplotlib.pyplot as plt
from lensless.hardware.utils import check_username_hostname
from lensless.hardware.sensor import SensorOptions, sensor_dict, SensorParam
from lensless.utils.image import rgb2gray, print_image_info
from lensless.utils.plot import plot_image, pixel_histogram
from lensless.utils.io import save_image
from lensless.utils.io import load_image


@hydra.main(version_base=None, config_path="../../configs", config_name="demo")
def liveview(config):

    sensor = config.capture.sensor
    assert sensor in SensorOptions.values(), f"Sensor must be one of {SensorOptions.values()}"

    bayer = config.capture.bayer
    rgb = config.capture.rgb
    gray = config.capture.gray

    check_username_hostname(config.rpi.username, config.rpi.hostname)
    username = config.rpi.username
    hostname = config.rpi.hostname
    legacy = config.capture.legacy
    nbits_out = config.capture.nbits_out
    fn = config.capture.raw_data_fn
    gamma = config.capture.gamma
    source = config.capture.source
    plot = config.plot

    assert (
        nbits_out in sensor_dict[sensor][SensorParam.BIT_DEPTH]
    ), f"capture.nbits_out must be one of {sensor_dict[sensor][SensorParam.BIT_DEPTH]} for sensor {sensor}"
    assert (
        config.capture.nbits in sensor_dict[sensor][SensorParam.BIT_DEPTH]
    ), f"capture.nbits must be one of {sensor_dict[sensor][SensorParam.BIT_DEPTH]} for sensor {sensor}"

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
    remote_fn = "remote_capture"
    print("\nTaking picture...")
    pic_command = (
        f"{config.rpi.python} {config.capture.script} sensor={sensor} bayer={bayer} fn={remote_fn} exp={config.capture.exp} iso={config.capture.iso} "
        f"config_pause={config.capture.config_pause} sensor_mode={config.capture.sensor_mode} nbits_out={config.capture.nbits_out} "
        f"legacy={config.capture.legacy} rgb={config.capture.rgb} gray={config.capture.gray} "
    )
    if config.capture.nbits > 8:
        pic_command += " sixteen=True"
    if config.capture.down:
        pic_command += f" down={config.capture.down}"
    if config.capture.awb_gains:
        pic_command += f" awb_gains=[{config.capture.awb_gains[0]},{config.capture.awb_gains[1]}]"

    print(f"COMMAND : {pic_command}")
    ssh = subprocess.Popen(
        ["ssh", "%s@%s" % (username, hostname), pic_command],
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    result = ssh.stdout.readlines()
    error = ssh.stderr.readlines()

    if error != [] and legacy:  # new camera software seems to return error even if it works
        print("ERROR: %s" % error)
        return
    if result == []:
        error = ssh.stderr.readlines()
        print("ERROR: %s" % error)
        return
    else:
        result = [res.decode("UTF-8") for res in result]
        result = [res for res in result if len(res) > 3]
        result_dict = dict()
        for res in result:
            _key = res.split(":")[0].strip()
            _val = "".join(res.split(":")[1:]).strip()
            result_dict[_key] = _val
        # result_dict = dict(map(lambda s: map(str.strip, s.split(":")), result))
        print("COMMAND OUTPUT : ")
        pprint(result_dict)

    if (
        "RPi distribution" in result_dict.keys()
        and "bullseye" in result_dict["RPi distribution"]
        and not legacy
    ):
        assert not rgb or not gray, "RGB and gray not supported for RPi HQ sensor"

        if bayer:

            assert config.capture.down is None

            # copy over DNG file
            remotefile = f"~/{remote_fn}.dng"
            localfile = os.path.join(save, f"{fn}.dng")
            print(f"\nCopying over picture as {localfile}...")
            os.system('scp "%s@%s:%s" %s' % (username, hostname, remotefile, localfile))

            img = load_image(localfile, verbose=True, bayer=bayer, nbits_out=nbits_out)

            # print image properties
            print_image_info(img)

            # save as PNG
            png_out = f"{fn}.png"
            print(f"Saving RGB file as: {png_out}")
            cv2.imwrite(png_out, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        else:

            remotefile = f"~/{remote_fn}.png"
            localfile = f"{fn}.png"
            if save:
                localfile = os.path.join(save, localfile)
            print(f"\nCopying over picture as {localfile}...")
            os.system('scp "%s@%s:%s" %s' % (username, hostname, remotefile, localfile))

            img = load_image(localfile, verbose=True)

    # legacy software running on RPi
    else:
        # copy over file
        # more pythonic? https://stackoverflow.com/questions/250283/how-to-scp-in-python
        remotefile = f"~/{remote_fn}.png"
        localfile = f"{fn}.png"
        if save:
            localfile = os.path.join(save, localfile)
        print(f"\nCopying over picture as {localfile}...")
        os.system('scp "%s@%s:%s" %s' % (username, hostname, remotefile, localfile))

        if rgb or gray:
            img = load_image(localfile, verbose=True)

        else:

            if not bayer:
                red_gain = config.camera.red_gain
                blue_gain = config.camera.blue_gain
            else:
                red_gain = None
                blue_gain = None
            # # get white balance gains
            # if red_gain is None:
            #     red_gain = float(result_dict["Red gain"])
            # if blue_gain is None:
            #     blue_gain = float(result_dict["Blue gain"])

            # # get white balance gains
            # if bayer:
            #     red_gain = 1
            #     blue_gain = 1
            # else:
            #     red_gain = float(result_dict["Red gain"])
            #     blue_gain = float(result_dict["Blue gain"])

            # load image
            print("\nLoading picture...")

            img = load_image(
                localfile,
                verbose=True,
                bayer=bayer,
                blue_gain=blue_gain,
                red_gain=red_gain,
                nbits_out=nbits_out,
            )

            # write RGB data
            if not bayer:
                cv2.imwrite(localfile, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # save image as viewable 8 bit
    fp = os.path.join(save, f"{fn}_8bit.png")
    save_image(img, fp, normalize=True)

    # plot RGB
    if plot:
        if not gray:
            ax = plot_image(img, gamma=gamma)
            ax.set_title("RGB")
            if save:
                plt.savefig(os.path.join(save, f"{fn}_plot.png"))

            # plot red channel
            if source == "red":
                img_1chan = img[:, :, 0]
            elif source == "green":
                img_1chan = img[:, :, 1]
            elif source == "blue":
                img_1chan = img[:, :, 2]
            else:
                img_1chan = rgb2gray(img[None, :, :, :])
            ax = plot_image(img_1chan)
            if source == "white":
                ax.set_title("Gray scale")
            else:
                ax.set_title(f"{source} channel")
            if save:
                plt.savefig(os.path.join(save, f"{fn}_1chan.png"))

            # plot histogram, useful for checking clipping
            pixel_histogram(img)
            if save:
                plt.savefig(os.path.join(save, f"{fn}_hist.png"))
            pixel_histogram(img_1chan)
            if save:
                plt.savefig(os.path.join(save, f"{fn}_1chan_hist.png"))

        else:
            ax = plot_image(img, gamma=gamma)
            if save:
                plt.savefig(os.path.join(save, f"{fn}_plot.png"))
            pixel_histogram(img)
            if save:
                plt.savefig(os.path.join(save, f"{fn}_hist.png"))

        plt.show()

    if save:
        print(f"\nSaved plots to: {save}")


if __name__ == "__main__":
    liveview()
