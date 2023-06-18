"""

python scripts/remote_capture.py

Check out the `configs/demo.yaml` file for parameters, specifically:

- `rpi`: RPi parameters
- `capture`: parameters for displaying image

"""

import hydra
import os
import subprocess
import cv2
from pprint import pprint
import matplotlib.pyplot as plt
import rawpy


from lensless.util import rgb2gray, print_image_info, check_username_hostname
from lensless.plot import plot_image, pixel_histogram
from lensless.io import load_image, save_image


@hydra.main(version_base=None, config_path="../configs", config_name="demo")
def liveview(config):

    bayer = config.capture.bayer
    rgb = config.capture.rgb
    gray = config.capture.gray

    username, hostname = check_username_hostname(config.rpi.username, config.rpi.hostname)
    legacy = config.capture.legacy
    nbits_out = config.capture.nbits_out
    fn = config.capture.raw_data_fn
    gamma = config.capture.gamma
    source = config.capture.source
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

    # proceed with capture
    # if bayer:
    #     assert not rgb
    #     assert not gray
    assert hostname is not None

    # take picture
    remote_fn = "remote_capture"
    print("\nTaking picture...")
    pic_command = (
        f"{config.rpi.python} {config.capture.script} bayer={bayer} fn={remote_fn} exp={config.capture.exp} iso={config.capture.iso} "
        f"config_pause={config.capture.config_pause} sensor_mode={config.capture.sensor_mode} nbits_out={config.capture.nbits_out}"
    )
    if config.capture.nbits > 8:
        pic_command += " sixteen=True"
    if config.capture.rgb:
        pic_command += " rgb=True"
    if config.capture.legacy:
        pic_command += " legacy=True"
    if config.capture.gray:
        pic_command += " gray=True"
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

    if error != []:
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
        # copy over DNG file
        remotefile = f"~/{remote_fn}.dng"
        localfile = f"{fn}.dng"
        print(f"\nCopying over picture as {localfile}...")
        os.system('scp "%s@%s:%s" %s' % (username, hostname, remotefile, localfile))
        raw = rawpy.imread(localfile)

        # https://letmaik.github.io/rawpy/api/rawpy.Params.html#rawpy.Params
        # https://www.libraw.org/docs/API-datastruct-eng.html
        if nbits_out > 8:
            # only 8 or 16 bit supported by postprocess
            if nbits_out != 16:
                print("casting to 16 bit...")
            output_bps = 16
        else:
            if nbits_out != 8:
                print("casting to 8 bit...")
            output_bps = 8
        img = raw.postprocess(
            adjust_maximum_thr=0,  # default 0.75
            no_auto_scale=False,
            gamma=(1, 1),
            output_bps=output_bps,
            bright=1,  # default 1
            exp_shift=1,
            no_auto_bright=True,
            use_camera_wb=True,
            use_auto_wb=False,  # default is False? f both use_camera_wb and use_auto_wb are True, then use_auto_wb has priority.
        )

        # print image properties
        print_image_info(img)

        # save as PNG
        png_out = f"{fn}.png"
        print(f"Saving RGB file as: {png_out}")
        cv2.imwrite(png_out, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if not bayer:
            os.remove(localfile)

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

            red_gain = config.camera.red_gain
            blue_gain = config.camera.blue_gain
            # get white balance gains
            if red_gain is None:
                red_gain = float(result_dict["Red gain"])
            if blue_gain is None:
                blue_gain = float(result_dict["Blue gain"])

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
    save_image(img, fp)

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
