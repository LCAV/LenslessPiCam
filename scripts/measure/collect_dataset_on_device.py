"""
To be run on the Raspberry Pi!
```
python scripts/measure/collect_dataset_on_device.py
```

Note that the script is configured for the  Raspberry Pi HQ camera

Parameters set in: configs/collect_dataset.yaml

To test on local machine, set dummy=True (which will just copy the files over).

"""

import numpy as np
import hydra
import time
import os
import pathlib as plib
import shutil
import tqdm
from picamerax import PiCamera
from fractions import Fraction
from lensless.utils.io import save_image
from lensless.hardware.sensor import SensorOptions, sensor_dict, SensorParam
import picamerax.array
from lensless.utils.image import bayer2rgb_cc, resize
import cv2


@hydra.main(version_base=None, config_path="../../configs", config_name="collect_dataset")
def collect_dataset(config):

    input_dir = config.input_dir
    output_dir = config.output_dir
    if output_dir is None:
        # create in same directory as input with timestamp
        output_dir = input_dir + "_measured_" + time.strftime("%Y%m%d-%H%M%S")

    MAX_TRIES = config.max_tries
    MIN_LEVEL = config.min_level
    MAX_LEVEL = config.max_level

    # get sensor parameters
    sensor = config.capture.sensor
    nbits_capture = config.capture.nbits_capture
    assert sensor in SensorOptions.values(), f"Sensor must be one of {SensorOptions.values()}"
    if SensorParam.BLACK_LEVEL in sensor_dict[sensor]:
        black_level = sensor_dict[sensor][SensorParam.BLACK_LEVEL] * (2**nbits_capture - 1)
    else:
        black_level = 0
    if SensorParam.CCM_MATRIX in sensor_dict[sensor]:
        ccm = sensor_dict[sensor][SensorParam.CCM_MATRIX]
    else:
        ccm = None
    assert sensor == "rpi_hq", "Only RPi HQ camera supported for now!"

    # if output dir exists check how many files done
    print(f"Output directory : {output_dir}")
    start_idx = 0
    if os.path.exists(output_dir):
        files = list(plib.Path(output_dir).glob(f"*.{config.output_file_ext}"))
        start_idx = len(files)
        print("\nNumber of completed measurements :", start_idx)

    # make output directory if need be
    output_dir = plib.Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # assert input directory exists
    assert os.path.exists(input_dir)

    # get number of files with glob
    files = list(plib.Path(input_dir).glob(f"*.{config.input_file_ext}"))
    n_files = len(files)
    print(f"\nNumber of {config.input_file_ext} files :", n_files)
    if config.n_files:
        print(f"TEST : collecting first {config.n_files} files!")
        files = files[: config.n_files]
        n_files = len(files)

    if config.runtime:
        # convert to minutes
        runtime_min = config.runtime * 60
        runtime_sec = runtime_min * 60
        if config.runtime:
            print(f"\nScript will run for (at most) {config.runtime} hour(s).")

    if config.start_delay:
        # wait for this time before starting script
        delay = config.start_delay * 60
        start_time = time.time() + delay
        start_time = time.strftime("%H:%M:%S", time.localtime(start_time))
        print(f"\nScript will start at {start_time}")
        time.sleep(delay)

    print("\nStarting measurement!\n")
    start_time = time.time()

    if not config.dummy:

        res = config.capture.res
        down = config.capture.down

        # set up camera for consistent photos
        # https://picamera.readthedocs.io/en/release-1.13/recipes1.html#capturing-consistent-images
        # https://picamerax.readthedocs.io/en/latest/fov.html?highlight=camera%20resolution#sensor-modes
        # -- just get max resolution of camera
        camera = PiCamera(framerate=30)
        if res:
            assert len(res) == 2
        else:
            res = np.array(camera.MAX_RESOLUTION)
            if down is not None:
                res = (np.array(res) / down).astype(int)
        camera.close()

        # -- now set up camera with desired settings
        camera = PiCamera(
            framerate=1 / config.capture.exposure, sensor_mode=0, resolution=tuple(res)
        )

        # Set ISO to the desired value
        camera.resolution = tuple(res)
        camera.iso = config.capture.iso
        # Wait for the automatic gain control to settle
        time.sleep(config.capture.config_pause)
        # Now fix the values

        if config.capture.exposure:
            # in microseconds
            init_shutter_speed = int(config.capture.exposure * 1e6)
        else:
            init_shutter_speed = camera.exposure_speed
        camera.shutter_speed = init_shutter_speed
        camera.exposure_mode = "off"

        # AWB
        if config.capture.awb_gains:
            assert len(config.capture.awb_gains) == 2
            g = (Fraction(config.capture.awb_gains[0]), Fraction(config.capture.awb_gains[1]))
            g = tuple(g)
        else:
            g = camera.awb_gains

        camera.awb_mode = "off"
        camera.awb_gains = g

        # for parameters to settle
        time.sleep(5)

        print("Capturing at resolution: ", res)
        print("AWB gains", float(camera.awb_gains[0]), float(camera.awb_gains[1]))

    init_brightness = config.display.brightness

    # loop over files with tqdm
    exposure_vals = []
    brightness_vals = []
    n_tries_vals = []
    for i, _file in enumerate(tqdm.tqdm(files), start=start_idx):

        # save file in output directory as PNG
        output_fp = output_dir / _file.name
        output_fp = output_fp.with_suffix(f".{config.output_file_ext}")

        # if not done, perform measurement
        if not os.path.isfile(output_fp):

            if config.dummy:
                shutil.copyfile(_file, output_fp)
                time.sleep(1)

            else:

                # -- show on display
                screen_res = np.array(config.display.screen_res)
                hshift = config.display.hshift
                vshift = config.display.vshift
                pad = config.display.pad
                brightness = init_brightness
                display_image_path = config.display.output_fp
                rot90 = config.display.rot90
                os.system(
                    f"python scripts/measure/prep_display_image.py --fp {_file} --output_path {display_image_path} --screen_res {screen_res[0]} {screen_res[1]} --hshift {hshift} --vshift {vshift} --pad {pad} --brightness {brightness} --rot90 {rot90}"
                )

                time.sleep(config.capture.delay)

                # -- take picture
                max_pixel_val = 0
                fact = 2
                n_tries = 0

                camera.shutter_speed = init_shutter_speed
                time.sleep(5)

                current_screen_brightness = init_brightness
                current_shutter_speed = camera.shutter_speed
                print(f"current shutter speed: {current_shutter_speed}")
                print(f"current screen brightness: {current_screen_brightness}")

                while max_pixel_val < MIN_LEVEL or max_pixel_val > MAX_LEVEL:

                    if n_tries > MAX_TRIES:
                        print("Max number of tries reached!")
                        break

                    # get bayer data
                    stream = picamerax.array.PiBayerArray(camera)
                    camera.capture(stream, "jpeg", bayer=True)
                    output_bayer = np.sum(stream.array, axis=2).astype(np.uint16)

                    # convert to RGB
                    output = bayer2rgb_cc(
                        output_bayer,
                        nbits=nbits_capture,
                        blue_gain=float(g[1]),
                        red_gain=float(g[0]),
                        black_level=black_level,
                        ccm=ccm,
                        nbits_out=config.capture.nbits_out,
                    )

                    if down:
                        output = resize(
                            output[None, ...], factor=1 / down, interpolation=cv2.INTER_CUBIC
                        )[0]

                    # print range
                    print(f"{output_fp}, range: {output.min()} - {output.max()}")
                    max_pixel_val = output.max()

                    if max_pixel_val < MIN_LEVEL:
                        current_shutter_speed = init_shutter_speed * fact
                        camera.shutter_speed = current_shutter_speed
                        time.sleep(5)
                        print(f"increasing shutter speed to {current_shutter_speed}")
                        fact += 1

                    elif max_pixel_val > MAX_LEVEL:

                        current_screen_brightness = current_screen_brightness - 10

                        screen_res = np.array(config.display.screen_res)
                        hshift = config.display.hshift
                        vshift = config.display.vshift
                        pad = config.display.pad
                        brightness = current_screen_brightness
                        display_image_path = config.display.output_fp
                        rot90 = config.display.rot90
                        os.system(
                            f"python scripts/measure/prep_display_image.py --fp {_file} --output_path {display_image_path} --screen_res {screen_res[0]} {screen_res[1]} --hshift {hshift} --vshift {vshift} --pad {pad} --brightness {brightness} --rot90 {rot90}"
                        )
                        print(f"decreasing screen brightness to {current_screen_brightness}")

                        time.sleep(config.capture.delay)

                    n_tries += 1

                    # save image
                    save_image(output, output_fp)

                exposure_vals.append(current_shutter_speed / 1e6)
                brightness_vals.append(current_screen_brightness)
                n_tries_vals.append(n_tries)

        # check if runtime is exceeded
        if config.runtime:
            proc_time = time.time() - start_time
            if proc_time > runtime_sec:
                print(f"-- measured {i+1} / {n_files} files")
                break

    proc_time = time.time() - start_time
    print(f"\nFinished, {proc_time/60.:.3f} minutes.")

    # print brightness and exposure range and average
    print(f"brightness range: {np.min(brightness_vals)} - {np.max(brightness_vals)}")
    print(f"exposure range: {np.min(exposure_vals)} - {np.max(exposure_vals)}")
    print(f"n_tries range: {np.min(n_tries_vals)} - {np.max(n_tries_vals)}")
    print(f"brightness average: {np.mean(brightness_vals)}")
    print(f"exposure average: {np.mean(exposure_vals)}")
    print(f"n_tries average: {np.mean(n_tries_vals)}")


if __name__ == "__main__":
    collect_dataset()
