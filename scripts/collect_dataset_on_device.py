"""
To be run on the Raspberry Pi!
```
python scripts/collect_dataset_on_device.py
```

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
from fractions import Fraction


@hydra.main(version_base=None, config_path="../configs", config_name="collect_dataset")
def collect_dataset(config):

    input_dir = config.input_dir
    output_dir = config.output_dir
    if output_dir is None:
        # create in same directory as input with timestamp
        output_dir = input_dir + "_measured_" + time.strftime("%Y%m%d-%H%M%S")

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

        from picamerax import PiCamera

        # set up camera for consistent photos
        # https://picamera.readthedocs.io/en/release-1.13/recipes1.html#capturing-consistent-images
        # https://picamerax.readthedocs.io/en/latest/fov.html?highlight=camera%20resolution#sensor-modes
        camera = PiCamera(framerate=30)
        if res:
            assert len(res) == 2
        else:
            res = np.array(camera.MAX_RESOLUTION)
            if down is not None:
                res = (np.array(res) / down).astype(int)

        # Set ISO to the desired value
        camera.resolution = tuple(res)
        camera.iso = config.capture.iso
        # Wait for the automatic gain control to settle
        time.sleep(config.capture.config_pause)
        # Now fix the values
        if config.capture.exposure:
            # in microseconds
            camera.shutter_speed = int(config.capture.exposure * 1e6)
        else:
            camera.shutter_speed = camera.exposure_speed
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
        time.sleep(0.1)

        print("Capturing at resolution: ", res)
        print("AWB gains", float(camera.awb_gains[0]), float(camera.awb_gains[1]))

    # loop over files with tqdm
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
                brightness = config.display.brightness
                display_image_path = config.display.output_fp
                rot90 = config.display.rot90
                os.system(
                    f"python scripts/prep_display_image.py --fp {_file} --output_path {display_image_path} --screen_res {screen_res[0]} {screen_res[1]} --hshift {hshift} --vshift {vshift} --pad {pad} --brightness {brightness} --rot90 {rot90}"
                )

                time.sleep(config.capture.delay)

                # -- take picture
                camera.capture(str(output_fp))

        # check if runtime is exceeded
        if config.runtime:
            proc_time = time.time() - start_time
            if proc_time > runtime_sec:
                print(f"-- measured {i+1} / {n_files} files")
                break

    proc_time = time.time() - start_time
    print(f"\nFinished, {proc_time/60.:.3f} minutes.")


if __name__ == "__main__":
    collect_dataset()
