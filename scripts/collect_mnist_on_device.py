import numpy as np
import time
import os
import pathlib as plib
import click
import cv2
from mlxtend.data import loadlocal_mnist
from PIL import Image
from picamerax import PiCamera
import json
import requests
import gzip
import shutil


@click.command()
@click.option(
    "--input_dir",
    type=str,
    default="data/MNIST",
    help="Where raw MNIST data is stored.",
)
@click.option(
    "--output_dir",
    type=str,
    help="Output directory for measured images.",
)
@click.option(
    "--n_files",
    type=int,
    help="Number of files to collect. Default is all files.",
)
@click.option(
    "--test",
    is_flag=True,
    help="Measure test set, otherwise do train.",
)
@click.option(
    "--runtime",
    type=float,
    default=None,
    help="Runtime for script in hours, namely script stops after this many hours.",
)
@click.option(
    "--progress",
    type=int,
    default=100,
    help="How often to print progress.",
)
@click.option("--start", type=int, default=0, help="Start index for measuring files.")
@click.option("-v", "--verbose", is_flag=True)
def collect_mnist(input_dir, output_dir, n_files, verbose, test, runtime, progress, start):

    assert output_dir is not None

    img_og_dim = (28, 28)

    # TODO use after measurement!!
    interpolation = cv2.INTER_NEAREST

    if runtime:
        print(f"Running script for {runtime} hours...")
        # convert to seconds
        runtime = runtime * 60 * 60

    # display param
    screen_res = np.array((1920, 1200))
    hshift = 0
    vshift = 0
    pad = 50
    brightness = 100
    display_image_path = "/home/pi/LenslessPiCam_display/test.png"

    # load data
    if test:
        images_fn = "t10k-images-idx3-ubyte"
        labels_fn = "t10k-labels-idx1-ubyte"
    else:
        images_fn = "train-images-idx3-ubyte"
        labels_fn = "train-labels-idx1-ubyte"

    images_path = os.path.join(input_dir, images_fn)
    labels_path = os.path.join(input_dir, labels_fn)

    # download data if not in local path
    if not os.path.exists(images_path):
        url = f"http://yann.lecun.com/exdb/mnist/{images_fn}.gz"
        print(f"Downloading images from {url}")
        zipped_file = images_path + ".gz"
        with open(zipped_file, "wb") as f:
            r = requests.get(url)
            f.write(r.content)
        with gzip.open(zipped_file, "rb") as f_in:
            with open(images_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
    if not os.path.exists(labels_path):
        url = f"http://yann.lecun.com/exdb/mnist/{labels_fn}.gz"
        print(f"Downloading labels from {url}")
        zipped_file = labels_path + ".gz"
        with open(zipped_file, "wb") as f:
            r = requests.get(url)
            f.write(r.content)
        with gzip.open(zipped_file, "rb") as f_in:
            with open(labels_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

    X, y = loadlocal_mnist(
        images_path=images_path,
        labels_path=labels_path,
    )

    print("\nNumber of files :", len(y))
    if n_files:
        print(f"TEST : collecting first {n_files} files!")
    else:
        n_files = len(y)

    # set up camera for consistent photos
    # https://picamera.readthedocs.io/en/release-1.13/recipes1.html#capturing-consistent-images
    # https://picamerax.readthedocs.io/en/latest/fov.html?highlight=camera%20resolution#sensor-modes
    resolution = (640, 480)
    framerate = 30
    camera_iso = 100
    camera = PiCamera(resolution=resolution, framerate=framerate)
    # Set ISO to the desired value
    camera.iso = camera_iso
    # Wait for the automatic gain control to settle
    time.sleep(2)
    # Now fix the values
    camera.shutter_speed = camera.exposure_speed
    camera.exposure_mode = "off"
    g = camera.awb_gains
    camera.awb_mode = "off"
    camera.awb_gains = g

    # make output directory
    output_dir = plib.Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # save collection parameters
    # TODO : check if file already exists and if different parameters!!
    metadata = {
        "screen_res": screen_res.tolist(),
        "hshift": hshift,
        "vshift": vshift,
        "pad": pad,
        "brightness": brightness,
        "resolution": resolution,
        "framerate": framerate,
        "camera_iso": camera_iso,
        "awb_gains": (float(g[0]), float(g[1])),
    }
    metadata_fp = output_dir / "metadata.json"
    with open(metadata_fp, "w") as fp:
        json.dump(metadata, fp)

    # loop over files
    if test:
        subdir = output_dir / "test"
    else:
        subdir = output_dir / "train"
    subdir.mkdir(exist_ok=True)
    labels = []
    start_time = time.time()
    if start:
        print(f"Starting at {start}.")
    for i in range(start, n_files):

        if runtime:
            proc_time = time.time() - start_time
            if proc_time > runtime:
                print(f"-- measured {i} / {n_files} files")
                break

        if verbose:
            print(f"\nFILE : {i+1} / {n_files}")

        labels.append(y[i])

        # TODO check if measurement already exists
        output_fp = subdir / f"img{i}.png"
        if not os.path.isfile(output_fp):

            # reshape and normalize
            img = X[i]
            img = np.reshape(img, img_og_dim)

            """ DISPLAY """
            img_display = np.zeros((screen_res[1], screen_res[0]), dtype=img.dtype)

            if screen_res[0] < screen_res[1]:
                new_width = int(screen_res[0] / (1 + 2 * pad / 100))
                ratio = new_width / float(img_og_dim[0])
                new_height = int(ratio * img_og_dim[1])
            else:
                new_height = int(screen_res[1] / (1 + 2 * pad / 100))
                ratio = new_height / float(img_og_dim[1])
                new_width = int(ratio * img_og_dim[0])
            image_res = (new_width, new_height)
            img = cv2.resize(img, image_res, interpolation=interpolation)
            img_display[: image_res[1], : image_res[0]] = img

            # center
            img_display = np.roll(
                img_display, shift=int((screen_res[1] - image_res[1]) / 2), axis=0
            )
            img_display = np.roll(
                img_display, shift=int((screen_res[0] - image_res[0]) / 2), axis=1
            )

            if vshift:
                nx, _, _ = img.shape
                img = np.roll(img, shift=int(vshift * nx / 100), axis=0)

            if hshift:
                _, ny, _ = img.shape
                img = np.roll(img, shift=int(hshift * ny / 100), axis=1)

            if brightness:
                img = (img * brightness / 100).astype(np.uint8)

            # save to file
            im = Image.fromarray(img_display)
            im.convert("L").save(display_image_path)
            time.sleep(2)

            """ TAKE PICTURE """
            camera.capture(str(output_fp))

        if (i + 1) % progress == 0:
            proc_time = time.time() - start_time
            print(f"\n{i+1} / {n_files}, {proc_time / 60.:.3f} minutes")

    with open(subdir / "labels.txt", "w") as f:
        for item in labels:
            f.write("%s\n" % item)

    proc_time = time.time() - start_time
    print(f"Finished, {proc_time/60.:.3f} minutes.")


if __name__ == "__main__":
    collect_mnist()
