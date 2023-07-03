import numpy as np
import os
from lensless import VirtualSensor, SensorOptions
from lensless.utils.io import load_image
from lensless.utils.image import rgb2gray


def test_sensor_size():
    for sensor_name in SensorOptions.values():
        sensor = VirtualSensor.from_name(sensor_name)
        if np.all(sensor.pixel_size * sensor.resolution > sensor.size):
            raise ValueError(
                f"Discrepancy in sensor {sensor_name} size. Pixel size * resolution = {sensor.pixel_size * sensor.resolution}, while size = {sensor.size}"
            )


def test_sensor_landscape():
    for sensor_name in SensorOptions.values():
        sensor = VirtualSensor.from_name(sensor_name)
        assert (
            sensor.resolution[0] <= sensor.resolution[1]
        ), f"Sensor {sensor_name} is not landscape."


def test_virtual_image():
    for sensor_name in SensorOptions.values():
        sensor = VirtualSensor.from_name(sensor_name)

        for bit_depth in sensor.bit_depth:
            img = sensor.capture()
            assert np.all(
                img.shape == sensor.image_shape
            ), f"Sensor {sensor_name} image shape is {img.shape}, while expected shape is {sensor.image_shape}."
            assert (
                img.max() <= 2**bit_depth - 1
            ), f"Sensor {sensor_name} image max value is {img.max()}, while expected max value is {2**bit_depth - 1}."


def test_virtual_image_from_rgb_file():
    """
    rgb > rgb
    rgb > gray
    """
    fp = os.path.join(os.path.dirname(__file__), "..", "data", "original", "tree.png")
    for sensor_name in SensorOptions.values():
        sensor = VirtualSensor.from_name(sensor_name)

        for bit_depth in sensor.bit_depth:
            img = sensor.capture(scene=fp, bit_depth=bit_depth)
            assert np.all(
                img.shape == sensor.image_shape
            ), f"Sensor {sensor_name} image shape is {img.shape}, while expected shape is {sensor.image_shape}."
            assert (
                img.max() <= 2**bit_depth - 1
            ), f"Sensor {sensor_name} image max value is {img.max()}, while expected max value is {2**bit_depth - 1}."


def test_virtual_image_from_gray_file():
    """
    gray > rgb
    gray > gray
    """
    fp = os.path.join(os.path.dirname(__file__), "..", "data", "original", "mnist_3.png")
    for sensor_name in SensorOptions.values():
        sensor = VirtualSensor.from_name(sensor_name)

        for bit_depth in sensor.bit_depth:
            img = sensor.capture(scene=fp, bit_depth=bit_depth)
            assert np.all(
                img.shape == sensor.image_shape
            ), f"Sensor {sensor_name} image shape is {img.shape}, while expected shape is {sensor.image_shape}."
            assert (
                img.max() <= 2**bit_depth - 1
            ), f"Sensor {sensor_name} image max value is {img.max()}, while expected max value is {2**bit_depth - 1}."


def test_virtual_image_from_rgb_data(save=False):
    fp = os.path.join(os.path.dirname(__file__), "..", "data", "original", "tree.png")

    # provided normalized float data
    img = load_image(fp)
    img = img.astype(np.float32)
    img /= img.max()
    for sensor_name in SensorOptions.values():
        sensor = VirtualSensor.from_name(sensor_name)

        for bit_depth in sensor.bit_depth:
            img_cap = sensor.capture(scene=img, bit_depth=bit_depth)
            assert np.all(
                img_cap.shape == sensor.image_shape
            ), f"Sensor {sensor_name} image shape is {img_cap.shape}, while expected shape is {sensor.image_shape}."
            assert (
                img_cap.max() <= 2**bit_depth - 1
            ), f"Sensor {sensor_name} image max value is {img_cap.max()}, while expected max value is {2**bit_depth - 1}."

        # save file
        if save:
            from lensless.utils.io import save_image

            save_image(img_cap, f"test_{sensor_name}.png")
            print(sensor_name, img_cap.shape)


def test_virtual_image_from_gray_data():
    fp = os.path.join(os.path.dirname(__file__), "..", "data", "original", "tree.png")

    # provided normalized float data
    img = load_image(fp)
    img = img.astype(np.float32)
    img /= img.max()
    img = rgb2gray(img, keepchanneldim=False)
    for sensor_name in SensorOptions.values():
        sensor = VirtualSensor.from_name(sensor_name)

        for bit_depth in sensor.bit_depth:
            img_cap = sensor.capture(scene=img, bit_depth=bit_depth)
            assert np.all(
                img_cap.shape == sensor.image_shape
            ), f"Sensor {sensor_name} image shape is {img_cap.shape}, while expected shape is {sensor.image_shape}."
            assert (
                img_cap.max() <= 2**bit_depth - 1
            ), f"Sensor {sensor_name} image max value is {img_cap.max()}, while expected max value is {2**bit_depth - 1}."


def test_downsample():
    fp = os.path.join(os.path.dirname(__file__), "..", "data", "original", "tree.png")
    downsample = 4

    for sensor_name in SensorOptions.values():
        sensor = VirtualSensor.from_name(sensor_name)
        new_res = (sensor.resolution / downsample).astype(int)
        sensor.downsample(downsample)

        img = sensor.capture(scene=fp)
        assert np.all(
            img.shape == sensor.image_shape
        ), f"Sensor {sensor_name} image shape is {img.shape}, while expected shape is {sensor.image_shape}."
        assert np.all(
            img.shape[:2] == new_res
        ), f"Sensor {sensor_name} image shape is {img.shape}, while expected shape is {new_res}."


if __name__ == "__main__":
    test_sensor_size()
    test_sensor_landscape()
    test_virtual_image()
    test_virtual_image_from_rgb_file()
    test_virtual_image_from_gray_file()
    test_virtual_image_from_rgb_data(save=True)
    test_virtual_image_from_gray_data()
    test_downsample()
