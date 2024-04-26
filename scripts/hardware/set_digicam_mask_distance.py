import hydra
from lensless.hardware.utils import set_mask_sensor_distance


@hydra.main(version_base=None, config_path="../../configs", config_name="digicam_config")
def config_digicam(config):

    rpi_username = config.rpi.username
    rpi_hostname = config.rpi.hostname

    # set mask to sensor distance
    set_mask_sensor_distance(config.z, rpi_username, rpi_hostname, max_distance=40)


if __name__ == "__main__":
    config_digicam()
