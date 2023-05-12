import torchvision.datasets as dset
import numpy as np
import time
import os
import subprocess
import pathlib as plib
import click


@click.command()
@click.option(
    "--hostname",
    type=str,
    help="Hostname or IP address for display device.",
)
@click.option(
    "--camera_hostname",
    type=str,
    help="Hostname or IP address for capture device. If not provided, same as hostname",
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
    "--bayer",
    is_flag=True,
    help="Copy over raw Bayer data. Otherwise do de-mosaicing and downsampling on Raspberry Pi.",
)
@click.option(
    "--downsample",
    type=float,
    default=128,
    help="Amount to downsample.",
)
@click.option("-v", "--verbose", is_flag=True)
def collect_mnist(hostname, camera_hostname, output_dir, n_files, verbose, test, bayer, downsample):

    assert hostname is not None
    assert output_dir is not None

    # TODO : save metadata in JSON
    progress = 10

    # display param
    display_python = "~/LenslessPiCam/lensless_env/bin/python"
    display_tmp_file = "~/tmp_display.png"
    display_image_prep_script = "~/LenslessPiCam/scripts/prep_display_image.py"
    display_image_path = "~/LenslessPiCam_display/test.png"
    display_res = np.array((1920, 1200))
    hshift = 0
    vshift = 0
    pad = 50
    brightness = 100
    local_display_file = "display_tmp.png"

    # camera param
    camera_python = "~/LenslessPiCam/lensless_env/bin/python"
    if camera_hostname is None:
        camera_hostname = hostname
    sensor_mode = "0"
    exp = 0.2
    config_pause = 0
    capture_script = "~/LenslessPiCam/scripts/on_device_capture.py"

    # load datasets
    root = "./data"
    tr = None
    # tr = transforms.ToTensor()
    dataset = dset.MNIST(root=root, train=not test, download=True, transform=tr)

    print("\nNumber of files :", len(dataset))
    if n_files:
        print(f"TEST : collecting first {n_files} files!")
    else:
        n_files = len(dataset)

    # make output directory
    output_dir = plib.Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # loop over files
    if test:
        subdir = output_dir / "test"
    else:
        subdir = output_dir / "train"
    subdir.mkdir(exist_ok=True)
    labels = []
    start_time = time.time()
    for i in range(n_files):

        if verbose:
            print(f"\nFILE : {i+1} / {n_files}")
        img, label = dataset[i]
        labels.append(label)

        # TODO check if measurement already exists
        output_fp = subdir / f"img{i}.png"
        if not os.path.isfile(output_fp):

            # load and save as PNG
            img.save(local_display_file)

            # send to display
            if verbose:
                print("-- Displaying picture...")
            # -- copy to RPi
            os.system('scp %s "pi@%s:%s" ' % (local_display_file, hostname, display_tmp_file))

            # -- prep image on Pi
            prep_command = f"{display_python} {display_image_prep_script} --fp {display_tmp_file} \
                --pad {pad} --vshift {vshift} --hshift {hshift} --screen_res {display_res[0]} {display_res[1]} \
                --brightness {brightness} --output_path {display_image_path} "
            if verbose:
                print(f"Command : {prep_command}")
            subprocess.Popen(
                ["ssh", "pi@%s" % hostname, prep_command],
                shell=False,
            )
            # time.sleep(2)

            # take picture
            remote_fn = "remote_capture"
            pic_command = (
                f"{camera_python} {capture_script} --fn {remote_fn} --exp {exp} --iso 100 "
                f"--config_pause {config_pause} --sensor_mode {sensor_mode} --nbits_out 8 --sixteen --legacy"
            )
            if not bayer:
                pic_command += " --gray"
                if downsample:
                    pic_command += f" --down {downsample}"
            if verbose:
                print("-- Taking picture...")
                print(f"Command : {pic_command}")
            ssh = subprocess.Popen(
                ["ssh", "pi@%s" % camera_hostname, pic_command],
                shell=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            result = ssh.stdout.readlines()
            error = ssh.stderr.readlines()
            if error != []:
                raise ValueError("CAPTURE ERROR: %s" % error)
            if result == []:
                error = ssh.stderr.readlines()
                raise ValueError("CAPTURE ERROR: %s" % error)

            # copy over file
            remotefile = f"~/{remote_fn}.png"
            if verbose:
                print(f"Copying over picture as {output_fp}...")
            os.system('scp "pi@%s:%s" %s' % (camera_hostname, remotefile, output_fp))

        if (i + 1) % progress == 0:
            proc_time = time.time() - start_time
            print(f"\n{i+1} / {n_files}, {proc_time / 60.:.3f} minutes")

    with open(subdir / "labels.txt", "w") as f:
        for item in labels:
            f.write("%s\n" % item)

    proc_time = time.time() - start_time
    print(f"Finished, {proc_time/60.:.3f} minutes.")

    if os.path.isfile(local_display_file):
        os.remove(local_display_file)


if __name__ == "__main__":
    collect_mnist()
