import os
import subprocess
import numpy as np
import time
from pprint import pprint
from lensless.plot import plot_image, pixel_histogram
from lensless.io import load_image, load_psf
from lensless.util import resize
import cv2
import matplotlib.pyplot as plt
from lensless import FISTA


username = "pi2"
hostname = "128.179.193.201"
gamma = 2.2

original_fp = "data/original/thumbs_up.png"
exp = 0.02
brightness = 90

original_fp = "data/original/tree.png"
exp = 0.24
brightness = 100

# -- display parameters
REMOTE_PYTHON = "~/LenslessPiCam/lensless_env/bin/python"
REMOTE_IMAGE_PREP_SCRIPT = "~/LenslessPiCam/scripts/prep_display_image.py"
REMOTE_DISPLAY_PATH = "~/LenslessPiCam_display/test.png"
REMOTE_TMP_PATH = "~/tmp_display.png"
screen_res = np.array((1920, 1200))  # width, height
pad = 0
vshift = 0
hshift = 0


# -- capture parameters
pic_delay = 2
SENSOR_MODES = [
    "off",
    "auto",
    "sunlight",
    "cloudy",
    "shade",
    "tungsten",
    "fluorescent",
    "incandescent",
    "flash",
    "horizon",
]
REMOTE_CAPTURE_FP = "~/LenslessPiCam/scripts/on_device_capture.py"
iso = 100
config_pause = 2
sensor_mode = "0"
nbits_out = 12
nbits = 12
legacy = True
bayer = True
rgb = False
gray = False
down = None
raw_data_fn = "raw_data"


# -- reconstruction parameters
red_gain = 1.9
blue_gain = 1.2
psf_fp = "data/psf/tape_rgb_31032023.png"
downsample = 4
dtype = "float32"
use_torch = True
torch_device = "cuda:0"
disp_iter = 50

# ---- fista
n_iter = 300
tk = 1


# 1) Copy file to Raspbery Pi
print("\nCopying over picture...")
os.system('scp %s "%s@%s:%s" ' % (original_fp, username, hostname, REMOTE_TMP_PATH))

prep_command = f"{REMOTE_PYTHON} {REMOTE_IMAGE_PREP_SCRIPT} --fp {REMOTE_TMP_PATH} \
    --pad {pad} --vshift {vshift} --hshift {hshift} --screen_res {screen_res[0]} {screen_res[1]} \
    --brightness {brightness} --output_path {REMOTE_DISPLAY_PATH} "
print(f"COMMAND : {prep_command}")
subprocess.Popen(
    ["ssh", "%s@%s" % (username, hostname), prep_command],
    shell=False,
)

# 2) Take picture
time.sleep(pic_delay)  # for picture to display
print("\nTaking picture...")

remote_fn = "remote_capture"
pic_command = (
    f"{REMOTE_PYTHON} {REMOTE_CAPTURE_FP} --fn {remote_fn} --exp {exp} --iso {iso} "
    f"--config_pause {config_pause} --sensor_mode {sensor_mode} --nbits_out {nbits_out}"
)
if nbits > 8:
    pic_command += " --sixteen"
if rgb:
    pic_command += " --rgb"
if legacy:
    pic_command += " --legacy"
if gray:
    pic_command += " --gray"
if down:
    pic_command += f" --down {down}"
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
    raise ValueError("ERROR: %s" % error)
if result == []:
    error = ssh.stderr.readlines()
    raise ValueError("ERROR: %s" % error)
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

# copy over file
# more pythonic? https://stackoverflow.com/questions/250283/how-to-scp-in-python
remotefile = f"~/{remote_fn}.png"
localfile = f"{raw_data_fn}.png"
print(f"\nCopying over picture as {localfile}...")
os.system('scp "%s@%s:%s" %s' % (username, hostname, remotefile, localfile))

if rgb or gray:
    img = load_image(localfile, verbose=True)

else:
    # get white balance gains
    if red_gain is None:
        red_gain = float(result_dict["Red gain"])
    if blue_gain is None:
        blue_gain = float(result_dict["Blue gain"])

    # load image
    print("\nLoading picture...")
    img = load_image(
        localfile,
        verbose=True,
        bayer=True,
        blue_gain=blue_gain,
        red_gain=red_gain,
        nbits_out=nbits_out,
    )

    # write RGB data
    if not bayer:
        cv2.imwrite(localfile, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


# plot RGB
if not gray:
    ax = plot_image(img, gamma=gamma)
    ax.set_title("RGB")

    # plot histogram, useful for checking clipping
    pixel_histogram(img)

else:
    ax = plot_image(img, gamma=gamma)
    pixel_histogram(img)


# 3) Reconstruct

# -- prepare data
psf, bg = load_psf(psf_fp, downsample=downsample, return_float=True, return_bg=True, dtype=dtype)
ax = plot_image(psf[0], gamma=gamma)
ax.set_title("PSF")

data = np.array(img, dtype=dtype)
data -= bg
data = np.clip(data, a_min=0, a_max=data.max())

if len(data.shape) == 3:
    data = data[np.newaxis, :, :, :]
elif len(data.shape) == 2:
    data = data[np.newaxis, :, :, np.newaxis]

if data.shape != psf.shape:
    # in DiffuserCam dataset, images are already reshaped
    data = resize(data, shape=psf.shape)
data /= np.linalg.norm(data.ravel())

psf = np.array(psf, dtype=dtype)
data = np.array(data, dtype=dtype)
if use_torch:
    import torch

    if dtype == "float32":
        torch_dtype = torch.float32
    elif dtype == "float64":
        torch_dtype = torch.float64
    else:
        raise ValueError("dtype must be float32 or float64")

    psf = torch.from_numpy(psf).type(torch_dtype).to(torch_device)
    data = torch.from_numpy(data).type(torch_dtype).to(torch_device)

# -- apply algo
start_time = time.time()
recon = FISTA(
    psf,
    tk=tk,
)
recon.set_data(data)
res = recon.apply(
    n_iter=n_iter,
    disp_iter=disp_iter,
    gamma=gamma,
    plot=True,
)
print(f"Processing time : {time.time() - start_time} s")


plt.show()
