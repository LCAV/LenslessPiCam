from lensless.utils.align_face import align_face
#from lensless.recon.ilo_stylegan2.ilo import LatentOptimizer
#print('Done')

import os
from torchvision.datasets.utils import download_and_extract_archive
import matplotlib.pyplot as plt
import shutil
import numpy as np

fp = 'data/celeba_mini/000019.jpg'

predictor_path = os.path.join("models", "shape_predictor_68_face_landmarks.dat")

if not os.path.exists("models"):
    os.makedirs("models")
if not os.path.exists(predictor_path):
    msg = "Do you want to download the face landmark model (61.1 Mo)?"
    valid = input("%s (Y/n) " % msg).lower() != "n"
    if valid:
        if os.path.exists(predictor_path + '.bz2'):
            os.remove(predictor_path + '.bz2')
        url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        filename = "shape_predictor_68_face_landmarks.dat.bz2"
        download_and_extract_archive(url, "models", filename=filename, remove_finished=True)


if not os.path.exists(fp):
    msg = "Do you want to download the sample CelebA dataset (764KB)?"
    valid = input("%s (Y/n) " % msg).lower() != "n"
    if valid:
        if os.path.exists("data/celeba_mini"):
            shutil.rmtree("data/celeba_mini")
        url = "https://drive.switch.ch/index.php/s/Q5OdDQMwhucIlt8/download"
        filename = "celeb_mini.zip"
        download_and_extract_archive(
            url, "data", filename=filename, remove_finished=True
        )


aligned = np.array(align_face(fp, predictor_path)) / 255

print(aligned.shape, aligned.min(), aligned.max())

plt.figure(figsize=(10,10))
plt.imshow(aligned)
plt.show()
