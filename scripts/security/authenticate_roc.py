"""
This script is related to a patent that has been filed.

Please contact the EPFL Technology Transfer Office (https://tto.epfl.ch/, info.tto@epfl.ch) for licensing inquiries.

----

These script computes ROC curves for lensless authentication.

For this script, install:
```
pip install scikit-learn seaborn
```
ROC curve docs: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html

"""

import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import json
import pandas as pd
import seaborn as sn

font_scale = 2.3
plt.rcParams.update({"font.size": 30})
lw = 5  # linewidth
linestyles = ["--", "-.", ":"]

# scores_paths = {
#     "ADMM10": "/root/LenslessPiCam/outputs/2024-03-25/23-36-06/scores_10_grayscaleTrue_down1_nfiles10000.json",
#     "ADMM25": "/root/LenslessPiCam/outputs/2024-03-26/17-52-49/scores_25_grayscaleTrue_down1_nfiles10000.json",
#     "ADMM50": "/root/LenslessPiCam/outputs/2024-03-27/10-49-08/scores_50_grayscaleTrue_down1_nfiles10000.json",
# }

# scores_paths = {
#     "Data fid.": {
#         # "path": "/root/LenslessPiCam/outputs/2024-12-07/20-26-06/scores_Unet4M+U5+Unet4M_wave_psfNN_down1_nfiles3750_metricrecon.txt",
#         "path": "/root/LenslessPiCam/authenticate_learned/data_fid/scores_Unet4M+U5+Unet4M_wave_psfNN_down1_nfiles3750_metricrecon.txt",
#         "invert": True,    # if lower score is True
#     },
#     # "MSE": {
#     #     # "path": "/root/LenslessPiCam/outputs/2024-12-07/22-12-52/scores_Unet4M+U5+Unet4M_wave_psfNN_down1_nfiles3750_metricmse.txt",
#     #     "path": "/root/LenslessPiCam/authenticate_learned/mse/scores_Unet4M+U5+Unet4M_wave_psfNN_down1_nfiles3750_metricmse.txt",
#     #     "invert": True,    # if lower score is True
#     # },
#     "LPIPS": {
#         # "path": "/root/LenslessPiCam/outputs/2024-12-07/18-23-12/scores_Unet4M+U5+Unet4M_wave_psfNN_down1_nfiles3750_metriclpips.txt",
#         "path": "/root/LenslessPiCam/authenticate_learned/lpips/scores_Unet4M+U5+Unet4M_wave_psfNN_down1_nfiles3750_metriclpips.txt",
#         "invert": True,    # if lower score is True
#     },
# }
scores_paths = {
    "Data fid.": {
        # "path": "/root/LenslessPiCam/outputs/2024-12-08/07-17-49/scores_admm100_down1_nfiles3750_metricrecon.txt",
        "path": "/root/LenslessPiCam/authenticate_admm/recon/scores_admm100_down1_nfiles3750_metricrecon.txt",
        "invert": True,  # if lower score is True
    },
    # "MSE": {
    #     "path": "/root/LenslessPiCam/outputs/2024-12-08/19-53-17/scores_admm100_down1_nfiles3750_metricmse.txt",
    #     "invert": True,    # if lower score is True
    # },
    "LPIPS": {
        # "path": "/root/LenslessPiCam/outputs/2024-12-07/18-26-43/scores_admm100_down1_nfiles3750_metriclpips.txt",
        "path": "/root/LenslessPiCam/authenticate_admm/lpips/scores_admm100_down1_nfiles3750_metriclpips.txt",
        "invert": True,  # if lower score is True
    },
}

print_incorrect = False

# TODO way to get this without loading dataset?
n_files_per_mask = 250
mask_labels = list(np.arange(15)) * n_files_per_mask
mask_labels = np.array(mask_labels)

# initialize figure
fig, ax = plt.subplots()
for method, scores_dict in scores_paths.items():
    print(f"--- Processing {method}...")
    scores_fp = scores_dict["path"]
    invert = scores_dict["invert"]

    scores = []
    with open(scores_fp, "r") as f:
        for line in f:
            scores.append(json.loads(line))
    scores = np.array(scores)
    n_psf = len(scores)
    n_files = len(scores[0])

    # compute and plot confusion matrix
    confusion_matrix = np.zeros((n_psf, n_psf))
    accuracy = np.zeros(n_psf)
    incorrect = dict()
    n_incorrect = 0
    y_true = []  # for ROC curve
    y_score = []  # for ROC curve
    for psf_idx in range(n_psf):

        source_psf_mask = mask_labels == psf_idx
        confusion_matrix[psf_idx] = np.mean(np.array(scores[:, source_psf_mask]), axis=1)

        # for ROC curve
        y_true += list(source_psf_mask)
        y_score += list(scores[psf_idx])

        # compute accuracy for each PSF
        detected_mask = np.argmin(scores[:, source_psf_mask], axis=0)
        if print_incorrect:
            print(f"PSF {psf_idx} detected as: ", detected_mask)
        accuracy[int(psf_idx)] = np.mean(detected_mask == int(psf_idx))
        if accuracy[int(psf_idx)] < 1:
            incorrect_idx = np.where(detected_mask != int(psf_idx))[0]

            # reconvert idx back to original idx
            incorrect_idx = np.array([np.where(source_psf_mask)[0][i] for i in incorrect_idx])
            incorrect[int(psf_idx)] = [int(i) for i in incorrect_idx]
            n_incorrect += len(incorrect_idx)

    total_accuracy = np.mean(accuracy)
    print("Total accuracy: ", total_accuracy)
    print("Number of incorrect detections: ", n_incorrect)

    #### FOR OLD ADMM SCORES
    # # load scores
    # with open(scores_fp, "r") as f:
    #     scores = json.load(f)
    #
    # # prepare scores
    # y_true = []
    # y_score = []
    # n_psf = len(scores)
    # accuracy = np.zeros(n_psf)
    # confusion_matrix = np.zeros((n_psf, n_psf))
    # for psf_idx in scores:
    #     y_true_idx = np.ones(n_psf)
    #     y_true_idx[int(psf_idx)] = 0
    #     for score in scores[psf_idx]:
    #         y_true += list(y_true_idx)
    #         y_score += list(score)

    #     # confusion matrix
    #     confusion_matrix[int(psf_idx)] = np.mean(np.array(scores[psf_idx]), axis=0)

    #     # compute accuracy for each PSF
    #     detected_mask = np.argmin(scores[psf_idx], axis=1)
    #     accuracy[int(psf_idx)] = np.mean(detected_mask == int(psf_idx))

    # total_accuracy = np.mean(accuracy)
    # print(f"Total accuracy ({method}): {total_accuracy:.2f}")

    # compute and plot confusion matrix
    df_cm = pd.DataFrame(
        confusion_matrix, index=[i for i in range(n_psf)], columns=[i for i in range(n_psf)]
    )
    plt.figure(figsize=(10, 7))
    # set font scale
    sn.set(font_scale=font_scale)
    sn.heatmap(df_cm, annot=False, cbar=True, xticklabels=5, yticklabels=5)
    confusion_fn = f"confusion_matrix_{method}.png"
    plt.savefig(confusion_fn, bbox_inches="tight")
    print(f"Confusion matrix saved as {confusion_fn}")

    # compute the ROC curve
    y_true = np.array(y_true).astype(bool)
    y_score = np.array(y_score)
    if invert:
        y_score = -1 * y_score
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    auc = metrics.roc_auc_score(y_true, y_score)

    # create ROC curve
    ax.plot(fpr, tpr, label=f"{method}, AUC={auc:.2f}", linewidth=lw, linestyle=linestyles.pop())


# set axis font size
ax.set_ylabel("True Positive Rate")
ax.set_xlabel("False Positive Rate")
ax.legend()
ax.grid()


# save ROC curve
plt.tight_layout()
fig.savefig("roc_curve.png", bbox_inches="tight")
