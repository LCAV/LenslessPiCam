"""
Install sklearn

pip install scikit-learn

ROC curve docs: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html


"""

import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import json
import pandas as pd
import seaborn as sn

scores_paths = {
    "ADMM10": "/root/LenslessPiCam/outputs/2024-03-25/23-36-06/scores_10_grayscaleTrue_down1_nfiles10000.json",
    "ADMM25": "/root/LenslessPiCam/outputs/2024-03-26/17-52-49/scores_25_grayscaleTrue_down1_nfiles10000.json",
    "ADMM50": "/root/LenslessPiCam/outputs/2024-03-27/10-49-08/scores_50_grayscaleTrue_down1_nfiles10000.json",
}

# initialize figure
plt.rcParams.update({"font.size": 20})
fig, ax = plt.subplots()
for method, scores_fp in scores_paths.items():
    # load scores
    with open(scores_fp, "r") as f:
        scores = json.load(f)

    # prepare scores
    y_true = []
    y_score = []
    n_psf = len(scores)
    accuracy = np.zeros(n_psf)
    confusion_matrix = np.zeros((n_psf, n_psf))
    for psf_idx in scores:
        y_true_idx = np.ones(n_psf)
        y_true_idx[int(psf_idx)] = 0
        for score in scores[psf_idx]:
            y_true += list(y_true_idx)
            y_score += list(score)

        # confusion matrix
        confusion_matrix[int(psf_idx)] = np.mean(np.array(scores[psf_idx]), axis=0)

        # compute accuracy for each PSF
        detected_mask = np.argmin(scores[psf_idx], axis=1)
        accuracy[int(psf_idx)] = np.mean(detected_mask == int(psf_idx))

    y_true = np.array(y_true).astype(bool)

    total_accuracy = np.mean(accuracy)
    print(f"Total accuracy ({method}): {total_accuracy:.2f}")

    # compute and plot confusion matrix
    df_cm = pd.DataFrame(
        confusion_matrix, index=[i for i in range(n_psf)], columns=[i for i in range(n_psf)]
    )
    plt.figure(figsize=(10, 7))
    # set font scale
    sn.set(font_scale=1.5)
    sn.heatmap(df_cm, annot=False, cbar=True, xticklabels=5, yticklabels=5)
    confusion_fn = f"confusion_matrix_{method}.png"
    plt.savefig(confusion_fn, bbox_inches="tight")
    print(f"Confusion matrix saved as {confusion_fn}")

    # compute the ROC curve
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    auc = metrics.roc_auc_score(y_true, y_score)

    # set font size
    lw = 3

    # create ROC curve
    ax.plot(fpr, tpr, label=f"{method}, AUC={auc:.2f}", linewidth=lw)

ax.set_ylabel("True Positive Rate")
ax.set_xlabel("False Positive Rate")
ax.legend()
ax.grid()

# save ROC curve
fig.savefig("roc_curve.png", bbox_inches="tight")
