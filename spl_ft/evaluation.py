"""Metric computation + confusion-matrix plotting shared by both
evaluation protocols (known-players / unseen-players).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "n_test": int(len(y_true)),
        "make_rate": float(np.mean(y_true)) if len(y_true) else float("nan"),
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if len(y_true) > 0 and len(set(y_true)) > 1 else (
            float(accuracy_score(y_true, y_pred)) if len(y_true) else float("nan")
        ),
        "confusion_matrix_tn_fp_fn_tp": cm.ravel().tolist(),  # [TN, FP, FN, TP] for labels [0,1]
    }


def plot_confusion_matrix(ax, cm: np.ndarray, title: str):
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["missed", "made"])
    ax.set_yticklabels(["missed", "made"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title(title, fontsize=9)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    return im
