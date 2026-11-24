"""Clinical and statistical evaluation metrics for RetinaInsight."""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from configs.training_config import config


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: dict[int, str] = config.CLASS_NAMES,
    inference_time_seconds: float | None = None,
) -> dict[str, Any]:
    """Compute all evaluation metrics required for medical screening assessment.

    Args:
        y_true: Ground truth integer class indices (N,).
        y_prob: Predicted probability distribution (N, num_classes).
        class_names: Mapping from integer class to clinical name.
        inference_time_seconds: Total duration spent computing predictions.

    Returns:
        Structured dictionary containing scalar metrics, per-class metrics, and confusion matrix.
    """
    y_pred = np.argmax(y_prob, axis=1)
    num_samples = len(y_true)

    # Core scalar metrics
    accuracy = float(accuracy_score(y_true, y_pred))
    macro_precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_precision = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_recall = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # Quadratic Weighted Kappa (QWK) - standard clinical benchmark metric for Diabetic Retinopathy
    # Penalizes misclassifications proportionally to the square of grade differences
    qwk = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))

    # Multi-class ROC-AUC (One-vs-Rest)
    try:
        # Convert y_true to one-hot for AUC calculation
        y_true_oh = np.eye(config.NUM_CLASSES)[y_true]
        roc_auc_macro = float(roc_auc_score(y_true_oh, y_prob, multi_class="ovr", average="macro"))
    except ValueError:
        roc_auc_macro = None

    # Per-class metrics
    per_class_report = classification_report(
        y_true,
        y_pred,
        labels=list(range(config.NUM_CLASSES)),
        target_names=[class_names[i] for i in range(config.NUM_CLASSES)],
        output_dict=True,
        zero_division=0,
    )

    per_class_metrics = {}
    for i in range(config.NUM_CLASSES):
        name = class_names[i]
        stats = per_class_report.get(name, {})
        per_class_metrics[name] = {
            "grade": i,
            "precision": float(stats.get("precision", 0.0)),
            "recall": float(stats.get("recall", 0.0)),
            "f1_score": float(stats.get("f1-score", 0.0)),
            "support": int(stats.get("support", 0)),
        }

    # Confusion matrix
    conf_mat = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(config.NUM_CLASSES)),
    ).tolist()

    # Inference speed benchmarking
    latency_ms_per_sample = (
        (inference_time_seconds * 1000.0 / num_samples)
        if inference_time_seconds and num_samples > 0
        else None
    )

    return {
        "num_samples": num_samples,
        "accuracy": round(accuracy, 4),
        "quadratic_weighted_kappa": round(qwk, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "macro_precision": round(macro_precision, 4),
        "weighted_precision": round(weighted_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "weighted_recall": round(weighted_recall, 4),
        "roc_auc_macro": round(roc_auc_macro, 4) if roc_auc_macro is not None else None,
        "inference_latency_ms_per_sample": round(latency_ms_per_sample, 2) if latency_ms_per_sample else None,
        "per_class": per_class_metrics,
        "confusion_matrix": conf_mat,
    }
