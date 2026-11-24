"""Visualization tools for training history, confusion matrices, and model comparison."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from configs.training_config import config


def plot_training_history(
    history: dict[str, list[float]],
    architecture: str,
    output_path: Path | str,
    stage_split_epoch: int | None = None,
) -> Path:
    """Plot and save training and validation loss and accuracy curves."""
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history.get("loss", [])) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    axes[0].plot(epochs, history.get("loss", []), "o-", label="Train Loss", color="#1f77b4")
    if "val_loss" in history:
        axes[0].plot(epochs, history.get("val_loss", []), "s--", label="Val Loss", color="#ff7f0e")
    if stage_split_epoch:
        axes[0].axvline(x=stage_split_epoch + 0.5, color="red", linestyle=":", label="Fine-tuning Start")
    axes[0].set_title(f"{architecture.upper()} - Cross-Entropy Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Accuracy curve
    axes[1].plot(epochs, history.get("accuracy", []), "o-", label="Train Accuracy", color="#2ca02c")
    if "val_accuracy" in history:
        axes[1].plot(epochs, history.get("val_accuracy", []), "s--", label="Val Accuracy", color="#d62728")
    if stage_split_epoch:
        axes[1].axvline(x=stage_split_epoch + 0.5, color="red", linestyle=":", label="Fine-tuning Start")
    axes[1].set_title(f"{architecture.upper()} - Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_confusion_matrix(
    conf_mat: list[list[int]] | np.ndarray,
    architecture: str,
    output_path: Path | str,
    class_names: list[str] | None = None,
    normalize: bool = True,
) -> Path:
    """Plot and save a confusion matrix heatmap."""
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if class_names is None:
        class_names = [config.CLASS_NAMES[i] for i in range(config.NUM_CLASSES)]

    cm = np.array(conf_mat)
    if normalize:
        cm_display = cm.astype("float") / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1)
        fmt = ".2%"
    else:
        cm_display = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix - {architecture.upper()} (Validation)")
    ax.set_ylabel("True Severity Grade")
    ax.set_xlabel("Predicted Severity Grade")
    ax.tick_params(axis="x", rotation=25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_model_comparison(
    comparison_records: list[dict[str, Any]],
    output_path: Path | str,
) -> Path:
    """Generate comparative bar plots across evaluated model architectures."""
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(comparison_records)
    metrics = ["quadratic_weighted_kappa", "macro_f1", "accuracy"]
    titles = ["Quadratic Weighted Kappa", "Macro F1-Score", "Overall Accuracy"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    palette = sns.color_palette("muted")

    for ax, metric, title in zip(axes, metrics, titles):
        if metric in df.columns:
            sns.barplot(data=df, x="architecture", y=metric, hue="architecture", palette=palette, legend=False, ax=ax)
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel("Architecture")
            ax.set_ylabel(title)
            ax.set_ylim(0, 1.0)
            # Add value labels
            for p in ax.patches:
                val = p.get_height()
                if not np.isnan(val):
                    ax.annotate(
                        f"{val:.3f}",
                        (p.get_x() + p.get_width() / 2.0, val),
                        ha="center",
                        va="bottom",
                        fontsize=10,
                        xytext=(0, 4),
                        textcoords="offset points",
                    )
        ax.grid(axis="y", linestyle="--", alpha=0.7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path
