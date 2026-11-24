"""Model evaluation runner across validation and test splits."""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Set KERAS_HOME before tensorflow
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_LOCAL_KERAS_CACHE = _PROJECT_ROOT / ".keras"
if _LOCAL_KERAS_CACHE.is_dir() and "KERAS_HOME" not in os.environ:
    os.environ["KERAS_HOME"] = str(_LOCAL_KERAS_CACHE)

import tensorflow as tf

from configs.training_config import config
from src.data.dataset_loader import create_dataset, load_manifest
from src.evaluation.metrics import compute_comprehensive_metrics
from src.evaluation.visualizer import plot_confusion_matrix

LOGGER = logging.getLogger(__name__)


def evaluate_model(
    model_or_path: tf.keras.Model | Path | str,
    manifest_or_path: pd.DataFrame | Path | str,
    architecture: str,
    batch_size: int = config.BATCH_SIZE,
    split_name: str = "validation",
    save_outputs: bool = True,
) -> dict[str, Any]:
    """Evaluate a trained model against a specified manifest split.

    Args:
        model_or_path: Trained Keras model instance or path to saved .keras file.
        manifest_or_path: Manifest DataFrame or path to manifest CSV.
        architecture: Backbone architecture identifier.
        batch_size: Evaluation batch size.
        split_name: Name of the partition ('validation' or 'test').
        save_outputs: Whether to write metrics JSON and confusion matrix plot to disk.

    Returns:
        Dictionary of comprehensive evaluation metrics.
    """
    # Load model if given a path
    if isinstance(model_or_path, (str, Path)):
        model_path = Path(model_or_path).resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"Model weight file not found: {model_path}")
        LOGGER.info("Loading model from %s", model_path)
        model = tf.keras.models.load_model(model_path)
    else:
        model = model_or_path

    # Load manifest
    if isinstance(manifest_or_path, (str, Path)):
        df = load_manifest(manifest_or_path)
    else:
        df = manifest_or_path.copy()

    y_true = df["diagnosis"].to_numpy()

    # Create unaugmented evaluation dataset
    dataset = create_dataset(
        manifest=df,
        architecture=architecture,
        batch_size=batch_size,
        is_training=False,
        shuffle=False,
    )

    LOGGER.info(
        "Evaluating %s on %s split (%d samples)...",
        architecture,
        split_name,
        len(df),
    )

    start_time = time.perf_counter()
    y_prob = model.predict(dataset, verbose=1)
    inference_duration = time.perf_counter() - start_time

    metrics = compute_comprehensive_metrics(
        y_true=y_true,
        y_prob=y_prob,
        inference_time_seconds=inference_duration,
    )
    metrics["architecture"] = architecture.lower()
    metrics["split"] = split_name

    LOGGER.info(
        "%s [%s] Results -> Accuracy: %.4f, QWK: %.4f, Macro F1: %.4f",
        architecture.upper(),
        split_name.capitalize(),
        metrics["accuracy"],
        metrics["quadratic_weighted_kappa"],
        metrics["macro_f1"],
    )

    if save_outputs:
        reports_dir = config.REPORTS_DIR
        figures_dir = config.FIGURES_DIR
        reports_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics JSON
        metrics_file = reports_dir / f"{architecture.lower()}_{split_name}_metrics.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        LOGGER.info("Saved evaluation metrics to %s", metrics_file)

        # Save Confusion Matrix plot
        cm_file = figures_dir / f"{architecture.lower()}_{split_name}_confusion_matrix.png"
        plot_confusion_matrix(
            conf_mat=metrics["confusion_matrix"],
            architecture=architecture,
            output_path=cm_file,
            normalize=True,
        )
        LOGGER.info("Saved confusion matrix to %s", cm_file)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a RetinaInsight trained transfer model.")
    parser.add_argument(
        "--architecture",
        type=str,
        required=True,
        choices=config.SUPPORTED_ARCHITECTURES,
        help="Model architecture to evaluate.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["validation", "test"],
        help="Dataset split to evaluate on.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional path to model file. Defaults to models/saved/{arch}_final.keras.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.BATCH_SIZE,
        help="Evaluation batch size.",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    model_path = args.model_path or (config.SAVED_MODELS_DIR / f"{args.architecture}_final.keras")
    manifest_path = config.VAL_METADATA_PATH if args.split == "validation" else config.TEST_METADATA_PATH

    evaluate_model(
        model_or_path=model_path,
        manifest_or_path=manifest_path,
        architecture=args.architecture,
        batch_size=args.batch_size,
        split_name=args.split,
    )


if __name__ == "__main__":
    main()
