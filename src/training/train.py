"""Two-stage transfer learning training orchestrator for RetinaInsight.

Implements controlled feature extraction followed by fine-tuning with early stopping,
learning-rate scheduling, class weighting, and model checkpointing.
"""
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

# Direct Keras cache inside project root before TensorFlow/Keras initializes
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_LOCAL_KERAS_CACHE = _PROJECT_ROOT / ".keras"
if _LOCAL_KERAS_CACHE.is_dir() and "KERAS_HOME" not in os.environ:
    os.environ["KERAS_HOME"] = str(_LOCAL_KERAS_CACHE)

import tensorflow as tf

from configs.training_config import config
from src.data.dataset_loader import (
    compute_balanced_class_weights,
    create_dataset,
    load_manifest,
)
from src.evaluation.evaluate import evaluate_model
from src.evaluation.visualizer import plot_training_history
from src.models.transfer_models import build_model, set_backbone_trainable

LOGGER = logging.getLogger(__name__)


def set_reproducible_seeds(seed: int = config.RANDOM_SEED) -> None:
    """Set global seeds across Python, NumPy, and TensorFlow."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def train_model(
    architecture: str,
    epochs_stage1: int = config.STAGE_1_EPOCHS,
    epochs_stage2: int = config.STAGE_2_EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    lr_stage1: float = config.STAGE_1_LEARNING_RATE,
    lr_stage2: float = config.STAGE_2_LEARNING_RATE,
    use_class_weights: bool = True,
    unfreeze_top_n: int = 20,
    subsample_ratio: float | None = None,
    seed: int = config.RANDOM_SEED,
) -> dict[str, Any]:
    """Execute complete 2-stage transfer learning workflow for a given architecture.

    Stage 1: Train classification head with convolutional backbone frozen.
    Stage 2: Fine-tune top backbone layers with a reduced learning rate.
    """
    set_reproducible_seeds(seed)
    arch = architecture.strip().lower()
    if arch not in config.SUPPORTED_ARCHITECTURES:
        raise ValueError(f"Architecture '{arch}' not in {config.SUPPORTED_ARCHITECTURES}")

    LOGGER.info("Starting training pipeline for %s...", arch.upper())
    total_start_time = time.perf_counter()

    # Load manifests
    train_df = load_manifest(config.TRAIN_METADATA_PATH)
    val_df = load_manifest(config.VAL_METADATA_PATH)

    if subsample_ratio is not None and 0.0 < subsample_ratio < 1.0:
        LOGGER.info("Subsampling training dataset by factor of %.2f", subsample_ratio)
        train_df = (
            train_df.groupby("diagnosis", group_keys=False)
            .apply(lambda g: g.sample(frac=subsample_ratio, random_state=seed))
            .reset_index(drop=True)
        )
        val_df = (
            val_df.groupby("diagnosis", group_keys=False)
            .apply(lambda g: g.sample(frac=subsample_ratio, random_state=seed))
            .reset_index(drop=True)
        )
        LOGGER.info(
            "Subsampled sizes -> Train: %d, Val: %d",
            len(train_df),
            len(val_df),
        )

    # Class weights strictly from training data
    class_weights = None
    if use_class_weights:
        class_weights = compute_balanced_class_weights(train_df)
        LOGGER.info("Class weights configured: %s", class_weights)

    # Build datasets
    train_ds = create_dataset(
        manifest=train_df,
        architecture=arch,
        batch_size=batch_size,
        is_training=True,
        shuffle=True,
        seed=seed,
    )
    val_ds = create_dataset(
        manifest=val_df,
        architecture=arch,
        batch_size=batch_size,
        is_training=False,
        shuffle=False,
    )

    # Build model with frozen backbone
    model = build_model(
        architecture=arch,
        num_classes=config.NUM_CLASSES,
        dropout_rate=config.DROPOUT_RATE,
        trainable_backbone=False,
    )

    # Compile for Stage 1
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_stage1),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    checkpoints_dir = config.CHECKPOINTS_DIR
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    stage1_ckpt_path = checkpoints_dir / f"{arch}_stage1_best.keras"

    stage1_callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(stage1_ckpt_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    LOGGER.info(
        "--- STAGE 1: Feature Extraction (%s, %d epochs, lr=%s) ---",
        arch.upper(),
        epochs_stage1,
        lr_stage1,
    )
    history_stage1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_stage1,
        class_weight=class_weights,
        callbacks=stage1_callbacks,
        verbose=1,
    )

    # Stage 2: Fine-Tuning
    actual_stage1_epochs = len(history_stage1.history["loss"])
    history_stage2 = None

    if epochs_stage2 > 0:
        LOGGER.info(
            "--- STAGE 2: Fine-Tuning top %d layers (%s, %d epochs, lr=%s) ---",
            unfreeze_top_n,
            arch.upper(),
            epochs_stage2,
            lr_stage2,
        )
        set_backbone_trainable(model, architecture=arch, unfreeze_top_n=unfreeze_top_n)

        # Recompile with small learning rate to preserve pretrained features
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_stage2),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"],
        )

        stage2_ckpt_path = checkpoints_dir / f"{arch}_stage2_best.keras"
        stage2_callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(stage2_ckpt_path),
                monitor="val_loss",
                mode="min",
                save_best_only=True,
                verbose=1,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=config.REDUCE_LR_FACTOR,
                patience=config.REDUCE_LR_PATIENCE,
                min_lr=1e-7,
                verbose=1,
            ),
        ]

        history_stage2 = model.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=actual_stage1_epochs,
            epochs=actual_stage1_epochs + epochs_stage2,
            class_weight=class_weights,
            callbacks=stage2_callbacks,
            verbose=1,
        )

    total_training_duration = time.perf_counter() - total_start_time

    # Combine histories
    combined_history: dict[str, list[float]] = {}
    for metric in history_stage1.history:
        combined_history[metric] = [float(v) for v in history_stage1.history[metric]]
        if history_stage2 and metric in history_stage2.history:
            combined_history[metric].extend([float(v) for v in history_stage2.history[metric]])

    # Save final model
    saved_models_dir = config.SAVED_MODELS_DIR
    saved_models_dir.mkdir(parents=True, exist_ok=True)
    final_model_path = saved_models_dir / f"{arch}_final.keras"
    model.save(final_model_path)
    LOGGER.info("Saved final model to %s", final_model_path)

    # Save history JSON
    reports_dir = config.REPORTS_DIR
    figures_dir = config.FIGURES_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    history_file = reports_dir / f"{arch}_history.json"
    training_summary = {
        "architecture": arch,
        "epochs_stage1_completed": actual_stage1_epochs,
        "epochs_stage2_completed": len(history_stage2.history["loss"]) if history_stage2 else 0,
        "total_training_time_seconds": round(total_training_duration, 2),
        "use_class_weights": use_class_weights,
        "class_weights": class_weights,
        "history": combined_history,
    }
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(training_summary, f, indent=2)
    LOGGER.info("Saved training history to %s", history_file)

    # Plot learning curves
    curves_path = figures_dir / f"{arch}_training_curves.png"
    plot_training_history(
        history=combined_history,
        architecture=arch,
        output_path=curves_path,
        stage_split_epoch=actual_stage1_epochs if history_stage2 else None,
    )
    LOGGER.info("Saved learning curves plot to %s", curves_path)

    # Run comprehensive evaluation on validation set
    val_metrics = evaluate_model(
        model_or_path=model,
        manifest_or_path=val_df,
        architecture=arch,
        batch_size=batch_size,
        split_name="validation",
        save_outputs=True,
    )

    return {
        "architecture": arch,
        "model_path": str(final_model_path),
        "training_time_seconds": total_training_duration,
        "validation_metrics": val_metrics,
        "training_history": combined_history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RetinaInsight transfer learning models.")
    parser.add_argument(
        "--architecture",
        type=str,
        default="mobilenetv2",
        choices=config.SUPPORTED_ARCHITECTURES,
        help="Model architecture to train.",
    )
    parser.add_argument(
        "--epochs-stage1",
        type=int,
        default=config.STAGE_1_EPOCHS,
        help="Epochs for Stage 1 (feature extraction with frozen backbone).",
    )
    parser.add_argument(
        "--epochs-stage2",
        type=int,
        default=config.STAGE_2_EPOCHS,
        help="Epochs for Stage 2 (fine-tuning top layers).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.BATCH_SIZE,
        help="Batch size for training and validation.",
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=None,
        help="Optional fraction (0.0 < f < 1.0) to subsample dataset for fast verification.",
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable balanced class weighting.",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    train_model(
        architecture=args.architecture,
        epochs_stage1=args.epochs_stage1,
        epochs_stage2=args.epochs_stage2,
        batch_size=args.batch_size,
        use_class_weights=not args.no_class_weights,
        subsample_ratio=args.subsample,
    )


if __name__ == "__main__":
    main()
