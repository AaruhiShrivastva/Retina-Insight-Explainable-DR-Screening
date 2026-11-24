"""Data loading, manifest validation, and tf.data input pipeline for RetinaInsight."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.applications import efficientnet, mobilenet_v2, resnet50

from configs.training_config import config

LOGGER = logging.getLogger(__name__)


def load_manifest(manifest_path: Path | str) -> pd.DataFrame:
    """Load and validate a dataset manifest CSV file."""
    path = Path(manifest_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {path}")

    df = pd.read_csv(path)
    required_cols = {"image_path", "diagnosis"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Manifest {path.name} missing required columns: {required_cols - set(df.columns)}"
        )

    # Validate image paths exist
    existing_mask = df["image_path"].apply(lambda p: Path(str(p)).is_file())
    if not existing_mask.all():
        missing_count = int((~existing_mask).sum())
        LOGGER.warning(
            "Found %d missing image files in manifest %s; excluding them.",
            missing_count,
            path.name,
        )
        df = df[existing_mask].copy()

    # Validate labels are integers in 0..4
    df["diagnosis"] = pd.to_numeric(df["diagnosis"], errors="coerce")
    valid_labels_mask = df["diagnosis"].isin(range(config.NUM_CLASSES))
    if not valid_labels_mask.all():
        invalid_count = int((~valid_labels_mask).sum())
        LOGGER.warning(
            "Found %d invalid diagnosis labels in manifest %s; excluding them.",
            invalid_count,
            path.name,
        )
        df = df[valid_labels_mask].copy()

    df["diagnosis"] = df["diagnosis"].astype(int)
    return df.reset_index(drop=True)


def compute_balanced_class_weights(train_manifest: pd.DataFrame | Path | str) -> dict[int, float]:
    """Calculate balanced class weights strictly from training data."""
    if isinstance(train_manifest, (str, Path)):
        df = load_manifest(train_manifest)
    else:
        df = train_manifest

    labels = df["diagnosis"].to_numpy()
    unique_classes = np.arange(config.NUM_CLASSES)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=labels,
    )
    class_weights = {int(cls): float(weight) for cls, weight in zip(unique_classes, weights)}
    LOGGER.info("Computed balanced training class weights: %s", class_weights)
    return class_weights


def get_preprocessing_function(architecture: str) -> Callable[[tf.Tensor], tf.Tensor]:
    """Return the architecture-specific Keras preprocessing function."""
    arch = architecture.strip().lower()
    if arch == "efficientnetb0":
        # EfficientNet in Keras includes internal Rescaling layers [0, 255] -> [0, 1]
        return efficientnet.preprocess_input
    elif arch == "resnet50":
        # Zero-centers RGB to BGR with respect to ImageNet dataset
        return resnet50.preprocess_input
    elif arch == "mobilenetv2":
        # Scales pixel values to [-1, 1]
        return mobilenet_v2.preprocess_input
    else:
        raise ValueError(
            f"Unsupported architecture '{architecture}'. Supported: {config.SUPPORTED_ARCHITECTURES}"
        )


def augment_retinal_image(image: tf.Tensor) -> tf.Tensor:
    """Apply medically reasonable data augmentation to a retinal image."""
    # Retinal fundus photographs are rotationally symmetric: horizontal & vertical flips are valid
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    # Subtle brightness adjustment without washing out retinal microaneurysms/exudates
    image = tf.image.random_brightness(image, max_delta=0.06)
    # Subtle contrast variation
    image = tf.image.random_contrast(image, lower=0.95, upper=1.05)
    # Clip to valid [0, 255] float range
    return tf.clip_by_value(image, 0.0, 255.0)


def parse_and_preprocess(
    file_path: tf.Tensor,
    label: tf.Tensor,
    architecture: str,
    is_training: bool = False,
    image_size: tuple[int, int] = config.IMAGE_SIZE,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Read image bytes, resize, apply augmentation if training, and apply model normalization."""
    # Read file
    img_bytes = tf.io.read_file(file_path)
    image = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    # Resize to standardized dimensions (224, 224)
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32)

    # Apply data augmentation strictly for training
    if is_training:
        image = augment_retinal_image(image)

    # Architecture-specific normalization
    preprocess_fn = get_preprocessing_function(architecture)
    image = preprocess_fn(image)

    # One-hot encode label for categorical classification
    one_hot_label = tf.one_hot(label, depth=config.NUM_CLASSES, dtype=tf.float32)
    return image, one_hot_label


def create_dataset(
    manifest: pd.DataFrame | Path | str,
    architecture: str = "efficientnetb0",
    batch_size: int = config.BATCH_SIZE,
    is_training: bool = False,
    shuffle: bool = False,
    seed: int = config.RANDOM_SEED,
) -> tf.data.Dataset:
    """Build an efficient tf.data.Dataset pipeline with parallel loading and prefetching."""
    if isinstance(manifest, (str, Path)):
        df = load_manifest(manifest)
    else:
        df = manifest.copy()

    file_paths = df["image_path"].astype(str).tolist()
    labels = df["diagnosis"].astype(int).tolist()

    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(len(file_paths), config.SHUFFLE_BUFFER_SIZE), seed=seed)

    # Map preprocessing in parallel
    dataset = dataset.map(
        lambda path, lbl: parse_and_preprocess(
            path,
            lbl,
            architecture=architecture,
            is_training=is_training,
            image_size=config.IMAGE_SIZE,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset
