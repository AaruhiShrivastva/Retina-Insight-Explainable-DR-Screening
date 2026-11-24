"""Pretrained transfer learning architectures for RetinaInsight.

Implements EfficientNetB0, ResNet50, and MobileNetV2 with standardized classification
heads and controlled fine-tuning capabilities.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Mapping

# Ensure local Keras cache directory is respected before TensorFlow/Keras initializes
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_LOCAL_KERAS_CACHE = _PROJECT_ROOT / ".keras"
if _LOCAL_KERAS_CACHE.is_dir() and "KERAS_HOME" not in os.environ:
    os.environ["KERAS_HOME"] = str(_LOCAL_KERAS_CACHE)

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2, ResNet50

from configs.training_config import config

LOGGER = logging.getLogger(__name__)

# Last convolutional layer names used for Explainable AI (Grad-CAM)
TARGET_CONV_LAYERS: Mapping[str, str] = {
    "efficientnetb0": "top_activation",
    "resnet50": "conv5_block3_out",
    "mobilenetv2": "out_relu",
}


def get_backbone(
    architecture: str,
    input_shape: tuple[int, int, int] = (*config.IMAGE_SIZE, config.IMAGE_CHANNELS),
) -> tf.keras.Model:
    """Instantiate a pretrained ImageNet convolutional backbone without top dense layers."""
    arch = architecture.strip().lower()
    if arch == "efficientnetb0":
        return EfficientNetB0(include_top=False, weights="imagenet", input_shape=input_shape)
    elif arch == "resnet50":
        return ResNet50(include_top=False, weights="imagenet", input_shape=input_shape)
    elif arch == "mobilenetv2":
        return MobileNetV2(include_top=False, weights="imagenet", input_shape=input_shape)
    else:
        raise ValueError(
            f"Unsupported architecture '{architecture}'. Supported: {config.SUPPORTED_ARCHITECTURES}"
        )


def build_model(
    architecture: str,
    num_classes: int = config.NUM_CLASSES,
    dropout_rate: float = config.DROPOUT_RATE,
    input_shape: tuple[int, int, int] = (*config.IMAGE_SIZE, config.IMAGE_CHANNELS),
    trainable_backbone: bool = False,
) -> tf.keras.Model:
    """Build a transfer learning model with a standardized classification head.

    Args:
        architecture: Name of the CNN backbone ('efficientnetb0', 'resnet50', 'mobilenetv2').
        num_classes: Number of target categories (5 for Diabetic Retinopathy grades).
        dropout_rate: Dropout fraction for regularization in the head.
        input_shape: Dimensions of the input tensor (H, W, C).
        trainable_backbone: Whether backbone weights are trainable initially.

    Returns:
        A compiled or uncompiled Keras Functional Model.
    """
    backbone = get_backbone(architecture=architecture, input_shape=input_shape)
    backbone.trainable = trainable_backbone

    inputs = tf.keras.Input(shape=input_shape, name="input_image")

    # Pass through feature extractor; keep training=False when frozen to preserve BN statistics
    x = backbone(inputs, training=trainable_backbone)

    # Standardized screening classification head
    x = layers.GlobalAveragePooling2D(name="head_gap")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dense(256, activation="relu", name="head_dense")(x)
    x = layers.Dropout(dropout_rate, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name=f"retinainsight_{architecture.lower()}",
    )
    LOGGER.info(
        "Built %s model with %d total parameters (trainable: %d, non-trainable: %d).",
        model.name,
        model.count_params(),
        sum(p.numpy().size for p in model.trainable_variables),
        sum(p.numpy().size for p in model.non_trainable_variables),
    )
    return model


def set_backbone_trainable(
    model: tf.keras.Model,
    architecture: str,
    unfreeze_top_n: int = 20,
) -> None:
    """Unfreeze the top N layers of the backbone for fine-tuning while preserving BatchNormalization.

    Deep learning best practice:
    During fine-tuning, keeping BatchNormalization layers frozen avoids destabilizing
    the moving mean and moving variance estimates when batch sizes are modest.
    """
    # Locate backbone layer inside the Functional model
    backbone = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            backbone = layer
            break

    if backbone is None:
        raise ValueError(f"Could not locate backbone sub-model in {model.name}")

    backbone.trainable = True

    # If unfreeze_top_n is specified, freeze layers prior to the cutoff
    if unfreeze_top_n > 0:
        cutoff = max(0, len(backbone.layers) - unfreeze_top_n)
        for i, layer in enumerate(backbone.layers):
            if i < cutoff:
                layer.trainable = False
            else:
                # Keep BatchNormalization layers frozen to prevent moving statistics corruption
                if isinstance(layer, layers.BatchNormalization):
                    layer.trainable = False
                else:
                    layer.trainable = True
    else:
        # Full unfreeze, but still freeze BatchNormalization layers
        for layer in backbone.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True

    trainable_count = sum(p.numpy().size for p in model.trainable_variables)
    LOGGER.info(
        "Backbone configured for fine-tuning. Trainable parameters: %d.",
        trainable_count,
    )


def get_target_conv_layer(model: tf.keras.Model, architecture: str) -> tf.keras.layers.Layer:
    """Retrieve the target convolutional layer for Grad-CAM explainability."""
    arch = architecture.strip().lower()
    if arch not in TARGET_CONV_LAYERS:
        raise ValueError(f"Unknown architecture '{architecture}' for Grad-CAM layer lookup.")

    target_layer_name = TARGET_CONV_LAYERS[arch]

    # Check inside the backbone model first
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            try:
                return layer.get_layer(target_layer_name)
            except ValueError:
                pass

    # Check top-level model layers
    try:
        return model.get_layer(target_layer_name)
    except ValueError as exc:
        raise ValueError(
            f"Target layer '{target_layer_name}' not found in model {model.name}."
        ) from exc
