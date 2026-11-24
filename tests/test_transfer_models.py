"""Unit tests for RetinaInsight transfer learning architectures."""
from __future__ import annotations

import os
from pathlib import Path

# Ensure local Keras cache is set before TensorFlow/Keras initializes
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LOCAL_KERAS_CACHE = _PROJECT_ROOT / ".keras"
if _LOCAL_KERAS_CACHE.is_dir():
    os.environ["KERAS_HOME"] = str(_LOCAL_KERAS_CACHE)

import unittest

import numpy as np
import tensorflow as tf

from configs.training_config import config
from src.models.transfer_models import (
    TARGET_CONV_LAYERS,
    build_model,
    get_target_conv_layer,
    set_backbone_trainable,
)


class TransferModelTests(unittest.TestCase):
    """Test suite for model construction, head consistency, and fine-tuning transitions."""

    def test_build_all_architectures(self):
        """Verify that all three architectures build with consistent shapes."""
        dummy_input = tf.random.uniform((2, 224, 224, 3), minval=0.0, maxval=255.0)

        for arch in config.SUPPORTED_ARCHITECTURES:
            model = build_model(architecture=arch, num_classes=5, trainable_backbone=False)
            self.assertEqual(model.input_shape, (None, 224, 224, 3))
            self.assertEqual(model.output_shape, (None, 5))

            # Run a forward pass
            preds = model(dummy_input, training=False)
            self.assertEqual(preds.shape, (2, 5))
            # Verify softmax output property: row sums equal 1.0
            np.testing.assert_allclose(np.sum(preds.numpy(), axis=1), np.ones(2), atol=1e-5)

    def test_backbone_freezing_and_unfreezing(self):
        """Verify that stage-1 has frozen backbone and stage-2 unfreezes top layers."""
        model = build_model(architecture="mobilenetv2", trainable_backbone=False)
        stage_1_trainable = len(model.trainable_variables)

        # Unfreeze top 10 layers for fine-tuning
        set_backbone_trainable(model, architecture="mobilenetv2", unfreeze_top_n=10)
        stage_2_trainable = len(model.trainable_variables)

        self.assertGreater(stage_2_trainable, stage_1_trainable)

    def test_gradcam_target_layers_exist(self):
        """Verify that every architecture has its designated Grad-CAM convolutional layer."""
        for arch in config.SUPPORTED_ARCHITECTURES:
            model = build_model(architecture=arch, trainable_backbone=False)
            layer = get_target_conv_layer(model, arch)
            self.assertIsNotNone(layer)
            self.assertEqual(layer.name, TARGET_CONV_LAYERS[arch])


if __name__ == "__main__":
    unittest.main()
