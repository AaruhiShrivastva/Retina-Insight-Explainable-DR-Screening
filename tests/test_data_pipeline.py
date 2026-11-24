"""Unit tests for RetinaInsight dataset loader and preprocessing pipeline."""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf

from configs.training_config import config
from src.data.dataset_loader import (
    compute_balanced_class_weights,
    create_dataset,
    get_preprocessing_function,
    load_manifest,
)


class DataPipelineTests(unittest.TestCase):
    """Test suite for data loading, manifest reading, and preprocessing."""

    def test_manifest_loads_and_verifies(self):
        train_df = load_manifest(config.TRAIN_METADATA_PATH)
        self.assertEqual(len(train_df), 2521)
        self.assertTrue({"image_path", "diagnosis"}.issubset(train_df.columns))
        self.assertTrue(train_df["diagnosis"].isin(range(5)).all())

    def test_class_weights_calculation(self):
        weights = compute_balanced_class_weights(config.TRAIN_METADATA_PATH)
        self.assertEqual(set(weights.keys()), {0, 1, 2, 3, 4})
        # Majority class (Grade 0) should have the smallest weight (< 1.0)
        # Minority class (Grade 3) should have the largest weight (> 1.0)
        self.assertLess(weights[0], weights[3])
        self.assertGreater(weights[3], weights[1])
        for w in weights.values():
            self.assertGreater(w, 0.0)

    def test_architecture_preprocessing_resolution(self):
        for arch in config.SUPPORTED_ARCHITECTURES:
            fn = get_preprocessing_function(arch)
            self.assertTrue(callable(fn))

        with self.assertRaises(ValueError):
            get_preprocessing_function("unsupported_model_xyz")

    def test_tf_dataset_batch_shapes(self):
        """Verify that a small batch yields correct tensor shapes and types."""
        val_df = load_manifest(config.VAL_METADATA_PATH).head(8)
        dataset = create_dataset(
            manifest=val_df,
            architecture="efficientnetb0",
            batch_size=4,
            is_training=False,
            shuffle=False,
        )

        for images, labels in dataset.take(1):
            self.assertEqual(images.shape, (4, 224, 224, 3))
            self.assertEqual(labels.shape, (4, 5))
            self.assertEqual(images.dtype, tf.float32)
            self.assertEqual(labels.dtype, tf.float32)
            # Verify one-hot encoding properties
            label_sums = np.sum(labels.numpy(), axis=1)
            np.testing.assert_allclose(label_sums, np.ones(4))


if __name__ == "__main__":
    unittest.main()
