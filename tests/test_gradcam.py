"""Unit tests for RetinaInsight Grad-CAM explainability module."""
from __future__ import annotations

import os
from pathlib import Path

# Direct Keras cache inside project root before TensorFlow/Keras initializes
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LOCAL_KERAS_CACHE = _PROJECT_ROOT / ".keras"
if _LOCAL_KERAS_CACHE.is_dir() and "KERAS_HOME" not in os.environ:
    os.environ["KERAS_HOME"] = str(_LOCAL_KERAS_CACHE)

import unittest

import numpy as np
import tensorflow as tf
from PIL import Image

from configs.training_config import config
from src.data.dataset_loader import load_manifest
from src.explainability.gradcam import (
    explain_prediction,
    generate_gradcam_heatmap,
    overlay_heatmap_on_image,
)
from src.models.transfer_models import build_model


class GradCAMTests(unittest.TestCase):
    """Test suite for Grad-CAM activation heatmaps and overlays."""

    @classmethod
    def setUpClass(cls):
        # Build a lightweight model for test verification
        cls.model = build_model(architecture="mobilenetv2", trainable_backbone=False)

    def test_heatmap_generation_shape_and_bounds(self):
        dummy_tensor = np.random.uniform(0.0, 255.0, (1, 224, 224, 3)).astype(np.float32)
        heatmap, probs, target_class = generate_gradcam_heatmap(
            model=self.model,
            image_tensor=dummy_tensor,
            architecture="mobilenetv2",
        )

        self.assertEqual(heatmap.shape, (7, 7))
        self.assertGreaterEqual(float(np.min(heatmap)), 0.0)
        self.assertLessEqual(float(np.max(heatmap)), 1.0)
        self.assertEqual(len(probs), 5)
        self.assertIn(target_class, range(5))

    def test_overlay_creation(self):
        dummy_img = Image.new("RGB", (224, 224), color=(120, 80, 50))
        dummy_heatmap = np.random.uniform(0.0, 1.0, (7, 7)).astype(np.float32)

        overlay = overlay_heatmap_on_image(dummy_img, dummy_heatmap, alpha=0.4)
        self.assertEqual(overlay.shape, (224, 224, 3))
        self.assertEqual(overlay.dtype, np.uint8)

    def test_explain_prediction_on_real_image(self):
        val_df = load_manifest(config.VAL_METADATA_PATH)
        sample_path = val_df.iloc[0]["image_path"]

        result = explain_prediction(
            model=self.model,
            image_path=sample_path,
            architecture="mobilenetv2",
        )

        self.assertIn("predicted_grade", result)
        self.assertIn("confidence", result)
        self.assertIn("overlay_image", result)
        self.assertEqual(result["overlay_image"].shape, (224, 224, 3))


if __name__ == "__main__":
    unittest.main()
