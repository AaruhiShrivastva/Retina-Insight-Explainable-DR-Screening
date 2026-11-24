"""Unit tests for RetinaInsight training and evaluation metrics."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

# Set KERAS_HOME before tensorflow
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LOCAL_KERAS_CACHE = _PROJECT_ROOT / ".keras"
if _LOCAL_KERAS_CACHE.is_dir() and "KERAS_HOME" not in os.environ:
    os.environ["KERAS_HOME"] = str(_LOCAL_KERAS_CACHE)

import unittest

import numpy as np

from configs.training_config import config
from src.evaluation.metrics import compute_comprehensive_metrics
from src.evaluation.visualizer import (
    plot_confusion_matrix,
    plot_model_comparison,
    plot_training_history,
)


class EvaluationMetricsTests(unittest.TestCase):
    """Test suite for clinical metrics calculations and visual plots."""

    def test_compute_comprehensive_metrics_synthetic(self):
        # 10 synthetic samples across 5 classes
        y_true = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        # Perfect predictions
        y_prob = np.eye(5)[y_true]

        metrics = compute_comprehensive_metrics(y_true, y_prob, inference_time_seconds=0.1)

        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["quadratic_weighted_kappa"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(metrics["weighted_f1"], 1.0)
        self.assertIsNotNone(metrics["roc_auc_macro"])
        self.assertEqual(len(metrics["per_class"]), 5)
        self.assertEqual(len(metrics["confusion_matrix"]), 5)

    def test_plots_generate_valid_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Test training curve plot
            history = {
                "loss": [1.5, 1.0, 0.7],
                "val_loss": [1.4, 0.9, 0.65],
                "accuracy": [0.5, 0.7, 0.8],
                "val_accuracy": [0.52, 0.68, 0.79],
            }
            curve_file = tmp_path / "curve.png"
            plot_training_history(history, "mobilenetv2", curve_file, stage_split_epoch=2)
            self.assertTrue(curve_file.is_file())
            self.assertGreater(curve_file.stat().st_size, 1000)

            # Test confusion matrix plot
            cm = [[10, 1, 0, 0, 0], [1, 8, 2, 0, 0], [0, 1, 15, 1, 0], [0, 0, 2, 5, 1], [0, 0, 0, 1, 7]]
            cm_file = tmp_path / "cm.png"
            plot_confusion_matrix(cm, "mobilenetv2", cm_file)
            self.assertTrue(cm_file.is_file())
            self.assertGreater(cm_file.stat().st_size, 1000)

            # Test comparison plot
            records = [
                {"architecture": "efficientnetb0", "quadratic_weighted_kappa": 0.82, "macro_f1": 0.78, "accuracy": 0.84},
                {"architecture": "resnet50", "quadratic_weighted_kappa": 0.80, "macro_f1": 0.75, "accuracy": 0.82},
                {"architecture": "mobilenetv2", "quadratic_weighted_kappa": 0.79, "macro_f1": 0.74, "accuracy": 0.81},
            ]
            cmp_file = tmp_path / "cmp.png"
            plot_model_comparison(records, cmp_file)
            self.assertTrue(cmp_file.is_file())
            self.assertGreater(cmp_file.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
