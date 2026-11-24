"""Unit tests for RetinaInsight SQLite database manager and analytics."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.database.db_manager import DatabaseManager


class DatabaseManagerTests(unittest.TestCase):
    """Test suite for SQLite prediction logging and analytics retrieval."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_retina.db"
        self.db = DatabaseManager(db_path=self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_database_initialization(self):
        self.assertTrue(self.db_path.is_file())
        summary = self.db.get_analytics_summary()
        self.assertEqual(summary["total_predictions"], 0)

    def test_log_and_retrieve_prediction(self):
        uid = self.db.log_prediction(
            image_name="sample_eye.png",
            image_path="/path/to/sample_eye.png",
            model_name="mobilenetv2",
            predicted_grade=2,
            predicted_severity="Moderate",
            confidence=0.88,
            probabilities=[0.02, 0.05, 0.88, 0.03, 0.02],
            overlay_path="/path/to/overlay.png",
            execution_time_ms=45.2,
        )
        self.assertTrue(len(uid) > 10)

        df = self.db.get_recent_predictions(limit=10)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["prediction_uid"], uid)
        self.assertEqual(df.iloc[0]["predicted_severity"], "Moderate")
        self.assertAlmostEqual(df.iloc[0]["confidence"], 0.88)

        summary = self.db.get_analytics_summary()
        self.assertEqual(summary["total_predictions"], 1)
        self.assertEqual(summary["grade_counts"].get("Moderate"), 1)
        self.assertEqual(summary["model_usage"].get("mobilenetv2"), 1)

    def test_log_and_retrieve_evaluation(self):
        metrics = {
            "architecture": "mobilenetv2",
            "split": "validation",
            "accuracy": 0.84,
            "quadratic_weighted_kappa": 0.81,
            "macro_f1": 0.77,
            "weighted_f1": 0.83,
            "macro_precision": 0.79,
            "macro_recall": 0.76,
            "roc_auc_macro": 0.92,
            "num_samples": 540,
        }
        self.db.log_evaluation_metrics(metrics)
        df = self.db.get_model_benchmarks()
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["architecture"], "mobilenetv2")
        self.assertAlmostEqual(df.iloc[0]["quadratic_weighted_kappa"], 0.81)


if __name__ == "__main__":
    unittest.main()
