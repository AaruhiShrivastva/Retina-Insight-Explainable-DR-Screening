"""Database manager handling CRUD operations and analytics queries for RetinaInsight."""
from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

import pandas as pd

from configs.training_config import config
from src.database.schema import (
    CREATE_EVALUATIONS_TABLE,
    CREATE_INDEXES,
    CREATE_PREDICTIONS_TABLE,
)

LOGGER = logging.getLogger(__name__)


class DatabaseManager:
    """Manages SQLite connections, schema initialization, and analytics queries."""

    def __init__(self, db_path: Path | str = config.DATABASE_PATH) -> None:
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager that opens and guarantees connection closure (critical for Windows)."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def init_db(self) -> None:
        """Create tables and indexes if they do not exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(CREATE_PREDICTIONS_TABLE)
            cursor.execute(CREATE_EVALUATIONS_TABLE)
            cursor.executescript(CREATE_INDEXES)
            conn.commit()
        LOGGER.info("Database initialized at %s", self.db_path)

    def log_prediction(
        self,
        image_name: str,
        image_path: str,
        model_name: str,
        predicted_grade: int,
        predicted_severity: str,
        confidence: float,
        probabilities: list[float],
        overlay_path: str | None = None,
        execution_time_ms: float | None = None,
    ) -> str:
        """Record a screening inference result."""
        prediction_uid = str(uuid.uuid4())
        # Pad probabilities to 5 classes if needed
        probs = (probabilities + [0.0] * 5)[:5]

        query = """
        INSERT INTO predictions (
            prediction_uid, image_name, image_path, model_name,
            predicted_grade, predicted_severity, confidence,
            prob_grade_0, prob_grade_1, prob_grade_2, prob_grade_3, prob_grade_4,
            overlay_path, execution_time_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        with self.get_connection() as conn:
            conn.execute(
                query,
                (
                    prediction_uid,
                    image_name,
                    str(image_path),
                    model_name,
                    predicted_grade,
                    predicted_severity,
                    confidence,
                    probs[0],
                    probs[1],
                    probs[2],
                    probs[3],
                    probs[4],
                    str(overlay_path) if overlay_path else None,
                    execution_time_ms,
                ),
            )
            conn.commit()
        return prediction_uid

    def get_recent_predictions(self, limit: int = 50) -> pd.DataFrame:
        """Retrieve recent predictions as a pandas DataFrame."""
        query = """
        SELECT prediction_uid, timestamp, image_name, model_name,
               predicted_grade, predicted_severity, confidence,
               execution_time_ms, overlay_path
        FROM predictions
        ORDER BY timestamp DESC
        LIMIT ?;
        """
        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=(limit,))
        return df

    def get_analytics_summary(self) -> dict[str, Any]:
        """Aggregate statistical metrics for dashboard analytics."""
        with self.get_connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM predictions;").fetchone()[0]
            if total == 0:
                return {
                    "total_predictions": 0,
                    "avg_confidence": 0.0,
                    "grade_counts": {},
                    "model_usage": {},
                }

            avg_conf = conn.execute("SELECT AVG(confidence) FROM predictions;").fetchone()[0] or 0.0

            grade_rows = conn.execute(
                "SELECT predicted_severity, COUNT(*) FROM predictions GROUP BY predicted_severity;"
            ).fetchall()
            grade_counts = {row[0]: row[1] for row in grade_rows}

            model_rows = conn.execute(
                "SELECT model_name, COUNT(*) FROM predictions GROUP BY model_name;"
            ).fetchall()
            model_usage = {row[0]: row[1] for row in model_rows}

        return {
            "total_predictions": total,
            "avg_confidence": round(float(avg_conf), 4),
            "grade_counts": grade_counts,
            "model_usage": model_usage,
        }

    def log_evaluation_metrics(self, metrics: dict[str, Any]) -> None:
        """Log a benchmark evaluation run."""
        query = """
        INSERT INTO model_evaluations (
            architecture, split, accuracy, quadratic_weighted_kappa,
            macro_f1, weighted_f1, macro_precision, macro_recall,
            roc_auc, num_samples
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        with self.get_connection() as conn:
            conn.execute(
                query,
                (
                    metrics.get("architecture", "unknown"),
                    metrics.get("split", "validation"),
                    metrics.get("accuracy"),
                    metrics.get("quadratic_weighted_kappa"),
                    metrics.get("macro_f1"),
                    metrics.get("weighted_f1"),
                    metrics.get("macro_precision"),
                    metrics.get("macro_recall"),
                    metrics.get("roc_auc_macro"),
                    metrics.get("num_samples"),
                ),
            )
            conn.commit()

    def get_model_benchmarks(self) -> pd.DataFrame:
        """Retrieve latest evaluation metrics per architecture and split."""
        query = """
        SELECT architecture, split, accuracy, quadratic_weighted_kappa,
               macro_f1, weighted_f1, roc_auc, num_samples, evaluated_at
        FROM model_evaluations
        ORDER BY evaluated_at DESC;
        """
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn)
