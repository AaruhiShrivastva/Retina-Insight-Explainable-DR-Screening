"""SQLite schema definitions for RetinaInsight screening history and model benchmarks."""
from __future__ import annotations

CREATE_PREDICTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_uid TEXT UNIQUE NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    image_name TEXT NOT NULL,
    image_path TEXT NOT NULL,
    model_name TEXT NOT NULL,
    predicted_grade INTEGER NOT NULL,
    predicted_severity TEXT NOT NULL,
    confidence REAL NOT NULL,
    prob_grade_0 REAL NOT NULL,
    prob_grade_1 REAL NOT NULL,
    prob_grade_2 REAL NOT NULL,
    prob_grade_3 REAL NOT NULL,
    prob_grade_4 REAL NOT NULL,
    overlay_path TEXT,
    execution_time_ms REAL
);
"""

CREATE_EVALUATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS model_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    architecture TEXT NOT NULL,
    split TEXT NOT NULL,
    accuracy REAL,
    quadratic_weighted_kappa REAL,
    macro_f1 REAL,
    weighted_f1 REAL,
    macro_precision REAL,
    macro_recall REAL,
    roc_auc REAL,
    num_samples INTEGER,
    evaluated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp);
CREATE INDEX IF NOT EXISTS idx_predictions_grade ON predictions(predicted_grade);
CREATE INDEX IF NOT EXISTS idx_predictions_model ON predictions(model_name);
CREATE INDEX IF NOT EXISTS idx_evaluations_arch ON model_evaluations(architecture, split);
"""
