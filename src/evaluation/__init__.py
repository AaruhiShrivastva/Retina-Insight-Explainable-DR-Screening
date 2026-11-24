"""Evaluation package for RetinaInsight."""
from src.evaluation.evaluate import evaluate_model
from src.evaluation.metrics import compute_comprehensive_metrics
from src.evaluation.visualizer import (
    plot_confusion_matrix,
    plot_model_comparison,
    plot_training_history,
)

__all__ = [
    "compute_comprehensive_metrics",
    "evaluate_model",
    "plot_confusion_matrix",
    "plot_model_comparison",
    "plot_training_history",
]
