"""Centralized configuration for RetinaInsight training, evaluation, and data pipelines."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Config:
    """Project configuration with filesystem paths and model hyperparameters."""

    # Project filesystem layout
    PROJECT_ROOT: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    DATA_DIR: Path = field(init=False)
    METADATA_DIR: Path = field(init=False)
    RAW_DATA_DIR: Path = field(init=False)
    TRAIN_METADATA_PATH: Path = field(init=False)
    VAL_METADATA_PATH: Path = field(init=False)
    TEST_METADATA_PATH: Path = field(init=False)
    MODELS_DIR: Path = field(init=False)
    CHECKPOINTS_DIR: Path = field(init=False)
    SAVED_MODELS_DIR: Path = field(init=False)
    OUTPUTS_DIR: Path = field(init=False)
    FIGURES_DIR: Path = field(init=False)
    REPORTS_DIR: Path = field(init=False)
    PREDICTIONS_DIR: Path = field(init=False)
    DATABASE_DIR: Path = field(init=False)
    DATABASE_PATH: Path = field(init=False)

    # Image properties
    IMAGE_SIZE: tuple[int, int] = (224, 224)
    IMAGE_CHANNELS: int = 3
    NUM_CLASSES: int = 5
    CLASS_NAMES: dict[int, str] = field(default_factory=lambda: {
        0: "No DR",
        1: "Mild",
        2: "Moderate",
        3: "Severe",
        4: "Proliferative DR",
    })

    # Reproducibility & Performance
    RANDOM_SEED: int = 42
    BATCH_SIZE: int = 32
    SHUFFLE_BUFFER_SIZE: int = 1000

    # Two-stage Transfer Learning Hyperparameters
    STAGE_1_EPOCHS: int = 5
    STAGE_2_EPOCHS: int = 10
    STAGE_1_LEARNING_RATE: float = 1e-3
    STAGE_2_LEARNING_RATE: float = 1e-4
    EARLY_STOPPING_PATIENCE: int = 4
    REDUCE_LR_PATIENCE: int = 2
    REDUCE_LR_FACTOR: float = 0.2
    DROPOUT_RATE: float = 0.3

    # Architectures
    SUPPORTED_ARCHITECTURES: tuple[str, ...] = (
        "efficientnetb0",
        "resnet50",
        "mobilenetv2",
    )

    def __post_init__(self) -> None:
        """Derive absolute paths and create required output directories."""
        root = self.PROJECT_ROOT
        object.__setattr__(self, "DATA_DIR", root / "data")
        object.__setattr__(self, "METADATA_DIR", root / "data" / "metadata")
        object.__setattr__(self, "RAW_DATA_DIR", root / "data" / "raw")
        object.__setattr__(self, "TRAIN_METADATA_PATH", root / "data" / "metadata" / "train_metadata.csv")
        object.__setattr__(self, "VAL_METADATA_PATH", root / "data" / "metadata" / "validation_metadata.csv")
        object.__setattr__(self, "TEST_METADATA_PATH", root / "data" / "metadata" / "test_metadata.csv")
        object.__setattr__(self, "MODELS_DIR", root / "models")
        object.__setattr__(self, "CHECKPOINTS_DIR", root / "models" / "checkpoints")
        object.__setattr__(self, "SAVED_MODELS_DIR", root / "models" / "saved")
        object.__setattr__(self, "OUTPUTS_DIR", root / "outputs")
        object.__setattr__(self, "FIGURES_DIR", root / "outputs" / "figures")
        object.__setattr__(self, "REPORTS_DIR", root / "outputs" / "reports")
        object.__setattr__(self, "PREDICTIONS_DIR", root / "outputs" / "predictions")
        object.__setattr__(self, "DATABASE_DIR", root / "data" / "database")
        object.__setattr__(self, "DATABASE_PATH", root / "data" / "database" / "retinainsight.db")

        # Create needed directories safely
        for directory in (
            self.CHECKPOINTS_DIR,
            self.SAVED_MODELS_DIR,
            self.FIGURES_DIR,
            self.REPORTS_DIR,
            self.PREDICTIONS_DIR,
            self.DATABASE_DIR,
        ):
            directory.mkdir(parents=True, exist_ok=True)


config = Config()
