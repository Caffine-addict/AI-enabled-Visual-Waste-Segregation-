"""Shared application configuration."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional, Union

import yaml
from pydantic import BaseModel, Field, model_validator


def _resolve_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


class ProjectSettings(BaseModel):
    name: str
    random_seed: int = 42


class PathSettings(BaseModel):
    raw_data_dir: Path
    external_data_dir: Path
    processed_data_dir: Path
    dataset_archive: Path
    manifests_dir: Path
    model_dir: Path
    reports_dir: Path


class DatasetSettings(BaseModel):
    expected_classes: list[str]
    train_ratio: float
    val_ratio: float
    test_ratio: float
    image_size: tuple[int, int]
    batch_size: int

    @model_validator(mode="after")
    def validate_split_ratios(self) -> "DatasetSettings":
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-9:
            raise ValueError("Dataset split ratios must sum to 1.0")
        return self


class TrainingSettings(BaseModel):
    epochs: int
    learning_rate: float
    model_name: str
    model_filename: str
    metadata_filename: str


class EvaluationSettings(BaseModel):
    minimum_accuracy: float = Field(ge=0.0, le=1.0)
    minimum_macro_f1: float = Field(ge=0.0, le=1.0)


class MlflowSettings(BaseModel):
    tracking_uri: str
    experiment_name: str


class ServingSettings(BaseModel):
    api_title: str
    api_version: str
    gradio_path: str


class AppConfig(BaseModel):
    project: ProjectSettings
    paths: PathSettings
    dataset: DatasetSettings
    training: TrainingSettings
    evaluation: EvaluationSettings
    mlflow: MlflowSettings
    serving: ServingSettings

    @property
    def model_path(self) -> Path:
        return self.paths.model_dir / self.training.model_filename

    @property
    def model_metadata_path(self) -> Path:
        return self.paths.model_dir / self.training.metadata_filename


def _to_absolute_paths(config_data: dict, project_root: Path) -> dict:
    path_settings = dict(config_data["paths"])
    absolute_paths = {
        name: project_root / Path(relative_path)
        for name, relative_path in path_settings.items()
    }
    config_data["paths"] = absolute_paths
    return config_data


@lru_cache(maxsize=1)
def load_config(config_path: Optional[Union[str, Path]] = None) -> AppConfig:
    project_root = _resolve_project_root()
    resolved_config_path = (
        Path(config_path) if config_path else project_root / "configs/base.yaml"
    )

    with resolved_config_path.open("r", encoding="utf-8") as config_file:
        config_data = yaml.safe_load(config_file)

    absolute_config = _to_absolute_paths(
        config_data=config_data, project_root=project_root
    )
    return AppConfig.model_validate(absolute_config)
