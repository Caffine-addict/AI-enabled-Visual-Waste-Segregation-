"""Unit tests for configuration module."""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from realwaste_mlops.config import load_config


class TestLoadConfig:
    """Test load_config function."""

    def test_load_yaml(self):
        """Test loading YAML configuration."""
        config = load_config()

        assert config is not None
        # Check project settings
        assert config.project.name == "realwaste-mlops"

    def test_config_paths(self):
        """Test that config paths are properly resolved."""
        config = load_config()

        # Paths should be absolute
        assert config.paths.manifests_dir.is_absolute()
        assert config.paths.model_dir.is_absolute()

    def test_dataset_settings(self):
        """Test dataset settings are valid."""
        config = load_config()

        # Check dataset settings
        assert len(config.dataset.expected_classes) == 6
        assert (
            config.dataset.train_ratio
            + config.dataset.val_ratio
            + config.dataset.test_ratio
            == 1.0
        )
        assert config.dataset.image_size == (224, 224)

    def test_training_settings(self):
        """Test training settings are valid."""
        config = load_config()

        # Check training settings
        assert config.training.epochs > 0
        assert config.training.learning_rate > 0
        # Model name can vary
        assert config.training.model_name is not None

    def test_serving_settings(self):
        """Test serving settings are valid."""
        config = load_config()

        # Check serving settings
        assert config.serving.api_title is not None
        assert config.serving.gradio_path is not None
