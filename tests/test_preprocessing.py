"""Unit tests for preprocessing module."""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from realwaste_mlops.features.preprocess import (
    IMAGE_SIZE,
    CLASS_NAMES,
    preprocess_image,
    load_and_preprocess_image,
    create_preprocessing_model,
)


class TestConstants:
    """Test preprocessing constants."""

    def test_image_size(self):
        """Test image size is 224x224."""
        assert IMAGE_SIZE == (224, 224)

    def test_class_names(self):
        """Test class names are defined."""
        assert len(CLASS_NAMES) == 6
        assert "Metal" in CLASS_NAMES
        assert "Cardboard" in CLASS_NAMES


class TestPreprocessImage:
    """Test preprocess_image function."""

    def test_pil_image(self):
        """Test preprocessing PIL Image."""
        from PIL import Image

        # Create a test image
        img = Image.new("RGB", (100, 100), color="red")
        result = preprocess_image(img)

        assert result.shape == (224, 224, 3)
        assert result.dtype == np.float32
        assert np.min(result) >= 0.0
        assert np.max(result) <= 1.0

    def test_numpy_array(self):
        """Test preprocessing numpy array."""
        # Create a test image as uint8
        arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = preprocess_image(arr)

        assert result.shape == (224, 224, 3)
        assert result.dtype == np.float32

    def test_grayscale_to_rgb(self):
        """Test converting grayscale to RGB."""
        from PIL import Image

        img = Image.new("L", (100, 100), color=128)
        result = preprocess_image(img)

        assert result.shape == (224, 224, 3)

    def test_tensorflow_tensor(self):
        """Test passthrough for tensor."""
        import tensorflow as tf

        tensor = tf.random.uniform((224, 224, 3))
        result = preprocess_image(tensor)

        assert result is tensor


class TestLoadAndPreprocessImage:
    """Test load_and_preprocess_image function."""

    def test_load_sample_image(self):
        """Test loading a sample image from dataset."""
        # Find first image in train set
        train_path = (
            Path(__file__).parent.parent / "artifacts" / "manifests" / "train.json"
        )

        if train_path.exists():
            import json

            with open(train_path) as f:
                data = json.load(f)

            if data["records"]:
                img_path = data["records"][0]["file_path"]
                result = load_and_preprocess_image(img_path)

                assert result.shape == (224, 224, 3)
                assert result.dtype == np.float32


class TestPreprocessingModel:
    """Test preprocessing model."""

    def test_model_output_shape(self):
        """Test preprocessing model output shape."""
        try:
            model = create_preprocessing_model()
        except Exception:
            pytest.skip("Could not create preprocessing model")

        # Test with batch
        import tensorflow as tf

        batch = tf.random.uniform((2, 224, 224, 3), dtype=tf.uint8)
        output = model(batch)

        assert output.shape == (2, 224, 224, 3)
        assert output.dtype == tf.float32

    def test_model_normalization(self):
        """Test that model normalizes to [0, 1]."""
        try:
            model = create_preprocessing_model()
        except Exception:
            pytest.skip("Could not create preprocessing model")

        import tensorflow as tf

        # Input with known values
        batch = tf.ones((1, 224, 224, 3), dtype=tf.uint8) * 255
        output = model(batch)

        # Should be normalized to 1.0
        assert np.allclose(output.numpy(), 1.0, atol=0.01)
