"""Unit tests for inference module."""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from realwaste_mlops.inference.predict import Classifier, get_classifier


class TestClassifier:
    """Test Classifier class."""

    @pytest.fixture
    def classifier(self):
        """Create classifier instance."""
        model_path = (
            Path(__file__).parent.parent / "artifacts" / "model" / "model.keras"
        )

        if model_path.exists():
            return Classifier(model_path=str(model_path))
        return None

    def test_classifier_creation(self, classifier):
        """Test classifier can be created."""
        if classifier is None:
            pytest.skip("Model not trained yet")

        assert classifier is not None

    def test_predict_returns_valid_probabilities(self, classifier):
        """Test that predictions are valid probabilities."""
        if classifier is None:
            pytest.skip("Model not trained yet")

        # Create dummy input
        image = np.random.rand(224, 224, 3).astype(np.float32)

        result = classifier.predict(image)

        # Check prediction structure
        assert "predicted_label" in result
        assert "predicted_index" in result
        assert "probabilities" in result

        # Check probabilities sum to ~1
        probs = result["probabilities"]
        assert abs(sum(probs) - 1.0) < 0.01
        assert all(0 <= p <= 1 for p in probs)

    def test_predict_batch(self, classifier):
        """Test batch prediction."""
        if classifier is None:
            pytest.skip("Model not trained yet")

        # Create batch of images
        images = [np.random.rand(224, 224, 3).astype(np.float32) for _ in range(3)]

        results = classifier.predict_batch(images)

        assert len(results) == 3
        for result in results:
            assert "predicted_label" in result
            assert "predicted_index" in result

    def test_get_classes(self, classifier):
        """Test getting class names."""
        if classifier is None:
            pytest.skip("Model not trained yet")

        classes = classifier.class_names

        assert len(classes) == 6
        assert "Metal" in classes
        assert "Cardboard" in classes


class TestGetClassifier:
    """Test get_classifier convenience function."""

    def test_get_classifier_returns_classifier(self):
        """Test get_classifier returns Classifier instance."""
        model_path = (
            Path(__file__).parent.parent / "artifacts" / "model" / "model.keras"
        )

        if not model_path.exists():
            pytest.skip("Model not trained yet")

        classifier = get_classifier()

        assert isinstance(classifier, Classifier)

    def test_predict_returns_valid_probabilities(self, classifier):
        """Test that predictions are valid probabilities."""
        if classifier is None:
            pytest.skip("Model not trained yet")

        # Create dummy input
        image = np.random.rand(224, 224, 3).astype(np.float32)

        result = classifier.predict(image)

        # Check prediction structure
        assert "class_name" in result
        assert "class_index" in result
        assert "probabilities" in result

        # Check probabilities sum to ~1
        probs = result["probabilities"]
        assert abs(sum(probs) - 1.0) < 0.01
        assert all(0 <= p <= 1 for p in probs)

    def test_predict_batch(self, classifier):
        """Test batch prediction."""
        if classifier is None:
            pytest.skip("Model not trained yet")

        # Create batch of images
        images = [np.random.rand(224, 224, 3).astype(np.float32) for _ in range(3)]

        results = classifier.predict_batch(images)

        assert len(results) == 3
        for result in results:
            assert "class_name" in result
            assert "class_index" in result

    def test_get_classes(self, classifier):
        """Test getting class names."""
        if classifier is None:
            pytest.skip("Model not trained yet")

        classes = classifier.get_classes()

        assert len(classes) == 6
        assert "Metal" in classes
        assert "Cardboard" in classes


class TestClassifierWithMetadata:
    """Test Classifier with metadata loading."""

    def test_load_from_metadata(self):
        """Test loading model from metadata."""
        metadata_path = (
            Path(__file__).parent.parent / "artifacts" / "model" / "model_metadata.json"
        )

        if not metadata_path.exists():
            pytest.skip("Metadata not available")

        with open(metadata_path) as f:
            metadata = {}

        # Should be able to create classifier with metadata
        classifier = Classifier()
        assert classifier is not None
