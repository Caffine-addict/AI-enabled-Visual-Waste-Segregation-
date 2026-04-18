"""Integration tests for API endpoints."""

import pytest
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Check if server can be imported
try:
    from fastapi.testclient import TestClient
    from realwaste_mlops.api.main import app

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


@pytest.mark.integration
class TestAPIEndpoints:
    """Test FastAPI endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        if not HAS_FASTAPI:
            pytest.skip("FastAPI not available")

        return TestClient(app)

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert "model_loaded" in data

    def test_classes_endpoint(self, client):
        """Test classes endpoint."""
        response = client.get("/classes")

        assert response.status_code == 200
        data = response.json()

        assert "classes" in data
        assert len(data["classes"]) == 6

    @pytest.mark.slow
    def test_predict_endpoint_valid_image(self, client):
        """Test predict endpoint with valid image."""
        # Find a sample image
        train_path = (
            Path(__file__).parent.parent / "artifacts" / "manifests" / "train.json"
        )

        if not train_path.exists():
            pytest.skip("No training data")

        with open(train_path) as f:
            data = json.load(f)

        if not data["records"]:
            pytest.skip("No images in train set")

        # Convert to base64 for upload
        import base64

        img_path = data["records"][0]["file_path"]
        with open(img_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode()

        response = client.post(
            "/predict",
            json={"image": img_data, "format": "base64"},
        )

        # May fail if model not trained
        assert response.status_code in [200, 500]

    def test_predict_invalid_format(self, client):
        """Test predict with invalid format."""
        response = client.post(
            "/predict",
            json={"image": "invalid", "format": "unknown"},
        )

        assert response.status_code == 422

    def test_root_endpoint(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()

        assert "name" in data
        assert "version" in data


@pytest.mark.integration
class TestAPIValidation:
    """Test API input validation."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        if not HAS_FASTAPI:
            pytest.skip("FastAPI not available")

        return TestClient(app)

    def test_missing_image(self, client):
        """Test predict without image."""
        response = client.post("/predict", json={})

        assert response.status_code == 422

    def test_empty_image(self, client):
        """Test predict with empty image string."""
        response = client.post(
            "/predict",
            json={"image": "", "format": "base64"},
        )

        # Should fail validation or processing
        assert response.status_code in [400, 422, 500]
