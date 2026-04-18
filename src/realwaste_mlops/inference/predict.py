"""Inference module for RealWaste classifier."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union, List

import numpy as np
import tensorflow as tf
from PIL import Image

from realwaste_mlops.features import preprocess


PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "artifacts" / "model"


class Classifier:
    """RealWaste image classifier."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        metadata_path: Optional[Union[str, Path]] = None,
    ):
        """Initialize the classifier.

        Args:
            model_path: path to model.keras (defaults to artifacts/model/)
            metadata_path: path to model_metadata.json
        """
        if model_path is None:
            model_path = MODEL_DIR / "model.keras"
        self.model_path = Path(model_path)

        if metadata_path is None:
            metadata_path = MODEL_DIR / "model_metadata.json"
        self.metadata_path = Path(metadata_path)

        self._model: Optional[tf.keras.Model] = None
        self._class_names: List[str] = []
        self._load()

    def _load(self) -> None:
        """Load model and metadata."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        self._model = tf.keras.models.load_model(self.model_path)

        if self.metadata_path.exists():
            with open(self.metadata_path) as f:
                metadata = json.load(f)
                self._class_names = metadata.get("class_names", preprocess.CLASS_NAMES)
        else:
            self._class_names = preprocess.CLASS_NAMES

    @property
    def model(self) -> tf.keras.Model:
        if self._model is None:
            raise RuntimeError("Model not loaded")
        return self._model

    @property
    def class_names(self) -> List[str]:
        return self._class_names

    def predict(
        self,
        image: Union[Image.Image, np.ndarray, tf.Tensor],
        return_probs: bool = True,
    ) -> dict:
        """Predict the class of an image.

        Args:
            image: PIL Image, numpy array, or file path
            return_probs: whether to return all probabilities

        Returns:
            Dict with predicted_label, predicted_index, confidence, probabilities, class_names
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image)

        processed = preprocess.preprocess_image(image)
        processed = tf.expand_dims(processed, axis=0)

        probs = self.model(processed, training=False)[0].numpy()
        predicted_index = int(np.argmax(probs))
        confidence = float(probs[predicted_index])

        result = {
            "predicted_label": self._class_names[predicted_index],
            "predicted_index": predicted_index,
            "confidence": confidence,
            "class_names": self._class_names,
        }

        if return_probs:
            result["probabilities"] = probs.tolist()

        return result

    def predict_batch(
        self,
        images: List[Union[Image.Image, np.ndarray]],
    ) -> List[dict]:
        """Predict classes for a batch of images.

        Args:
            images: list of PIL Images or numpy arrays

        Returns:
            List of prediction dicts
        """
        processed = tf.stack([preprocess.preprocess_image(img) for img in images])
        probs = self.model(processed, training=False).numpy()

        results = []
        for prob in probs:
            predicted_index = int(np.argmax(prob))
            confidence = float(prob[predicted_index])
            results.append(
                {
                    "predicted_label": self._class_names[predicted_index],
                    "predicted_index": predicted_index,
                    "confidence": confidence,
                    "probabilities": prob.tolist(),
                    "class_names": self._class_names,
                }
            )

        return results


def get_classifier() -> Classifier:
    """Get a classifier instance (convenience function)."""
    return Classifier()
