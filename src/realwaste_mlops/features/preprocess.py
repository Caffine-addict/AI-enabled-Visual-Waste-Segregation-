"""Preprocessing for EfficientNetV2B2 - shared across training, evaluation, and inference."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, List, Union

import numpy as np
from PIL import Image
import tensorflow as tf


# Contract constants
IMAGE_SIZE = (224, 224)
CLASS_NAMES = [
    "Cardboard",
    "Food Organics",
    "Glass",
    "Metal",
    "Miscellaneous Trash",
    "Paper",
]


def load_class_names(metadata_path: Optional[Union[str, Path]] = None) -> List[str]:
    """Load class names from metadata or return defaults."""
    if metadata_path is None:
        return CLASS_NAMES.copy()
    path = Path(metadata_path)
    if path.exists():
        with open(path) as f:
            data = json.load(f)
            return data.get("class_names", CLASS_NAMES)
    return CLASS_NAMES.copy()


def preprocess_image(image: Union[Image.Image, np.ndarray, tf.Tensor]) -> tf.Tensor:
    """Preprocess an image for the model.

    Args:
        image: PIL Image, numpy array, or TensorFlow tensor
               If PIL/numpy: converted to RGB if needed, resized to 224x224
               If tensor: assumed already preprocessed

    Returns:
        Tensor with shape (224, 224, 3), dtype float32, values in [0, 1]
    """
    if isinstance(image, tf.Tensor):
        return image

    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)

    if not isinstance(image, Image.Image):
        raise TypeError(
            f"Expected PIL Image, numpy array, or tensor, got {type(image)}"
        )

    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize(IMAGE_SIZE, Image.BILINEAR)
    image_array = np.array(image, dtype=np.float32)
    image_array = image_array / 255.0

    return tf.constant(image_array)


def load_and_preprocess_image(image_path: Union[str, Path]) -> tf.Tensor:
    """Load an image from disk and preprocess it.

    Args:
        image_path: path to image file

    Returns:
        Preprocessed tensor with shape (224, 224, 3)
    """
    image = Image.open(image_path)
    return preprocess_image(image)


def create_preprocessing_model() -> tf.keras.Model:
    """Create a preprocessing layer for Keras models.

    Returns:
        Model that takes (224, 224, 3) uint8 and outputs (224, 224, 3) float32
    """
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), dtype=tf.uint8)
    x = tf.cast(inputs, tf.float32) / 255.0
    return tf.keras.Model(inputs, x)
