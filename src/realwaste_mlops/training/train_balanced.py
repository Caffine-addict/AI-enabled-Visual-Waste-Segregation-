"""Simple and robust training with proper class balancing."""

from __future__ import annotations

import json
import os
from pathlib import Path
from collections import Counter
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from realwaste_mlops.features import preprocess


PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
MANIFESTS_DIR = PROJECT_ROOT / "artifacts" / "manifests"
MODEL_DIR = PROJECT_ROOT / "artifacts" / "model"


def load_split_manifest(split: str) -> dict:
    """Load a split manifest (train/val/test)."""
    manifest_path = MANIFESTS_DIR / f"{split}.json"
    with open(manifest_path) as f:
        return json.load(f)


def compute_class_weights(split: str = "train") -> dict[int, float]:
    """Compute balanced class weights."""
    manifest = load_split_manifest(split)
    class_names = preprocess.CLASS_NAMES
    class_counts = Counter(record["class_name"] for record in manifest["records"])

    idx_counts = {
        class_names.index(name): count for name, count in class_counts.items()
    }
    total = sum(idx_counts.values())
    n_classes = len(class_names)

    # Balanced weight: total / (n_classes * count)
    weights = {}
    for i in range(n_classes):
        count = idx_counts.get(i, 1)
        weights[i] = total / (n_classes * count)

    print(f"Class counts: {idx_counts}")
    print(f"Class weights: {weights}")
    return weights


def create_dataset(
    split: str, batch_size: int = 16, shuffle: bool = True, augment: bool = False
):
    """Create TF Dataset from manifest."""
    manifest = load_split_manifest(split)
    class_names = preprocess.CLASS_NAMES
    class_to_index = {name: i for i, name in enumerate(class_names)}

    image_paths = [r["file_path"] for r in manifest["records"]]
    labels = [class_to_index[r["class_name"]] for r in manifest["records"]]

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    def load_image(path, label):
        image = tf.py_function(
            func=lambda p: preprocess.load_and_preprocess_image(p.numpy()),
            inp=[path],
            Tout=tf.float32,
        )
        image.set_shape((*preprocess.IMAGE_SIZE, 3))
        return image, label

    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle and split == "train":
        dataset = dataset.shuffle(buffer_size=2000)

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def build_model(num_classes: int = 6) -> keras.Model:
    """Build MobileNetV2 model (lighter, faster to train)."""
    # Use MobileNetV2 - faster training, less prone to overfitting
    base_model = keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(*preprocess.IMAGE_SIZE, 3),
        pooling="avg",
    )
    base_model.trainable = False  # Start frozen

    model = keras.Sequential(
        [
            layers.Input(shape=(*preprocess.IMAGE_SIZE, 3)),
            layers.Lambda(lambda x: x / 255.0),
            base_model,
            layers.Dropout(0.3),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    return model


def train(
    epochs: int = 15,
    batch_size: int = 16,
    mlflow_tracking_uri: Optional[str] = None,
) -> keras.Model:
    """Train with proper class balancing."""
    if mlflow_tracking_uri:
        os.environ["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri

    import mlflow

    # Load data
    train_ds = create_dataset("train", batch_size=batch_size, augment=True)
    val_ds = create_dataset("val", batch_size=batch_size, shuffle=False)
    test_ds = create_dataset("test", batch_size=batch_size, shuffle=False)

    # Class weights
    class_weights = compute_class_weights("train")

    # Stage 1: Train classifier head
    print("\n=== Stage 1: Train classifier head ===")
    model = build_model()
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5, min_lr=1e-5),
        ],
    )

    print(f"Stage 1 - Val Accuracy: {history1.history['val_accuracy'][-1]:.4f}")

    # Stage 2: Fine-tune
    print("\n=== Stage 2: Fine-tune ===")
    # Unfreeze top layers
    base_model = model.layers[1]
    base_model.trainable = True
    # Unfreeze last 30 layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-7),
        ],
    )

    # Evaluate on all splits
    print("\n=== Evaluation ===")
    train_results = model.evaluate(train_ds, verbose=0)
    val_results = model.evaluate(val_ds, verbose=0)
    test_results = model.evaluate(test_ds, verbose=0)

    print(f"Train Accuracy: {train_results[1]:.4f}")
    print(f"Val Accuracy: {val_results[1]:.4f}")
    print(f"Test Accuracy: {test_results[1]:.4f}")

    # Get predictions to verify all classes are predicted
    print("\n=== Prediction Distribution ===")
    y_true = []
    y_pred = []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    pred_counts = Counter(y_pred)
    for i, name in enumerate(preprocess.CLASS_NAMES):
        print(f"  {name}: {pred_counts.get(i, 0)} predictions")

    # Save model
    model.save(MODEL_DIR / "model.keras")

    # Save metadata
    metadata = {
        "model_name": "realwaste-mobilenetv2",
        "model_path": "artifacts/model/model.keras",
        "image_size": preprocess.IMAGE_SIZE,
        "class_names": preprocess.CLASS_NAMES,
        "preprocessing": {
            "color_mode": "RGB",
            "resize": list(preprocess.IMAGE_SIZE),
            "dtype": "float32",
        },
        "metrics": {
            "accuracy": float(val_results[1]),
            "test_accuracy": float(test_results[1]),
            "train_accuracy": float(train_results[1]),
        },
        "training_config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "class_weights": True,
            "augmentation": True,
        },
    }

    with open(MODEL_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nModel saved to {MODEL_DIR / 'model.keras'}")
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size)
