"""Simple CNN training that's guaranteed to work."""

import json
import os
from pathlib import Path
from collections import Counter

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from realwaste_mlops.features import preprocess


PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
MANIFESTS_DIR = PROJECT_ROOT / "artifacts" / "manifests"
MODEL_DIR = PROJECT_ROOT / "artifacts" / "model"


def load_split_manifest(split: str) -> dict:
    manifest_path = MANIFESTS_DIR / f"{split}.json"
    with open(manifest_path) as f:
        return json.load(f)


def compute_class_weights(split: str = "train") -> dict[int, float]:
    manifest = load_split_manifest(split)
    class_counts = Counter(record["class_name"] for record in manifest["records"])
    class_names = preprocess.CLASS_NAMES

    idx_counts = {
        class_names.index(name): count for name, count in class_counts.items()
    }
    total = sum(idx_counts.values())
    n_classes = len(class_names)

    weights = {}
    for i in range(n_classes):
        count = idx_counts.get(i, 1)
        weights[i] = total / (n_classes * count)

    print(f"Class counts: {idx_counts}")
    print(f"Class weights: {weights}")
    return weights


def load_images_and_labels(split: str):
    """Load all images and labels into memory for faster training."""
    manifest = load_split_manifest(split)
    class_names = preprocess.CLASS_NAMES
    class_to_index = {name: i for i, name in enumerate(class_names)}

    images = []
    labels = []

    print(f"Loading {split} images...")
    for i, record in enumerate(manifest["records"]):
        if i % 200 == 0:
            print(f"  Loaded {i}/{len(manifest['records'])}")
        try:
            img = preprocess.load_and_preprocess_image(record["file_path"])
            images.append(img.numpy())
            labels.append(class_to_index[record["class_name"]])
        except Exception as e:
            print(f"Error loading {record['file_path']}: {e}")

    print(f"  Loaded {len(images)} images")
    return np.array(images), np.array(labels)


def build_simple_cnn(num_classes: int = 6) -> keras.Model:
    """Build a simple CNN that will actually learn."""
    model = keras.Sequential(
        [
            # Input
            layers.Input(shape=(*preprocess.IMAGE_SIZE, 3)),
            # First conv block
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            # Second conv block
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            # Third conv block
            layers.Conv2D(128, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            # Fourth conv block
            layers.Conv2D(256, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            # Dense layers
            layers.Dropout(0.5),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    return model


def train(epochs: int = 30, batch_size: int = 32):
    """Train the simple CNN."""
    print("Loading data into memory...")

    # Load all data
    X_train, y_train = load_images_and_labels("train")
    X_val, y_val = load_images_and_labels("val")
    X_test, y_test = load_images_and_labels("test")

    print(f"\nDataset shapes:")
    print(f"  Train: {X_train.shape}, {y_train.shape}")
    print(f"  Val: {X_val.shape}, {y_val.shape}")
    print(f"  Test: {X_test.shape}, {y_test.shape}")

    # Compute class weights
    class_weights = compute_class_weights("train")

    # Build model
    print("\nBuilding model...")
    model = build_simple_cnn()
    model.summary()

    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-6),
        keras.callbacks.ModelCheckpoint(
            MODEL_DIR / "best_model.keras", save_best_only=True
        ),
    ]

    # Train
    print("\n=== Training ===")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate
    print("\n=== Evaluation ===")
    train_results = model.evaluate(X_train, y_train, verbose=0)
    val_results = model.evaluate(X_val, y_val, verbose=0)
    test_results = model.evaluate(X_test, y_test, verbose=0)

    print(f"\nTrain Accuracy: {train_results[1]:.4f}")
    print(f"Val Accuracy: {val_results[1]:.4f}")
    print(f"Test Accuracy: {test_results[1]:.4f}")

    # Prediction distribution
    print("\n=== Prediction Distribution (Test) ===")
    preds = model.predict(X_test, verbose=0)
    pred_labels = np.argmax(preds, axis=1)
    pred_counts = Counter(pred_labels)
    for i, name in enumerate(preprocess.CLASS_NAMES):
        print(f"  {name}: {pred_counts.get(i, 0)}")

    # True distribution
    print("\n=== True Distribution (Test) ===")
    true_counts = Counter(y_test)
    for i, name in enumerate(preprocess.CLASS_NAMES):
        print(f"  {name}: {true_counts.get(i, 0)}")

    # Save model
    print("\nSaving model...")
    model.save(MODEL_DIR / "model.keras")

    # Save metadata
    metadata = {
        "model_name": "realwaste-simple-cnn",
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
    }

    with open(MODEL_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nModel saved!")
    return model


if __name__ == "__main__":
    train(epochs=30, batch_size=32)
