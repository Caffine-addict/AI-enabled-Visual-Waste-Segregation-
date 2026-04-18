"""Class-balanced training with weighted loss for handling imbalanced dataset."""

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
    """Compute inverse frequency class weights."""
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


def load_images_with_augmentation(split: str):
    """Load images."""
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


def build_balanced_cnn(num_classes: int = 6) -> keras.Model:
    """Build a CNN optimized for class-balanced learning."""
    model = keras.Sequential(
        [
            # Input
            layers.Input(shape=(*preprocess.IMAGE_SIZE, 3)),
            # First conv block
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            # Second conv block
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            # Third conv block
            layers.Conv2D(128, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            # Fourth conv block
            layers.Conv2D(256, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            # Dense layers with stronger regularization
            layers.Dropout(0.5),
            layers.Dense(
                128, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01)
            ),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    return model


def train(epochs: int = 5, batch_size: int = 32):
    """Train with weighted cross-entropy and class weights."""
    print("=" * 50)
    print("CLASS-BALANCED TRAINING WITH WEIGHTED LOSS")
    print("=" * 50)

    print("\nLoading data...")
    X_train, y_train = load_images_with_augmentation("train")
    X_val, y_val = load_images_with_augmentation("val")

    print(f"\nDataset shapes:")
    print(f"  Train: {X_train.shape}, {y_train.shape}")
    print(f"  Val: {X_val.shape}, {y_val.shape}")

    # Show class distribution
    print("\n=== Class Distribution ===")
    train_counts = Counter(y_train)
    for i, name in enumerate(preprocess.CLASS_NAMES):
        print(f"  {name}: {train_counts.get(i, 0)}")

    # Compute class weights
    class_weights = compute_class_weights("train")

    # Build model
    print("\nBuilding model...")
    model = build_balanced_cnn()
    model.summary()

    # Compile with weighted categorical cross-entropy
    print("\nCompiling with Weighted Sparse Categorical Cross-Entropy...")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            patience=3, restore_best_weights=True, monitor="val_accuracy"
        ),
        keras.callbacks.ReduceLROnPlateau(
            patience=2, factor=0.5, min_lr=1e-6, monitor="val_loss"
        ),
        keras.callbacks.ModelCheckpoint(
            MODEL_DIR / "best_model.keras", save_best_only=True, monitor="val_accuracy"
        ),
    ]

    # Train with class weights
    print("\n=== Training (5 epochs) ===")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,  # Key for class balancing!
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate
    print("\n=== Evaluation ===")
    train_results = model.evaluate(X_train, y_train, verbose=0)
    val_results = model.evaluate(X_val, y_val, verbose=0)

    print(f"\nTrain Accuracy: {train_results[1]:.4f}")
    print(f"Val Accuracy: {val_results[1]:.4f}")

    # Prediction distribution - CRITICAL CHECK
    print("\n=== Prediction Distribution (Validation) ===")
    preds = model.predict(X_val, verbose=0)
    pred_labels = np.argmax(preds, axis=1)
    pred_counts = Counter(pred_labels)

    all_classes_predicted = True
    for i, name in enumerate(preprocess.CLASS_NAMES):
        count = pred_counts.get(i, 0)
        print(f"  {name}: {count}")
        if count == 0:
            all_classes_predicted = False

    print(f"\n>>> All classes predicted: {all_classes_predicted}")

    if not all_classes_predicted:
        print(
            "WARNING: Model not predicting all classes! Training may need more epochs."
        )
    else:
        print("SUCCESS: Model can classify all 6 classes!")

    # True distribution for comparison
    print("\n=== True Distribution (Validation) ===")
    true_counts = Counter(y_val)
    for i, name in enumerate(preprocess.CLASS_NAMES):
        print(f"  {name}: {true_counts.get(i, 0)}")

    # Save model
    print("\nSaving model...")
    model.save(MODEL_DIR / "model.keras")

    # Save metadata
    metadata = {
        "model_name": "realwaste-balanced-cnn",
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
            "train_accuracy": float(train_results[1]),
        },
        "training": {
            "epochs": epochs,
            "loss": "WeightedSparseCategoricalCrossentropy",
            "class_weights": {str(k): v for k, v in class_weights.items()},
        },
    }

    with open(MODEL_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'=' * 50}")
    print(f"Training complete!")
    print(f"Val Accuracy: {val_results[1]:.4f}")
    print(f"{'=' * 50}")

    return model


if __name__ == "__main__":
    train(epochs=5, batch_size=32)
