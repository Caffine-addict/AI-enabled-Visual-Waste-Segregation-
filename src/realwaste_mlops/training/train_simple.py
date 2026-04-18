"""Simplified training with better hyperparameters."""

import json
import os
from pathlib import Path
from collections import Counter

import numpy as np
import tensorflow as tf
from tensorflow import keras

from realwaste_mlops.features import preprocess


PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
MANIFESTS_DIR = PROJECT_ROOT / "artifacts" / "manifests"
MODEL_DIR = PROJECT_ROOT / "artifacts" / "model"


def load_split_manifest(split: str) -> dict:
    manifest_path = MANIFESTS_DIR / f"{split}.json"
    with open(manifest_path) as f:
        return json.load(f)


def create_dataset(
    split: str, batch_size: int = 32, augment: bool = False
) -> tf.data.Dataset:
    manifest = load_split_manifest(split)
    class_names = preprocess.CLASS_NAMES
    class_to_index = {name: i for i, name in enumerate(class_names)}

    image_paths = []
    labels = []
    for record in manifest["records"]:
        image_paths.append(record["file_path"])
        labels.append(class_to_index[record["class_name"]])

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    def load_image(path, label):
        image = tf.py_function(
            func=lambda p: preprocess.load_and_preprocess_image(p.numpy()),
            inp=[path],
            Tout=tf.float32,
        )
        image.set_shape((*preprocess.IMAGE_SIZE, 3))

        if augment:
            # Gentle augmentation - only horizontal flip and slight rotation
            image = tf.image.random_flip_left_right(image)

        return image, label

    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    if split == "train":
        dataset = dataset.shuffle(buffer_size=500)

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_model(num_classes: int = 6, dropout: float = 0.3) -> keras.Model:
    """Build a clean EfficientNetV2B2 model."""
    base = keras.applications.EfficientNetV2B2(
        include_top=False,
        weights="imagenet",
        input_shape=(*preprocess.IMAGE_SIZE, 3),
        pooling="avg",
    )

    # Freeze base initially
    base.trainable = False

    inputs = keras.Input(shape=(*preprocess.IMAGE_SIZE, 3))
    x = inputs / 255.0
    x = base(x)
    x = keras.layers.Dropout(dropout)(x)
    outputs = keras.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs)


def train_simple(epochs: int = 30, batch_size: int = 16):
    """Simple, well-tuned training."""

    # Compute class weights - simpler approach
    manifest = load_split_manifest("train")
    class_counts = Counter(r["class_name"] for r in manifest["records"])
    total = sum(class_counts.values())
    n_classes = len(preprocess.CLASS_NAMES)

    class_weights = {}
    for i, name in enumerate(preprocess.CLASS_NAMES):
        count = class_counts.get(name, 1)
        class_weights[i] = total / (n_classes * count)

    print(f"Class weights: {class_weights}")

    # Create datasets
    train_ds = create_dataset("train", batch_size=batch_size, augment=True)
    val_ds = create_dataset("val", batch_size=batch_size)

    print("Phase 1: Train classification head only...")
    model = build_model(dropout=0.4)

    # Phase 1: Train only the classifier head
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-6),
        ],
    )

    best_val_acc = max(history1.history["val_accuracy"])
    print(f"Phase 1 best val accuracy: {best_val_acc:.2%}")

    # Phase 2: Fine-tune entire model with very low learning rate
    print("Phase 2: Fine-tune entire model...")
    model.get_layer(name="efficientnetv2b2").trainable = True

    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),  # Very low LR for fine-tuning
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-7),
        ],
    )

    final_val_acc = max(history2.history["val_accuracy"])
    print(f"Phase 2 best val accuracy: {final_val_acc:.2%}")

    # Save model
    model.save(MODEL_DIR / "model.keras")

    # Save metadata
    metadata = {
        "model_name": "realwaste-efficientnetv2b2",
        "model_path": "artifacts/model/model.keras",
        "image_size": preprocess.IMAGE_SIZE,
        "class_names": preprocess.CLASS_NAMES,
        "metrics": {"accuracy": float(final_val_acc)},
        "training_config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "class_weights": True,
            "augmentation": True,
            "two_stage": True,
        },
    }

    with open(MODEL_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Model saved with val accuracy: {final_val_acc:.2%}")
    return model


if __name__ == "__main__":
    train_simple(epochs=30, batch_size=16)
