"""Training module for RealWaste EfficientNetV2B2 classifier."""

from __future__ import annotations

import json
import os
from pathlib import Path
from collections import Counter
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras

from realwaste_mlops.features import preprocess


PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
MANIFESTS_DIR = PROJECT_ROOT / "artifacts" / "manifests"
MODEL_DIR = PROJECT_ROOT / "artifacts" / "model"
REPORTS_DIR = PROJECT_ROOT / "reports"


def load_split_manifest(split: str) -> dict:
    """Load a split manifest (train/val/test)."""
    manifest_path = MANIFESTS_DIR / f"{split}.json"
    with open(manifest_path) as f:
        return json.load(f)


def compute_class_weights(split: str = "train") -> dict[int, float]:
    """Compute class weights for balanced training."""
    manifest = load_split_manifest(split)
    class_names = preprocess.CLASS_NAMES
    class_counts = Counter(record["class_name"] for record in manifest["records"])

    # Convert to indices
    idx_counts = {
        class_names.index(name): count for name, count in class_counts.items()
    }

    total = sum(idx_counts.values())
    n_classes = len(class_names)

    # Inverse frequency weighting
    weights = {}
    for i in range(n_classes):
        if idx_counts.get(i, 0) > 0:
            weights[i] = total / (n_classes * idx_counts[i])
        else:
            weights[i] = 1.0

    return weights


def create_dataset(
    split: str,
    batch_size: int = 32,
    shuffle: bool = True,
    augment: bool = False,
) -> tf.data.Dataset:
    """Create a TF Dataset from a split manifest with optional augmentation.

    Args:
        split: 'train', 'val', or 'test'
        batch_size: batch size
        shuffle: whether to shuffle (only for train)
        augment: whether to apply data augmentation (only for train)

    Returns:
        tf.data.Dataset of (image, label) pairs
    """
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

        if augment and split == "train":
            # Apply augmentation
            image = augment_image(image)

        return image, label

    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle and split == "train":
        dataset = dataset.shuffle(buffer_size=1000)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def augment_image(image: tf.Tensor) -> tf.Tensor:
    """Apply data augmentation to a single image."""
    # Random horizontal flip
    image = tf.image.random_flip_left_right(image)

    # Random vertical flip
    image = tf.image.random_flip_up_down(image)

    # Random rotation (90, 180, 270 degrees)
    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k)

    # Random brightness
    image = tf.image.random_brightness(image, max_delta=0.2)

    # Random contrast
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)

    # Random saturation
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)

    # Clip to valid range
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image


def build_model(
    num_classes: int = 6,
    trainable_base: bool = False,
    dropout_rate: float = 0.5,
) -> keras.Model:
    """Build EfficientNetV2B2 model with custom head.

    Args:
        num_classes: number of output classes
        trainable_base: whether to unfreeze the base model for fine-tuning
        dropout_rate: dropout rate for regularization

    Returns:
        Compiled Keras model
    """
    base_model = keras.applications.EfficientNetV2B2(
        include_top=False,
        weights="imagenet",
        input_shape=(*preprocess.IMAGE_SIZE, 3),
        pooling="avg",
    )
    base_model.trainable = trainable_base

    inputs = keras.Input(shape=(*preprocess.IMAGE_SIZE, 3))
    x = inputs / 255.0
    x = base_model(x)
    x = keras.layers.Dropout(dropout_rate)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    return model


def train(
    epochs: int = 50,
    batch_size: int = 32,
    mlflow_tracking_uri: Optional[str] = None,
    fine_tune: bool = True,
    use_class_weights: bool = True,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """Train the model with MLflow tracking, class weights, and augmentation.

    Args:
        epochs: number of training epochs
        batch_size: batch size
        mlflow_tracking_uri: MLflow tracking URI (defaults to local)
        fine_tune: if True, do 2-stage training (head-only then full fine-tune)
        use_class_weights: if True, apply class weights for imbalanced data
        learning_rate: base learning rate

    Returns:
        Trained model
    """
    if mlflow_tracking_uri:
        os.environ["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri

    import mlflow

    # Custom callback to print less frequently
    class PrintEveryNSteps(keras.callbacks.Callback):
        def __init__(self, n_steps: int = 34):
            super().__init__()
            self.n_steps = n_steps
            self.step_count = 0

        def on_train_batch_begin(self, batch, logs=None):
            self.step_count += 1
            if self.step_count % self.n_steps == 0:
                print(f"Step {self.step_count} - ", end="")

        def on_train_batch_end(self, batch, logs=None):
            if self.step_count % self.n_steps == 0:
                print(
                    f"loss: {logs.get('loss', 0):.4f}, accuracy: {logs.get('accuracy', 0):.4f}"
                )

    mlflow.set_experiment("realwaste-efficientnetv2b2")

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.log_param("model_name", "EfficientNetV2B2")
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("fine_tune", fine_tune)
        mlflow.log_param("class_weights", use_class_weights)
        mlflow.log_param("augmentation", True)

        # Load data
        train_ds = create_dataset("train", batch_size=batch_size, augment=True)
        val_ds = create_dataset("val", batch_size=batch_size, shuffle=False)

        # Compute class weights
        if use_class_weights:
            class_weights = compute_class_weights("train")
            print(f"Class weights: {class_weights}")
        else:
            class_weights = None

        print_callback = PrintEveryNSteps(n_steps=34)

        if fine_tune:
            # Stage 1: Train classification head only with higher dropout
            print("Stage 1: Training classification head with augmentation...")
            model = build_model(trainable_base=False, dropout_rate=0.5)

            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )

            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=max(5, epochs // 10),
                class_weight=class_weights,
                callbacks=[
                    print_callback,
                    keras.callbacks.EarlyStopping(
                        patience=3, restore_best_weights=True
                    ),
                    keras.callbacks.ReduceLROnPlateau(
                        patience=2, factor=0.5, min_lr=1e-6
                    ),
                ],
            )

            # Stage 2: Fine-tune entire model with lower learning rate
            print("Stage 2: Fine-tuning entire model...")
            for layer in model.layers:
                if hasattr(layer, "trainable"):
                    layer.trainable = True

            # Lower learning rate for fine-tuning
            fine_tune_lr = 1e-5

            model.compile(
                optimizer=keras.optimizers.Adam(fine_tune_lr),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )

            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=epochs,
                class_weight=class_weights,
                callbacks=[
                    print_callback,
                    keras.callbacks.EarlyStopping(
                        patience=5, restore_best_weights=True
                    ),
                    keras.callbacks.ReduceLROnPlateau(
                        patience=3, factor=0.5, min_lr=1e-7
                    ),
                ],
            )
        else:
            model = build_model(trainable_base=False, dropout_rate=0.5)

            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )

            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=epochs,
                class_weight=class_weights,
                callbacks=[
                    print_callback,
                    keras.callbacks.EarlyStopping(
                        patience=5, restore_best_weights=True
                    ),
                    keras.callbacks.ReduceLROnPlateau(
                        patience=3, factor=0.5, min_lr=1e-6
                    ),
                ],
            )

        # Log metrics
        for epoch, (loss, val_loss) in enumerate(
            zip(history.history["loss"], history.history["val_loss"])
        ):
            mlflow.log_metrics({"loss": loss, "val_loss": val_loss}, step=epoch)

        val_accuracy = history.history["val_accuracy"][-1]
        mlflow.log_metric("val_accuracy", val_accuracy)

        # Save model
        model.save(MODEL_DIR / "model.keras")
        mlflow.log_artifact(str(MODEL_DIR / "model.keras"))

        # Save metadata
        metadata = {
            "model_name": "realwaste-efficientnetv2b2",
            "model_path": "artifacts/model/model.keras",
            "image_size": preprocess.IMAGE_SIZE,
            "class_names": preprocess.CLASS_NAMES,
            "preprocessing": {
                "color_mode": "RGB",
                "resize": list(preprocess.IMAGE_SIZE),
                "dtype": "float32",
            },
            "metrics": {
                "accuracy": float(val_accuracy),
                "macro_f1": 0.0,
            },
            "mlflow_run_id": run_id,
            "promoted": False,
            "training_config": {
                "epochs": epochs,
                "batch_size": batch_size,
                "class_weights": use_class_weights,
                "augmentation": True,
                "fine_tune": fine_tune,
            },
        }

        with open(MODEL_DIR / "model_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        mlflow.log_artifact(str(MODEL_DIR / "model_metadata.json"))

        return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train RealWaste classifier")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--mlflow-uri", type=str, default=None)
    parser.add_argument("--fine-tune", action="store_true", default=True)
    parser.add_argument("--no-class-weights", action="store_true", default=False)
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        mlflow_tracking_uri=args.mlflow_uri,
        fine_tune=args.fine_tune,
        use_class_weights=not args.no_class_weights,
    )
