"""Training module with focal loss for imbalanced data."""

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
    """Compute class weights for balanced training using effective number of samples."""
    manifest = load_split_manifest(split)
    class_names = preprocess.CLASS_NAMES
    class_counts = Counter(record["class_name"] for record in manifest["records"])

    # Convert to indices
    idx_counts = {
        class_names.index(name): count for name, count in class_counts.items()
    }

    total = sum(idx_counts.values())
    n_classes = len(class_names)

    # Effective number of samples weighting (better for highly imbalanced)
    # https://arxiv.org/abs/1901.05555
    beta = 0.9999
    weights = {}
    effective_counts = {}
    for i in range(n_classes):
        count = idx_counts.get(i, 0)
        effective_counts[i] = (1 - beta**count) / (1 - beta)
        weights[i] = total / (n_classes * effective_counts[i])

    print(f"Class counts: {idx_counts}")
    print(f"Effective counts: {effective_counts}")
    print(f"Class weights: {weights}")

    return weights


class FocalLoss(keras.losses.Loss):
    """Focal Loss for addressing class imbalance - handles sparse labels."""

    def __init__(self, gamma=2.0, alpha=None, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        # y_true is shape (batch,) - sparse integer labels
        # y_pred is shape (batch, num_classes) - softmax probabilities

        # Convert sparse labels to one-hot
        y_true = tf.cast(y_true, tf.int32)
        y_true_one_hot = tf.one_hot(y_true, tf.shape(y_pred)[-1])
        y_true_one_hot = tf.cast(y_true_one_hot, tf.float32)

        # Cross entropy
        ce = -y_true_one_hot * tf.math.log(y_pred + tf.keras.backend.epsilon())

        # Focal weight
        p_t = y_true_one_hot * y_pred + (1 - y_true_one_hot) * (1 - y_pred)
        focal_weight = (1 - p_t) ** self.gamma

        loss = focal_weight * ce
        return tf.reduce_sum(loss, axis=-1)


def create_dataset(
    split: str,
    batch_size: int = 32,
    shuffle: bool = True,
    augment: bool = False,
) -> tf.data.Dataset:
    """Create a TF Dataset from a split manifest with optional augmentation."""
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

    # Random brightness
    image = tf.image.random_brightness(image, max_delta=0.15)

    # Random contrast
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)

    # Clip to valid range
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image


def build_model(
    num_classes: int = 6,
    dropout_rate: float = 0.2,
) -> keras.Model:
    """Build EfficientNetV2B0 (lighter) model with custom head."""
    # Use B0 instead of B2 for faster training and less overfitting
    base_model = keras.applications.EfficientNetV2B0(
        include_top=False,
        weights="imagenet",
        input_shape=(*preprocess.IMAGE_SIZE, 3),
        pooling="avg",
    )
    # Freeze base initially
    base_model.trainable = False

    inputs = keras.Input(shape=(*preprocess.IMAGE_SIZE, 3))
    x = inputs / 255.0
    x = base_model(x)
    x = keras.layers.Dropout(dropout_rate)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    return model


def train(
    epochs: int = 30,
    batch_size: int = 16,
    mlflow_tracking_uri: Optional[str] = None,
    use_focal_loss: bool = True,
    use_class_weights: bool = True,
) -> keras.Model:
    """Train the model with focal loss and class weights."""
    if mlflow_tracking_uri:
        os.environ["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri

    import mlflow

    class PrintEveryNSteps(keras.callbacks.Callback):
        def __init__(self, n_steps: int = 50):
            super().__init__()
            self.n_steps = n_steps
            self.step_count = 0

        def on_train_batch_end(self, batch, logs=None):
            self.step_count += 1
            if self.step_count % self.n_steps == 0:
                print(
                    f"Step {self.step_count} - loss: {logs.get('loss', 0):.4f}, acc: {logs.get('accuracy', 0):.4f}"
                )

    mlflow.set_experiment("realwaste-efficientnetv2b0-focal")

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.log_param("model_name", "EfficientNetV2B0")
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("focal_loss", use_focal_loss)
        mlflow.log_param("class_weights", use_class_weights)
        mlflow.log_param("augmentation", True)

        # Load data
        train_ds = create_dataset("train", batch_size=batch_size, augment=True)
        val_ds = create_dataset("val", batch_size=batch_size, shuffle=False)

        print(f"Train samples: {sum(1 for _ in train_ds.unbatch())}")
        print(f"Val samples: {sum(1 for _ in val_ds.unbatch())}")

        # Compute class weights
        class_weights = compute_class_weights("train") if use_class_weights else None

        # Build model
        model = build_model(dropout_rate=0.3)

        # Use focal loss or weighted categorical crossentropy
        if use_focal_loss:
            loss = FocalLoss(gamma=2.0)
        else:
            loss = keras.losses.SparseCategoricalCrossentropy()

        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss=loss,
            metrics=["accuracy"],
        )

        print_callback = PrintEveryNSteps(n_steps=50)

        # Stage 1: Train classification head only
        print("Stage 1: Training classification head...")
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=min(10, epochs // 3),
            class_weight=class_weights,
            callbacks=[
                print_callback,
                keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5, min_lr=1e-5),
            ],
        )

        # Stage 2: Fine-tune with lower learning rate
        print("Stage 2: Fine-tuning...")
        for layer in model.layers:
            if hasattr(layer, "trainable"):
                layer.trainable = True

        model.compile(
            optimizer=keras.optimizers.Adam(1e-5),
            loss=loss,
            metrics=["accuracy"],
        )

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            class_weight=class_weights,
            callbacks=[
                print_callback,
                keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-7),
            ],
        )

        # Log metrics
        for epoch, (loss, val_loss) in enumerate(
            zip(history.history["loss"], history.history["val_loss"])
        ):
            mlflow.log_metrics({"loss": loss, "val_loss": val_loss}, step=epoch)

        val_accuracy = history.history["val_accuracy"][-1]
        mlflow.log_metric("val_accuracy", val_accuracy)

        # Evaluate on test set
        test_ds = create_dataset("test", batch_size=batch_size, shuffle=False)
        test_results = model.evaluate(test_ds)
        mlflow.log_metric("test_accuracy", test_results[1])

        # Save model
        model.save(MODEL_DIR / "model.keras")
        mlflow.log_artifact(str(MODEL_DIR / "model.keras"))

        # Save metadata
        metadata = {
            "model_name": "realwaste-efficientnetv2b0",
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
                "test_accuracy": float(test_results[1]),
                "macro_f1": 0.0,
            },
            "mlflow_run_id": run_id,
            "promoted": False,
            "training_config": {
                "epochs": epochs,
                "batch_size": batch_size,
                "focal_loss": use_focal_loss,
                "class_weights": use_class_weights,
                "augmentation": True,
            },
        }

        with open(MODEL_DIR / "model_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        mlflow.log_artifact(str(MODEL_DIR / "model_metadata.json"))

        print(
            f"Training complete. Val accuracy: {val_accuracy:.4f}, Test accuracy: {test_results[1]:.4f}"
        )

        return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train RealWaste classifier with focal loss"
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--mlflow-uri", type=str, default=None)
    parser.add_argument("--no-focal-loss", action="store_true", default=False)
    parser.add_argument("--no-class-weights", action="store_true", default=False)
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        mlflow_tracking_uri=args.mlflow_uri,
        use_focal_loss=not args.no_focal_loss,
        use_class_weights=not args.no_class_weights,
    )
