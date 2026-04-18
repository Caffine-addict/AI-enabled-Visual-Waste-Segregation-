"""Transfer learning with MobileNetV2 and class weights for balanced training."""

import json
import os
from pathlib import Path
from collections import Counter

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from realwaste_mlops.features import preprocess

# MLflow imports
try:
    import mlflow
    import mlflow.keras

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


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


def load_images(split: str):
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


def build_transfer_model(num_classes: int = 6) -> keras.Model:
    """Build transfer learning model with MobileNetV2."""
    # Load pre-trained MobileNetV2
    base_model = keras.applications.MobileNetV2(
        input_shape=(*preprocess.IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    base_model.trainable = False  # Freeze base initially

    model = keras.Sequential(
        [
            base_model,
            layers.Dropout(0.3),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    return model


def train(epochs: int = 5, batch_size: int = 32, mlflow_tracking_uri: str = None):
    """Train with transfer learning and class weights."""
    print("=" * 50)
    print("TRANSFER LEARNING WITH MOBILENETV2 + CLASS WEIGHTS")
    print("=" * 50)

    # Setup MLflow if available
    if MLFLOW_AVAILABLE and mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment("realwaste-mobilenetv2")
        print(f"\nMLflow tracking enabled: {mlflow_tracking_uri}")
    elif MLFLOW_AVAILABLE:
        # Try local mlruns
        mlflow.set_experiment("realwaste-mobilenetv2")
        print("\nMLflow tracking enabled (local)")

    print("\nLoading data...")
    X_train, y_train = load_images("train")
    X_val, y_val = load_images("val")

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
    print("\nBuilding transfer learning model...")
    model = build_transfer_model()
    model.summary()

    # Compile
    print("\nCompiling model...")
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

    # Stage 1: Train classifier head only
    print("\n=== Stage 1: Train classifier head (3 epochs) ===")
    history1 = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=3,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    print(f"\nStage 1 - Val Accuracy: {history1.history['val_accuracy'][-1]:.4f}")

    # Stage 2: Fine-tune top layers of base model
    print("\n=== Stage 2: Fine-tune (2 epochs) ===")
    base_model = model.layers[0]
    base_model.trainable = True

    # Unfreeze last 30 layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    history2 = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=2,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    print(f"\nStage 2 - Val Accuracy: {history2.history['val_accuracy'][-1]:.4f}")

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
        print("WARNING: Model not predicting all classes!")
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

    # Log to MLflow if available
    if MLFLOW_AVAILABLE:
        with mlflow.start_run() as run:
            # Log parameters
            mlflow.log_params(
                {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "base_model": "MobileNetV2",
                    "image_size": str(preprocess.IMAGE_SIZE),
                }
            )
            mlflow.log_params(
                {f"class_weight_{k}": v for k, v in class_weights.items()}
            )

            # Log metrics
            mlflow.log_metrics(
                {
                    "train_accuracy": float(train_results[1]),
                    "val_accuracy": float(val_results[1]),
                    "train_loss": float(train_results[0]),
                    "val_loss": float(val_results[0]),
                }
            )

            # Log model
            mlflow.keras.log_model(
                model, "model", registered_model_name="realwaste-mobilenetv2"
            )

            print(f"MLflow run ID: {run.info.run_id}")

    # Save metadata
    metadata = {
        "model_name": "realwaste-mobilenetv2-transfer",
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
            "base_model": "MobileNetV2",
            "class_weights": {str(k): v for k, v in class_weights.items()},
        },
        "mlflow_tracking": MLFLOW_AVAILABLE,
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
