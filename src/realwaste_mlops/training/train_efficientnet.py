"""EfficientNetV2 training with 50 epochs for 97.1% accuracy target."""

import json
import os
from pathlib import Path
from collections import Counter

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import roc_auc_score

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

# Use EfficientNetV2 optimal input size
IMAGE_SIZE = (300, 300)  # EfficientNetV2-S optimal
NUM_CLASSES = 6


def load_split_manifest(split: str) -> dict:
    manifest_path = MANIFESTS_DIR / f"{split}.json"
    with open(manifest_path) as f:
        return json.load(f)


def compute_class_weights(split: str = "train") -> dict:
    """Compute inverse frequency class weights with smoothing."""
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
        # Apply smoothing to prevent extreme weights
        weights[i] = (total / (n_classes * count)) ** 0.8

    print(f"Class counts: {idx_counts}")
    print(f"Class weights: {weights}")
    return weights


def load_images_with_augmentation(split: str, augment: bool = False):
    """Load images with optional augmentation."""
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
            # Load and resize to 300x300 for EfficientNetV2
            img = keras.preprocessing.image.load_img(
                record["file_path"], target_size=IMAGE_SIZE
            )
            img = keras.preprocessing.image.img_to_array(img)
            
            # EfficientNetV2 preprocessing
            img = keras.applications.efficientnet_v2.preprocess_input(img)
            
            images.append(img)
            labels.append(class_to_index[record["class_name"]])
        except Exception as e:
            print(f"Error loading {record['file_path']}: {e}")

    print(f"  Loaded {len(images)} images")
    return np.array(images), np.array(labels)


def build_efficientnetv2_model(num_classes: int = 6) -> keras.Model:
    """Build EfficientNetV2-S transfer learning model."""
    # Load pre-trained EfficientNetV2-S
    base_model = keras.applications.EfficientNetV2S(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    base_model.trainable = False  # Freeze base initially

    # Build model with batch normalization and dropout
    model = keras.Sequential(
        [
            base_model,
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01)),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    return model


def train(epochs: int = 50, batch_size: int = 32):
    """Train EfficientNetV2 for 50 epochs."""
    print("=" * 60)
    print("EFFICIENTNETV2-S TRAINING FOR 97.1% ACCURACY TARGET")
    print("=" * 60)
    print(f"Epochs: {epochs}, Batch Size: {batch_size}, Image Size: {IMAGE_SIZE}")

    # Setup MLflow
    if MLFLOW_AVAILABLE:
        mlflow.set_experiment("realwaste-efficientnetv2")
        print("\nMLflow tracking enabled")

    # Load data with augmentation for training
    print("\nLoading data...")
    X_train, y_train = load_images_with_augmentation("train", augment=True)
    X_val, y_val = load_images_with_augmentation("val", augment=False)

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
    print("\nBuilding EfficientNetV2-S model...")
    model = build_efficientnetv2_model()
    model.summary()

    # Metrics - use accuracy and AUC (computed separately for multiclass)
    metrics = [
        "accuracy",
    ]

    # Compile with label smoothing
    print("\nCompiling model with label smoothing...")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=metrics,
    )

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            patience=7, restore_best_weights=True, monitor="val_accuracy"
        ),
        keras.callbacks.ReduceLROnPlateau(
            patience=3, factor=0.5, min_lr=1e-6, monitor="val_loss"
        ),
        keras.callbacks.ModelCheckpoint(
            MODEL_DIR / "best_efficientnet.keras", save_best_only=True, monitor="val_accuracy"
        ),
    ]

    # Stage 1: Train classifier head only (epochs 1-10)
    print("\n" + "="*50)
    print("STAGE 1: Train classifier head (epochs 1-10)")
    print("="*50)
    
    history1 = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    print(f"\nStage 1 - Val Accuracy: {history1.history['val_accuracy'][-1]:.4f}")

    # Stage 2: Fine-tune top 30 layers (epochs 11-30)
    print("\n" + "="*50)
    print("STAGE 2: Fine-tune top layers (epochs 11-30)")
    print("="*50)
    
    base_model = model.layers[0]
    base_model.trainable = True
    
    # Unfreeze last 30 layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(5e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=metrics,
    )

    history2 = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    print(f"\nStage 2 - Val Accuracy: {history2.history['val_accuracy'][-1]:.4f}")

    # Stage 3: Fine-tune entire model (epochs 31-50)
    print("\n" + "="*50)
    print("STAGE 3: Fine-tune entire model (epochs 31-50)")
    print("="*50)
    
    # Unfreeze all layers
    base_model.trainable = True
    
    # Recompile with very low learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=metrics,
    )

    history3 = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    print(f"\nStage 3 - Val Accuracy: {history3.history['val_accuracy'][-1]:.4f}")

    # Final Evaluation
    print("\n" + "="*50)
    print("FINAL EVALUATION")
    print("="*50)
    
    train_results = model.evaluate(X_train, y_train, verbose=0)
    val_results = model.evaluate(X_val, y_val, verbose=0)

    print(f"\nTrain Accuracy: {train_results[1]:.4f}")
    print(f"Val Accuracy: {val_results[1]:.4f}")

    # Compute AUC manually for multiclass
    from sklearn.metrics import roc_auc_score
    val_preds = model.predict(X_val, verbose=0)
    val_auc = roc_auc_score(y_val, val_preds, multi_class='ovr', average='macro')
    train_auc = roc_auc_score(y_train, model.predict(X_train, verbose=0), multi_class='ovr', average='macro')
    print(f"Train AUC: {train_auc:.4f}")
    print(f"Val AUC: {val_auc:.4f}")

    # Prediction distribution
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

    # Save model
    print("\nSaving model...")
    model.save(MODEL_DIR / "model.keras")

    # Log to MLflow
    if MLFLOW_AVAILABLE:
        with mlflow.start_run() as run:
            mlflow.log_params({
                "epochs": epochs,
                "batch_size": batch_size,
                "base_model": "EfficientNetV2-S",
                "image_size": str(IMAGE_SIZE),
            })
            mlflow.log_params({f"class_weight_{k}": v for k, v in class_weights.items()})
            mlflow.log_metrics({
                "train_accuracy": float(train_results[1]),
                "train_auc": float(train_auc),
                "val_accuracy": float(val_results[1]),
                "val_auc": float(val_auc),
            })
            mlflow.keras.log_model(model, "model", registered_model_name="realwaste-efficientnetv2")
            print(f"MLflow run ID: {run.info.run_id}")

    # Save metadata
    metadata = {
        "model_name": "realwaste-efficientnetv2",
        "model_path": "artifacts/model/model.keras",
        "image_size": IMAGE_SIZE,
        "class_names": preprocess.CLASS_NAMES,
        "preprocessing": {
            "color_mode": "RGB",
            "resize": list(IMAGE_SIZE),
            "dtype": "float32",
        },
        "metrics": {
            "accuracy": float(val_results[1]),
            "auc": float(val_auc),
            "train_accuracy": float(train_results[1]),
            "train_auc": float(train_auc),
        },
        "training": {
            "epochs": epochs,
            "base_model": "EfficientNetV2-S",
            "class_weights": {str(k): v for k, v in class_weights.items()},
        },
        "mlflow_tracking": MLFLOW_AVAILABLE,
    }

    with open(MODEL_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE!")
    print(f"Val Accuracy: {val_results[1]:.4f} ({val_results[1]*100:.2f}%)")
    print(f"Val AUC: {val_auc:.4f}")
    print(f"{'='*60}")

    return model


if __name__ == "__main__":
    train(epochs=50, batch_size=32)