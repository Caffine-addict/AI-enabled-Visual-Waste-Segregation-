"""Evaluation module for RealWaste classifier."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union, List, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

from realwaste_mlops.features import preprocess


PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
MANIFESTS_DIR = PROJECT_ROOT / "artifacts" / "manifests"
MODEL_DIR = PROJECT_ROOT / "artifacts" / "model"


def load_split_manifest(split: str) -> dict:
    """Load a split manifest (train/val/test)."""
    manifest_path = MANIFESTS_DIR / f"{split}.json"
    with open(manifest_path) as f:
        return json.load(f)


def load_model_predictions(
    model_path: Union[str, Path],
    split: str,
    batch_size: int = 32,
) -> Tuple[List[int], List[int]]:
    """Load model and generate predictions for a split.

    Args:
        model_path: path to model.keras
        split: 'train', 'val', or 'test'
        batch_size: batch size for prediction

    Returns:
        Tuple of (true_labels, predicted_labels)
    """
    import tensorflow as tf
    from realwaste_mlops.inference.predict import Classifier

    classifier = Classifier(model_path=str(model_path))
    manifest = load_split_manifest(split)

    true_labels = []
    predicted_labels = []

    class_to_index = {name: i for i, name in enumerate(preprocess.CLASS_NAMES)}

    # Process in batches
    batch_images = []
    batch_indices = []

    for record in manifest["records"]:
        try:
            image = preprocess.load_and_preprocess_image(record["file_path"])
            batch_images.append(image)
            batch_indices.append(class_to_index[record["class_name"]])

            if len(batch_images) >= batch_size:
                # Get predictions
                results = classifier.predict_batch(batch_images)
                predicted_labels.extend(r["predicted_index"] for r in results)
                true_labels.extend(batch_indices)
                batch_images = []
                batch_indices = []
        except Exception as e:
            print(f"Error processing {record['file_path']}: {e}")
            continue

    # Process remaining
    if batch_images:
        results = classifier.predict_batch(batch_images)
        predicted_labels.extend(r["predicted_index"] for r in results)
        true_labels.extend(batch_indices)

    return true_labels, predicted_labels


def evaluate_split(
    true_labels: List[int],
    predicted_labels: List[int],
    class_names: Optional[List[str]] = None,
) -> dict[str, Any]:
    """Compute evaluation metrics for a split.

    Args:
        true_labels: ground truth labels
        predicted_labels: model predictions
        class_names: list of class names

    Returns:
        Dictionary with metrics
    """
    if class_names is None:
        class_names = preprocess.CLASS_NAMES

    n_classes = len(class_names)
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    accuracy = accuracy_score(true_labels, predicted_labels)

    # Macro metrics (unweighted average)
    macro_f1 = f1_score(true_labels, predicted_labels, average="macro", zero_division=0)
    macro_precision = precision_score(
        true_labels, predicted_labels, average="macro", zero_division=0
    )
    macro_recall = recall_score(
        true_labels, predicted_labels, average="macro", zero_division=0
    )

    # Weighted metrics
    weighted_f1 = f1_score(
        true_labels, predicted_labels, average="weighted", zero_division=0
    )
    weighted_precision = precision_score(
        true_labels, predicted_labels, average="weighted", zero_division=0
    )
    weighted_recall = recall_score(
        true_labels, predicted_labels, average="weighted", zero_division=0
    )

    # Per-class metrics - use labels parameter to ensure all classes are included
    per_class_f1 = f1_score(
        true_labels,
        predicted_labels,
        average=None,
        labels=range(n_classes),
        zero_division=0,
    )
    per_class_precision = precision_score(
        true_labels,
        predicted_labels,
        average=None,
        labels=range(n_classes),
        zero_division=0,
    )
    per_class_recall = recall_score(
        true_labels,
        predicted_labels,
        average=None,
        labels=range(n_classes),
        zero_division=0,
    )

    # Confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=range(n_classes))

    # Per-class breakdown
    per_class_metrics = {}
    for i, name in enumerate(class_names):
        per_class_metrics[name] = {
            "f1": float(per_class_f1[i]),
            "precision": float(per_class_precision[i]),
            "recall": float(per_class_recall[i]),
            "support": int((true_labels == i).sum()),
        }

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "weighted_f1": float(weighted_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist(),
        "num_samples": len(true_labels),
    }

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "weighted_f1": float(weighted_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist(),
        "num_samples": len(true_labels),
    }


def evaluate_model(
    model_path: Optional[Union[str, Path]] = None,
    split: str = "test",
    output_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Evaluate model on a split and optionally save results.

    Args:
        model_path: path to model.keras (defaults to artifacts/model/)
        split: which split to evaluate on
        output_path: optional path to save results

    Returns:
        Dictionary with evaluation metrics
    """
    if model_path is None:
        model_path = MODEL_DIR / "model.keras"

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    print(f"Loading model from {model_path}")
    print(f"Evaluating on {split} split...")

    true_labels, predicted_labels = load_model_predictions(model_path, split)

    if not true_labels:
        raise ValueError("No predictions generated")

    print(f"Generated {len(true_labels)} predictions")

    results = evaluate_split(true_labels, predicted_labels)

    if output_path is None:
        output_path = MODEL_DIR / f"evaluation_{split}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_path}")

    return results


def print_evaluation_report(results: dict[str, Any]) -> None:
    """Print a formatted evaluation report."""
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)

    print(f"\nOverall Metrics:")
    print(f"  Accuracy:    {results['accuracy']:.4f}")
    print(f"  Macro F1:    {results['macro_f1']:.4f}")
    print(f"  Macro Precision: {results['macro_precision']:.4f}")
    print(f"  Macro Recall:    {results['macro_recall']:.4f}")
    print(f"\n  Weighted F1:     {results['weighted_f1']:.4f}")
    print(f"  Weighted Precision: {results['weighted_precision']:.4f}")
    print(f"  Weighted Recall:    {results['weighted_recall']:.4f}")

    print(f"\nPer-Class Metrics:")
    print("-" * 60)
    print(f"{'Class':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 60)

    for class_name, metrics in results["per_class"].items():
        print(
            f"{class_name:<25} "
            f"{metrics['precision']:>10.4f} "
            f"{metrics['recall']:>10.4f} "
            f"{metrics['f1']:>10.4f} "
            f"{metrics['support']:>10}"
        )

    print("-" * 60)
    print(f"\nTotal samples: {results['num_samples']}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate RealWaste classifier")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument(
        "--split", type=str, default="test", choices=["train", "val", "test"]
    )
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    results = evaluate_model(
        model_path=args.model_path,
        split=args.split,
        output_path=Path(args.output) if args.output else None,
    )

    print_evaluation_report(results)
