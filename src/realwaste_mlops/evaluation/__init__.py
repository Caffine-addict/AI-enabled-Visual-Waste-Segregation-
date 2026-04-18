"""Evaluation package for RealWaste classifier."""

from realwaste_mlops.evaluation.evaluate import (
    evaluate_model,
    evaluate_split,
    print_evaluation_report,
)

__all__ = [
    "evaluate_model",
    "evaluate_split",
    "print_evaluation_report",
]
