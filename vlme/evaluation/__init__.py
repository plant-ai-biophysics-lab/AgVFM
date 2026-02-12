"""Evaluation: metrics (mAP@0.5, P, R) and run_evaluation pipeline."""

from vlme.evaluation.metrics import compute_metrics_at_iou
from vlme.evaluation.run_eval import run_evaluation

__all__ = ["compute_metrics_at_iou", "run_evaluation"]
