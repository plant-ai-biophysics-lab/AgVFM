"""Evaluation metrics for detection and counting."""

from agvfm.evaluation.counting import compute_counting_metrics
from agvfm.evaluation.metrics import (
    box_iou,
    compute_map_coco,
    compute_metrics_at_iou,
    match_predictions_to_gt,
)

__all__ = [
    "box_iou",
    "match_predictions_to_gt",
    "compute_metrics_at_iou",
    "compute_map_coco",
    "compute_counting_metrics",
]
