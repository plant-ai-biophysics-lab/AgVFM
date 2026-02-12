"""Evaluation visualizations: metrics summary and P-R curve."""

from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt


def _pr_curve_from_metrics(
    list_gt_xyxy: List[np.ndarray],
    list_pred_xyxy: List[np.ndarray],
    list_pred_conf: List[np.ndarray],
    iou_threshold: float = 0.5,
) -> tuple:
    """Build P-R curve data (recalls, precisions) for plotting."""
    all_confs = []
    all_tp = []
    n_gt_total = 0
    from vlme.evaluation.metrics import match_predictions_to_gt

    for gt_xyxy, pred_xyxy, pred_conf in zip(list_gt_xyxy, list_pred_xyxy, list_pred_conf):
        gt_xyxy = np.asarray(gt_xyxy).reshape(-1, 4)
        pred_xyxy = np.asarray(pred_xyxy).reshape(-1, 4)
        pred_conf = np.asarray(pred_conf).ravel()
        n_gt_total += len(gt_xyxy)
        if len(pred_xyxy) == 0:
            continue
        tp_mask, _ = match_predictions_to_gt(pred_xyxy, pred_conf, gt_xyxy, iou_threshold=iou_threshold)
        for c, tp in zip(pred_conf, tp_mask):
            all_confs.append(float(c))
            all_tp.append(1 if tp else 0)
    all_confs = np.array(all_confs)
    all_tp = np.array(all_tp, dtype=np.int32)
    if len(all_confs) == 0 or n_gt_total == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 0.0])
    order = np.argsort(-all_confs)
    tp_cum = np.cumsum(all_tp[order])
    fp_cum = np.cumsum(1 - all_tp[order])
    precisions = tp_cum / (tp_cum + fp_cum)
    recalls = tp_cum / n_gt_total
    return recalls, precisions


def plot_eval_summary(
    metrics: Dict,
    *,
    show_pr_curve: bool = True,
    iou_threshold: float = 0.5,
    list_gt_xyxy: Optional[List[np.ndarray]] = None,
    list_pred_xyxy: Optional[List[np.ndarray]] = None,
    list_pred_conf: Optional[List[np.ndarray]] = None,
    show: bool = True,
) -> tuple:
    """
    Plot evaluation summary: metrics text and optional P-R curve.

    Args:
        metrics: dict from compute_metrics_at_iou (precision, recall, map, ...).
        show_pr_curve: If True, draw P-R curve (requires list_* if not stored in metrics).
        iou_threshold: IoU used for curve (e.g. 0.5).
        list_gt_xyxy, list_pred_xyxy, list_pred_conf: needed for P-R curve if show_pr_curve and not in metrics.
        show: Whether to call plt.show().

    Returns:
        (fig, axes).
    """
    n_plots = 2 if show_pr_curve else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    # Summary text
    ax = axes[0]
    ax.axis("off")
    p = metrics.get("precision", 0)
    r = metrics.get("recall", 0)
    m = metrics.get("map", 0)
    n_img = metrics.get("n_images", 0)
    n_gt = metrics.get("n_gt_total", 0)
    text = (
        f"mAP@{iou_threshold:.1f} = {m:.4f}\n"
        f"Precision = {p:.4f}\n"
        f"Recall = {r:.4f}\n"
        f"Images = {n_img}  |  GT boxes = {n_gt}\n"
        f"TP = {metrics.get('total_tp', 0)}  FP = {metrics.get('total_fp', 0)}  FN = {metrics.get('total_fn', 0)}"
    )
    ax.text(0.1, 0.5, text, transform=ax.transAxes, fontsize=12, verticalalignment="center", family="monospace")

    if show_pr_curve and list_gt_xyxy is not None and list_pred_xyxy is not None and list_pred_conf is not None:
        recalls, precisions = _pr_curve_from_metrics(
            list_gt_xyxy, list_pred_xyxy, list_pred_conf, iou_threshold=iou_threshold
        )
        ax2 = axes[1]
        ax2.plot(recalls, precisions, "b-", lw=2)
        ax2.set_xlabel("Recall")
        ax2.set_ylabel("Precision")
        ax2.set_title(f"P-R curve (IoU ≥ {iou_threshold})")
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if show:
        plt.show()
    return fig, axes
