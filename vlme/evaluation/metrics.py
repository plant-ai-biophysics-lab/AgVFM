"""
Detection metrics: IoU, matching at IoU threshold, precision, recall, mAP@0.5.
"""

from typing import List, Optional, Tuple

import numpy as np


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute pairwise IoU between boxes a (N,4) and b (M,4) in xyxy format.
    Returns (N, M) IoU matrix.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    union = np.maximum(union, 1e-6)
    return inter / union


def match_predictions_to_gt(
    pred_xyxy: np.ndarray,
    pred_conf: np.ndarray,
    gt_xyxy: np.ndarray,
    iou_threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Greedy match: sort preds by conf descending, assign each to best unmatched GT if IoU >= threshold.

    Returns:
        tp_mask: boolean array len(pred), True where pred is TP.
        gt_matched: boolean array len(gt), True where GT was matched.
    """
    pred_xyxy = np.asarray(pred_xyxy).reshape(-1, 4)
    pred_conf = np.asarray(pred_conf).ravel()
    gt_xyxy = np.asarray(gt_xyxy).reshape(-1, 4)
    n_pred = len(pred_xyxy)
    n_gt = len(gt_xyxy)
    tp_mask = np.zeros(n_pred, dtype=bool)
    gt_matched = np.zeros(n_gt, dtype=bool)
    if n_gt == 0:
        return tp_mask, gt_matched
    if n_pred == 0:
        return tp_mask, gt_matched
    order = np.argsort(-pred_conf)
    iou = box_iou(pred_xyxy, gt_xyxy)  # (n_pred, n_gt)
    for i in order:
        j = np.argmax(iou[i])
        if iou[i, j] >= iou_threshold and not gt_matched[j]:
            tp_mask[i] = True
            gt_matched[j] = True
    return tp_mask, gt_matched


def ap_from_pr_curve(recalls: np.ndarray, precisions: np.ndarray, num_recall_points: int = 11) -> float:
    """
    11-point interpolation: at recall points 0, 0.1, ..., 1, take max precision at R >= r.
    AP = (1/11) * sum of those precisions.
    """
    if len(recalls) == 0 or len(precisions) == 0:
        return 0.0
    recall_points = np.linspace(0, 1, num_recall_points)
    ap = 0.0
    for r in recall_points:
        mask = recalls >= r
        if not np.any(mask):
            p = 0.0
        else:
            p = np.max(precisions[mask])
        ap += p
    return ap / num_recall_points


def compute_metrics_at_iou(
    list_gt_xyxy: List[np.ndarray],
    list_pred_xyxy: List[np.ndarray],
    list_pred_conf: List[np.ndarray],
    iou_threshold: float = 0.5,
) -> dict:
    """
    Compute precision, recall, and mAP@iou_threshold across images.

    Args:
        list_gt_xyxy: Per-image GT boxes (N_i, 4) in xyxy.
        list_pred_xyxy: Per-image pred boxes (M_i, 4) in xyxy.
        list_pred_conf: Per-image pred confidences (M_i,).
        iou_threshold: IoU threshold for TP (default 0.5 for mAP@0.5).

    Returns:
        dict with keys: precision, recall, map (AP at the given IoU), total_tp, total_fp, total_fn, n_images, n_gt_total.
    """
    all_confs: List[float] = []
    all_tp: List[int] = []
    total_fn = 0
    n_gt_total = 0

    for gt_xyxy, pred_xyxy, pred_conf in zip(list_gt_xyxy, list_pred_xyxy, list_pred_conf):
        gt_xyxy = np.asarray(gt_xyxy).reshape(-1, 4)
        pred_xyxy = np.asarray(pred_xyxy).reshape(-1, 4)
        pred_conf = np.asarray(pred_conf).ravel()
        n_gt_total += len(gt_xyxy)
        if len(pred_xyxy) == 0:
            total_fn += len(gt_xyxy)
            continue
        tp_mask, gt_matched = match_predictions_to_gt(
            pred_xyxy, pred_conf, gt_xyxy, iou_threshold=iou_threshold
        )
        total_fn += len(gt_xyxy) - np.sum(gt_matched)
        for c, tp in zip(pred_conf, tp_mask):
            all_confs.append(float(c))
            all_tp.append(1 if tp else 0)

    all_confs = np.array(all_confs)
    all_tp = np.array(all_tp, dtype=np.int32)
    total_tp = int(np.sum(all_tp))
    total_fp = len(all_tp) - total_tp

    # Precision and recall at current threshold (all preds)
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    # mAP@iou: sort all preds by conf, build P-R curve, then 11-point AP
    if len(all_confs) > 0 and n_gt_total > 0:
        order = np.argsort(-all_confs)
        tp_cum = np.cumsum(all_tp[order])
        fp_cum = np.cumsum(1 - all_tp[order])
        precisions_curve = tp_cum / (tp_cum + fp_cum)
        recalls_curve = tp_cum / n_gt_total
        map_val = ap_from_pr_curve(recalls_curve, precisions_curve)
    else:
        map_val = 0.0

    return {
        "precision": precision,
        "recall": recall,
        "map": map_val,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "n_images": len(list_gt_xyxy),
        "n_gt_total": n_gt_total,
    }
