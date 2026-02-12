"""
Detection metrics: IoU, matching, precision, recall, mAP at multiple IoU thresholds.
"""

from typing import List, Tuple

import numpy as np


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute pairwise IoU between boxes a (N,4) and b (M,4) in xyxy format.

    Args:
        a: Array of shape (N, 4) with boxes in xyxy format
        b: Array of shape (M, 4) with boxes in xyxy format

    Returns:
        Array of shape (N, M) with pairwise IoU values
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)

    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    # Compute intersection
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)

    # Compute union
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
    Greedy matching: sort predictions by confidence descending, assign each to best unmatched GT if IoU >= threshold.

    Args:
        pred_xyxy: Array of shape (N, 4) with predicted boxes in xyxy format
        pred_conf: Array of shape (N,) with prediction confidences
        gt_xyxy: Array of shape (M, 4) with ground truth boxes in xyxy format
        iou_threshold: IoU threshold for true positive matching

    Returns:
        Tuple of (tp_mask, gt_matched):
        - tp_mask: Boolean array of length N, True where prediction is a true positive
        - gt_matched: Boolean array of length M, True where GT was matched
    """
    pred_xyxy = np.asarray(pred_xyxy).reshape(-1, 4)
    pred_conf = np.asarray(pred_conf).ravel()
    gt_xyxy = np.asarray(gt_xyxy).reshape(-1, 4)

    n_pred = len(pred_xyxy)
    n_gt = len(gt_xyxy)
    tp_mask = np.zeros(n_pred, dtype=bool)
    gt_matched = np.zeros(n_gt, dtype=bool)

    if n_gt == 0 or n_pred == 0:
        return tp_mask, gt_matched

    # Sort predictions by confidence descending
    order = np.argsort(-pred_conf)
    iou = box_iou(pred_xyxy, gt_xyxy)  # (n_pred, n_gt)

    for i in order:
        j = np.argmax(iou[i])
        if iou[i, j] >= iou_threshold and not gt_matched[j]:
            tp_mask[i] = True
            gt_matched[j] = True

    return tp_mask, gt_matched


def ap_from_pr_curve(
    recalls: np.ndarray,
    precisions: np.ndarray,
) -> float:
    """
    Compute true AP (area under the precision-recall curve) using 101-point
    interpolation (recall thresholds 0.00, 0.01, ..., 1.00).

    This matches the per-IoU AP used inside the COCO evaluation protocol and
    avoids the 1/11 ≈ 0.0909 floor that the legacy 11-point formula produces
    for any prompt with at least one confident true positive.

    Args:
        recalls:    Array of recall values along the sorted-confidence curve.
        precisions: Array of precision values (same length as recalls).

    Returns:
        AP value in [0, 1].
    """
    if len(recalls) == 0 or len(precisions) == 0:
        return 0.0

    recall_points = np.linspace(0, 1, 101)
    ap = 0.0
    for r in recall_points:
        mask = recalls >= r
        ap += float(np.max(precisions[mask])) if np.any(mask) else 0.0

    return ap / 101


def ap_from_pr_curve_11pt(
    recalls: np.ndarray,
    precisions: np.ndarray,
) -> float:
    """
    Legacy 11-point interpolated AP (Pascal VOC style).

    Kept for backwards compatibility / comparison.  Has a hard floor of
    1/11 ≈ 0.0909 for any prompt with ≥ 1 true positive, which makes
    ranking in low-recall regimes unreliable.  Prefer ap_from_pr_curve().
    """
    if len(recalls) == 0 or len(precisions) == 0:
        return 0.0

    ap = 0.0
    for r in np.linspace(0, 1, 11):
        mask = recalls >= r
        ap += float(np.max(precisions[mask])) if np.any(mask) else 0.0

    return ap / 11


def compute_metrics_at_iou(
    list_gt_xyxy: List[np.ndarray],
    list_pred_xyxy: List[np.ndarray],
    list_pred_conf: List[np.ndarray],
    iou_threshold: float = 0.5,
) -> dict:
    """
    Compute precision, recall, mAP, and F1 at a specific IoU threshold.

    Args:
        list_gt_xyxy: Per-image GT boxes, each array of shape (N_i, 4) in xyxy format
        list_pred_xyxy: Per-image pred boxes, each array of shape (M_i, 4) in xyxy format
        list_pred_conf: Per-image pred confidences, each array of shape (M_i,)
        iou_threshold: IoU threshold for true positive matching

    Returns:
        Dictionary with keys: precision, recall, map, f1, total_tp, total_fp, total_fn, n_images, n_gt_total
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

    # mAP@iou: sort all preds by conf, build P-R curve, then 101-point AP
    if len(all_confs) > 0 and n_gt_total > 0:
        order = np.argsort(-all_confs)
        tp_cum = np.cumsum(all_tp[order])
        fp_cum = np.cumsum(1 - all_tp[order])
        precisions_curve = tp_cum / (tp_cum + fp_cum)
        recalls_curve = tp_cum / n_gt_total
        map_val = ap_from_pr_curve(recalls_curve, precisions_curve)
    else:
        map_val = 0.0

    # F1 score
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "map": map_val,
        "f1": f1,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "n_images": len(list_gt_xyxy),
        "n_gt_total": n_gt_total,
        "n_pred_total": len(all_confs),
    }


def compute_f1_max(
    list_gt_xyxy: List[np.ndarray],
    list_pred_xyxy: List[np.ndarray],
    list_pred_conf: List[np.ndarray],
    iou_threshold: float = 0.5,
) -> dict:
    """
    Compute F1-max and the associated precision/recall by sweeping confidence
    thresholds over the full sorted prediction list.

    Unlike ``compute_metrics_at_iou``, which reads precision and recall at a
    single fixed confidence cut-off, this function builds the complete sorted
    TP/FP curve across all images (same as the mAP curve) and returns the
    operating point where F1 is maximised.  The result is therefore
    **threshold-independent**: no specific confidence value needs to be chosen.

    The mAP (101-point interpolated AP) is computed from the same curve at no
    extra cost, so callers can use this as a single entry-point for both
    threshold-free metrics.

    Args:
        list_gt_xyxy:   Per-image GT boxes, each array of shape (N_i, 4) xyxy.
        list_pred_xyxy: Per-image pred boxes, each array of shape (M_i, 4) xyxy.
        list_pred_conf: Per-image pred confidences, each array of shape (M_i,).
        iou_threshold:  IoU threshold for TP matching (default 0.5).

    Returns:
        Dictionary with keys:
          - ``f1_max``      : maximum F1 across the threshold sweep.
          - ``precision``   : precision at the F1-max operating point.
          - ``recall``      : recall at the F1-max operating point.
          - ``map``         : 101-point interpolated AP (same IoU threshold).
          - ``best_conf``   : confidence score at the F1-max operating point
                              (useful for diagnostics / threshold selection).
          - ``n_images``    : number of images evaluated.
          - ``n_gt_total``  : total ground-truth boxes.
          - ``n_pred_total``: total predictions retained (all confidences).
    """
    all_confs: List[float] = []
    all_tp: List[int] = []
    n_gt_total = 0

    for gt_xyxy, pred_xyxy, pred_conf in zip(list_gt_xyxy, list_pred_xyxy, list_pred_conf):
        gt_xyxy = np.asarray(gt_xyxy).reshape(-1, 4)
        pred_xyxy = np.asarray(pred_xyxy).reshape(-1, 4)
        pred_conf = np.asarray(pred_conf).ravel()
        n_gt_total += len(gt_xyxy)

        if len(pred_xyxy) == 0:
            continue

        tp_mask, _ = match_predictions_to_gt(
            pred_xyxy, pred_conf, gt_xyxy, iou_threshold=iou_threshold
        )
        for c, tp in zip(pred_conf, tp_mask):
            all_confs.append(float(c))
            all_tp.append(1 if tp else 0)

    n_pred_total = len(all_confs)

    if n_pred_total == 0 or n_gt_total == 0:
        return {
            "f1_max": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "map": 0.0,
            "best_conf": 0.0,
            "total_tp": 0,
            "total_fp": n_pred_total,
            "total_fn": n_gt_total,
            "n_images": len(list_gt_xyxy),
            "n_gt_total": n_gt_total,
            "n_pred_total": n_pred_total,
        }

    all_confs_arr = np.array(all_confs)
    all_tp_arr = np.array(all_tp, dtype=np.int32)

    # Sort by confidence descending — shared for both F1-max and mAP
    order = np.argsort(-all_confs_arr)
    sorted_confs = all_confs_arr[order]
    tp_cum = np.cumsum(all_tp_arr[order])
    fp_cum = np.cumsum(1 - all_tp_arr[order])

    precisions_curve = tp_cum / (tp_cum + fp_cum + 1e-16)
    recalls_curve = tp_cum / n_gt_total
    f1_curve = (2 * precisions_curve * recalls_curve
                / (precisions_curve + recalls_curve + 1e-16))

    best_idx = int(np.argmax(f1_curve))
    f1_max = float(f1_curve[best_idx])
    prec_at_best = float(precisions_curve[best_idx])
    rec_at_best = float(recalls_curve[best_idx])
    best_conf = float(sorted_confs[best_idx])

    # TP/FP/FN counts at the F1-max operating point (predictions with conf >= best_conf)
    tp_at_best = int(tp_cum[best_idx])
    fp_at_best = int(fp_cum[best_idx])
    fn_at_best = n_gt_total - tp_at_best

    map_val = ap_from_pr_curve(recalls_curve, precisions_curve)

    return {
        "f1_max": f1_max,
        "precision": prec_at_best,
        "recall": rec_at_best,
        "map": map_val,
        "best_conf": best_conf,
        "total_tp": tp_at_best,
        "total_fp": fp_at_best,
        "total_fn": fn_at_best,
        "n_images": len(list_gt_xyxy),
        "n_gt_total": n_gt_total,
        "n_pred_total": n_pred_total,
    }


def compute_map_coco(
    list_gt_xyxy: List[np.ndarray],
    list_pred_xyxy: List[np.ndarray],
    list_pred_conf: List[np.ndarray],
) -> float:
    """
    Compute mAP@0.5:0.95 (COCO-style) by averaging the 101-point AP at IoU
    thresholds 0.50, 0.55, …, 0.95.

    Args:
        list_gt_xyxy: Per-image GT boxes, each array of shape (N_i, 4) in xyxy format
        list_pred_xyxy: Per-image pred boxes, each array of shape (M_i, 4) in xyxy format
        list_pred_conf: Per-image pred confidences, each array of shape (M_i,)

    Returns:
        mAP@0.5:0.95 value
    """
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    map_values = []
    for iou_thresh in iou_thresholds:
        metrics = compute_metrics_at_iou(
            list_gt_xyxy, list_pred_xyxy, list_pred_conf, iou_threshold=iou_thresh
        )
        map_values.append(metrics["map"])

    return np.mean(map_values) if map_values else 0.0
