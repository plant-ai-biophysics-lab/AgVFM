"""
Flower counting metrics: R², RMSE, MAE, MAPE, slope, intercept.
"""

from typing import List

import numpy as np
from scipy import stats


def compute_counting_metrics(
    list_gt_xyxy: List[np.ndarray],
    list_pred_xyxy: List[np.ndarray],
) -> dict:
    """
    Compute counting accuracy metrics: predicted vs actual flower counts.

    Args:
        list_gt_xyxy: Per-image GT boxes, each array of shape (N_i, 4)
        list_pred_xyxy: Per-image pred boxes, each array of shape (M_i, 4)

    Returns:
        Dictionary with keys: r2, rmse, mae, mape, slope, intercept, bias
    """
    # Count flowers per image
    gt_counts = np.array([len(gt) for gt in list_gt_xyxy])
    pred_counts = np.array([len(pred) for pred in list_pred_xyxy])

    # Linear regression: pred = slope * gt + intercept
    if len(gt_counts) > 1 and np.std(gt_counts) > 0:
        slope, intercept, r_value, p_value, std_err = stats.linregress(gt_counts, pred_counts)
        r2 = r_value ** 2
    else:
        slope = 0.0
        intercept = np.mean(pred_counts) if len(pred_counts) > 0 else 0.0
        r2 = 0.0

    # Error metrics
    errors = pred_counts - gt_counts
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    mape = np.mean(np.abs(errors / (gt_counts + 1e-6))) * 100  # Avoid division by zero
    bias = np.mean(errors)

    return {
        "r2": float(r2),
        "rmse": float(rmse),
        "mae": float(mae),
        "mape": float(mape),
        "slope": float(slope),
        "intercept": float(intercept),
        "bias": float(bias),
        "n_images": len(gt_counts),
    }
