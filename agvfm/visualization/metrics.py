"""Visualization for metrics and performance analysis."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_pr_curve(
    results_list: List[Dict],
    labels: Optional[List[str]] = None,
    iou_threshold: float = 0.5,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> plt.Figure:
    """
    Plot Precision-Recall curve with F1 iso-lines.

    Args:
        results_list: List of result dictionaries (from evaluate_prompt)
        labels: Optional list of labels for each result
        iou_threshold: IoU threshold for metrics
        save_path: Optional path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if labels is None:
        labels = [f"Config {i+1}" for i in range(len(results_list))]
    
    # Plot P-R points
    for result, label in zip(results_list, labels):
        metrics = result.get("metrics_by_iou", {}).get(f"iou_{iou_threshold}", {})
        precision = metrics.get("precision", 0.0)
        recall = metrics.get("recall", 0.0)
        
        ax.scatter(recall, precision, s=100, label=label, alpha=0.7)
    
    # Add F1 iso-lines
    f1_levels = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    recall_range = np.linspace(0.01, 0.99, 100)
    
    for f1 in f1_levels:
        # F1 = 2 * (P * R) / (P + R)
        # Solving for P: P = (F1 * R) / (2 * R - F1)
        precision_curve = (f1 * recall_range) / (2 * recall_range - f1)
        # Only plot valid region (P, R in [0, 1])
        valid = (precision_curve >= 0) & (precision_curve <= 1) & (recall_range > 0)
        if np.any(valid):
            ax.plot(
                recall_range[valid],
                precision_curve[valid],
                "--",
                color="gray",
                alpha=0.3,
                linewidth=0.5,
            )
            # Add label at one point
            if f1 in [0.5, 0.7]:
                idx = np.where(valid)[0][len(valid) // 2]
                ax.text(
                    recall_range[idx],
                    precision_curve[idx],
                    f"F1={f1}",
                    fontsize=8,
                    alpha=0.5,
                )
    
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"Precision-Recall Analysis (IoU={iou_threshold})", fontsize=14, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved P-R plot to {save_path}")
    
    return fig


def plot_confidence_sweep(
    sweep_results: Dict,
    metrics: List[str] = None,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot performance metrics vs confidence threshold.

    Args:
        sweep_results: Dictionary with keys as confidence thresholds and values as result dicts
        metrics: List of metrics to plot (default: ['map', 'f1', 'precision', 'recall'])
        save_path: Optional path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    if metrics is None:
        metrics = ["map", "f1", "precision", "recall"]
    
    # Extract data
    thresholds = sorted([float(k) for k in sweep_results.keys()])
    metric_data = {metric: [] for metric in metrics}
    
    for thresh in thresholds:
        result = sweep_results[str(thresh)]
        metrics_dict = result.get("metrics_by_iou", {}).get("iou_0.5", {})
        for metric in metrics:
            metric_data[metric].append(metrics_dict.get(metric, 0.0))
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = sns.color_palette("husl", len(metrics))
    for metric, color in zip(metrics, colors):
        ax.plot(
            thresholds,
            metric_data[metric],
            marker="o",
            label=metric.upper(),
            color=color,
            linewidth=2,
            markersize=4,
        )
    
    ax.set_xlabel("Confidence Threshold", fontsize=12)
    ax.set_ylabel("Metric Value", fontsize=12)
    ax.set_title("Performance vs Confidence Threshold", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim([min(thresholds), max(thresholds)])
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved confidence sweep plot to {save_path}")
    
    return fig


def plot_counting_metrics(
    results_list: List[Dict],
    labels: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 8),
) -> plt.Figure:
    """
    Plot counting metrics comparison (R², RMSE, MAE, slope).

    Args:
        results_list: List of result dictionaries
        labels: Optional list of labels for each result
        save_path: Optional path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    if labels is None:
        labels = [f"Config {i+1}" for i in range(len(results_list))]
    
    # Extract counting metrics
    metrics_to_plot = ["r2", "rmse", "mae", "slope"]
    metric_labels = ["R²", "RMSE", "MAE", "Slope"]
    
    data = {label: [] for label in labels}
    for result, label in zip(results_list, labels):
        counting = result.get("counting", {})
        for metric in metrics_to_plot:
            data[label].append(counting.get(metric, 0.0))
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    x_pos = np.arange(len(labels))
    width = 0.6
    
    colors = sns.color_palette("husl", len(labels))
    
    for idx, (metric, metric_label) in enumerate(zip(metrics_to_plot, metric_labels)):
        ax = axes[idx]
        values = [data[label][idx] for label in labels]
        
        bars = ax.bar(x_pos, values, width, color=colors, alpha=0.7)
        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_title(metric_label, fontsize=12, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    
    fig.suptitle("Counting Metrics Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved counting metrics plot to {save_path}")
    
    return fig
