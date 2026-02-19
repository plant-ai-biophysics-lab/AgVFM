"""Visualization for factor analysis results."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from agvfm.config.experiments import FACTOR_AXES


def plot_factor_contributions(
    results: Dict,
    metric: str = "map",
    iou_threshold: float = 0.5,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 7),
) -> plt.Figure:
    """
    Plot per-axis factor contributions (delta from baseline).

    Args:
        results: Results dictionary from factor analysis
        metric: Metric to plot ('map', 'f1', 'precision', 'recall')
        iou_threshold: IoU threshold for metrics
        save_path: Optional path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    results_dict = results.get("results", {})
    baseline_result = results_dict.get("baseline", {})
    
    # If no baseline, try to find the true baseline config (all axes at baseline values)
    if not baseline_result:
        for key, value in results_dict.items():
            components = value.get("components", {})
            # Check if all components are at baseline values
            is_baseline = all(
                (components.get(axis.name, "") == "" or components.get(axis.name, "") == axis.baseline)
                for axis in FACTOR_AXES
            )
            if is_baseline:
                baseline_result = value
                break
        
        # If still no baseline, use first result as reference (with warning)
        if not baseline_result and results_dict:
            import warnings
            warnings.warn("No baseline found, using first result as reference")
            baseline_result = next(iter(results_dict.values()))
    
    if not baseline_result:
        raise ValueError("No results found - cannot generate plot")
    
    baseline_value = (
        baseline_result.get("metrics_by_iou", {})
        .get(f"iou_{iou_threshold}", {})
        .get(metric, 0.0)
    )
    
    # Collect contributions per axis
    # For OFAT, we only include results where ONLY the current axis is changed from baseline
    axis_data = {}
    for axis in FACTOR_AXES:
        axis_name = axis.name
        axis_values = []
        
        for combo_name, result in results_dict.items():
            if combo_name == "baseline":
                continue
            
            components = result.get("components", {})
            if axis_name not in components:
                continue
            
            # Check if this is a valid OFAT result: only current axis changed, all others at baseline
            is_valid_ofat = True
            for other_axis in FACTOR_AXES:
                if other_axis.name == axis_name:
                    continue  # Skip current axis
                other_value = components.get(other_axis.name, "")
                other_baseline = other_axis.baseline if other_axis.baseline is not None else ""
                if other_value != other_baseline:
                    is_valid_ofat = False
                    break
            
            if not is_valid_ofat:
                continue  # Skip this result - it's not a valid OFAT result for this axis
            
            value = components[axis_name]
            metric_value = (
                result.get("metrics_by_iou", {})
                .get(f"iou_{iou_threshold}", {})
                .get(metric, 0.0)
            )
            delta = metric_value - baseline_value
            
            # Include all values, even if they're at baseline (delta = 0)
            # This ensures we show all levels, including those with zero contribution
            axis_values.append({
                "value": value,
                "delta": delta,
                "metric": metric_value,
            })
        
        if axis_values:
            axis_data[axis_name] = axis_values
    
    # Create plot - one bar per axis showing best value, or grouped bars for multiple values
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare data for grouped bar chart
    axes_names = list(axis_data.keys())
    n_axes = len(axes_names)
    
    if n_axes == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return fig
    
    # Get max values per axis to determine bar width
    max_values_per_axis = max(len(axis_data[ax]) for ax in axes_names)
    
    x_pos = np.arange(n_axes)
    width = 0.8 / max_values_per_axis if max_values_per_axis > 0 else 0.8
    
    # One distinct color per axis - makes it clear which axis each bar belongs to
    axis_colors = sns.color_palette("Set2", n_colors=n_axes)
    
    # For legend: collect all unique value-axis pairs
    legend_handles = []
    
    # Plot bars grouped by axis
    for axis_idx, axis_name in enumerate(axes_names):
        values = axis_data[axis_name]
        # Sort values by delta for consistent ordering (best first)
        values = sorted(values, key=lambda x: x["delta"], reverse=True)
        
        base_color = axis_colors[axis_idx]
        
        for val_idx, val_data in enumerate(values):
            offset = (val_idx - len(values) / 2 + 0.5) * width
            
            # Use different shades of the same color for different values within an axis
            # Darker = better performance
            n_shades = len(values)
            shade_idx = val_idx if val_data["delta"] >= 0 else n_shades - val_idx - 1
            color = sns.light_palette(base_color, n_colors=n_shades + 2, reverse=True)[shade_idx + 1]
            
            # Draw bar even if delta is zero (for visibility)
            bar_height = val_data["delta"] if abs(val_data["delta"]) > 0.001 else 0.001
            bar = ax.bar(
                x_pos[axis_idx] + offset,
                bar_height,
                width * 0.9,
                color=color,
                alpha=0.8 if abs(val_data["delta"]) > 0.001 else 0.3,  # Lighter for zero
                edgecolor="white",
                linewidth=0.5,
            )
            
            # Add to legend only for first axis (to avoid duplicates)
            if axis_idx == 0:
                legend_handles.append(bar[0])
            
            # Always add value label, even if delta is zero
            # Don't truncate - show full label, let padding handle overflow
            label = val_data["value"]
            if abs(val_data["delta"]) > 0.005:
                # For positive deltas: attach beginning of label to end of bar (va="bottom", label starts at bar top)
                # For negative deltas: attach end of label to end of bar (va="top", label ends at bar bottom)
                if val_data["delta"] > 0:
                    ax.text(
                        x_pos[axis_idx] + offset,
                        val_data["delta"],  # Position at top of bar
                        label,
                        ha="left",  # Start of label at bar position
                        va="bottom",  # Bottom of text at bar top
                        fontsize=7,
                        rotation=45,
                    )
                else:
                    ax.text(
                        x_pos[axis_idx] + offset,
                        val_data["delta"],  # Position at bottom of bar
                        label,
                        ha="right",  # End of label at bar position
                        va="top",  # Top of text at bar bottom
                        fontsize=7,
                        rotation=45,
                    )
            else:
                # Label for zero or near-zero bars
                ax.text(
                    x_pos[axis_idx] + offset,
                    0.01,  # Slightly above baseline
                    label,
                    ha="left",
                    va="bottom",
                    fontsize=7,
                    rotation=45,
                    style='italic',  # Make zero values visually distinct
                )
    
    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Factor Axis", fontsize=12)
    ax.set_ylabel(f"Δ{metric.upper()} from Baseline", fontsize=12)
    
    # Add baseline value to title
    metric_label = metric.upper() if metric != "map" else "mAP"
    title = f"Factor Contributions ({results.get('model', 'Unknown Model')}, IoU={iou_threshold})"
    title += f"\nBaseline {metric_label} = {baseline_value:.4f}"
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([ax.replace("_", " ").title() for ax in axes_names], rotation=45, ha="right")
    
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    
    # Adjust y-axis limits to give more space for labels
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    ax.set_ylim(y_min - 0.15 * y_range, y_max + 0.2 * y_range)  # More space above and below for labels
    
    # Adjust margins to prevent label cutoff - leave more room on all sides
    plt.tight_layout(rect=[0.08, 0.08, 0.92, 0.92])
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.5)
        print(f"Saved plot to {save_path}")
    
    return fig


def plot_factor_comparison(
    yolo_results: Dict,
    sam3_results: Dict,
    metric: str = "map",
    iou_threshold: float = 0.5,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 6),
) -> plt.Figure:
    """
    Plot side-by-side comparison of factor contributions for YOLO World vs SAM3.

    Args:
        yolo_results: YOLO World factor analysis results
        sam3_results: SAM3 factor analysis results
        metric: Metric to plot ('map', 'f1', 'precision', 'recall')
        iou_threshold: IoU threshold for metrics
        save_path: Optional path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)
    
    # Plot YOLO World
    yolo_fig = plot_factor_contributions(
        yolo_results, metric=metric, iou_threshold=iou_threshold, save_path=None
    )
    yolo_ax = yolo_fig.axes[0]
    ax1_data = yolo_ax.get_children()
    plt.close(yolo_fig)
    
    # Recreate on subplot
    _plot_single_factor_contributions(ax1, yolo_results, metric, iou_threshold, "YOLO World")
    
    # Plot SAM3
    _plot_single_factor_contributions(ax2, sam3_results, metric, iou_threshold, "SAM3")
    
    fig.suptitle(
        f"Factor Contributions Comparison ({metric.upper()}@IoU={iou_threshold})",
        fontsize=16,
        fontweight="bold",
    )
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved comparison plot to {save_path}")
    
    return fig


def _plot_single_factor_contributions(
    ax: plt.Axes,
    results: Dict,
    metric: str,
    iou_threshold: float,
    model_name: str,
    show_legend: bool = True,
):
    """Helper to plot factor contributions on a single axes."""
    results_dict = results.get("results", {})
    baseline_result = results_dict.get("baseline", {})
    
    if not baseline_result:
        return
    
    baseline_value = (
        baseline_result.get("metrics_by_iou", {})
        .get(f"iou_{iou_threshold}", {})
        .get(metric, 0.0)
    )
    
    # Collect contributions per axis
    # For OFAT, we only include results where ONLY the current axis is changed from baseline
    axis_data = {}
    for axis in FACTOR_AXES:
        axis_name = axis.name
        axis_values = []
        
        for combo_name, result in results_dict.items():
            if combo_name == "baseline":
                continue
            
            components = result.get("components", {})
            if axis_name not in components:
                continue
            
            # Check if this is a valid OFAT result: only current axis changed, all others at baseline
            is_valid_ofat = True
            for other_axis in FACTOR_AXES:
                if other_axis.name == axis_name:
                    continue  # Skip current axis
                other_value = components.get(other_axis.name, "")
                other_baseline = other_axis.baseline if other_axis.baseline is not None else ""
                if other_value != other_baseline:
                    is_valid_ofat = False
                    break
            
            if not is_valid_ofat:
                continue  # Skip this result - it's not a valid OFAT result for this axis
            
            value = components[axis_name]
            metric_value = (
                result.get("metrics_by_iou", {})
                .get(f"iou_{iou_threshold}", {})
                .get(metric, 0.0)
            )
            delta = metric_value - baseline_value
            
            # Include all values, even if they're at baseline (delta = 0)
            # This ensures we show all levels, including those with zero contribution
            axis_values.append({
                "value": value,
                "delta": delta,
                "metric": metric_value,
            })
        
        if axis_values:
            # Sort by delta (best first)
            axis_values.sort(key=lambda x: x["delta"], reverse=True)
            axis_data[axis_name] = axis_values
    
    # Create plot
    axes_names = list(axis_data.keys())
    n_axes = len(axes_names)
    
    if n_axes == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return
    
    max_values_per_axis = max(len(axis_data[ax]) for ax in axes_names)
    x_pos = np.arange(n_axes)
    width = 0.8 / max_values_per_axis if max_values_per_axis > 0 else 0.8
    
    # Use same color scheme as individual plots: one color per axis with shades
    axis_colors = sns.color_palette("Set2", n_colors=n_axes)
    
    for axis_idx, axis_name in enumerate(axes_names):
        values = axis_data[axis_name]
        base_color = axis_colors[axis_idx]
        
        for val_idx, val_data in enumerate(values):
            offset = (val_idx - len(values) / 2 + 0.5) * width
            
            # Use different shades of the same color for different values within an axis
            # Darker = better performance
            n_shades = len(values)
            shade_idx = val_idx if val_data["delta"] >= 0 else n_shades - val_idx - 1
            color = sns.light_palette(base_color, n_colors=n_shades + 2, reverse=True)[shade_idx + 1]
            
            # Draw bar even if delta is zero (for visibility)
            bar_height = val_data["delta"] if abs(val_data["delta"]) > 0.001 else 0.001
            bar = ax.bar(
                x_pos[axis_idx] + offset,
                bar_height,
                width * 0.9,
                color=color,
                alpha=0.7 if abs(val_data["delta"]) > 0.001 else 0.3,  # Lighter for zero
                edgecolor="white",
                linewidth=0.5,
            )
            
            # Add value label on bar - show full label
            value_label = val_data["value"]
            
            # Always label, even if delta is zero
            if abs(val_data["delta"]) > 0.005:
                # For positive deltas: attach beginning of label to end of bar (va="bottom", label starts at bar top)
                # For negative deltas: attach end of label to end of bar (va="top", label ends at bar bottom)
                if val_data["delta"] > 0:
                    ax.text(
                        x_pos[axis_idx] + offset,
                        val_data["delta"],  # Position at top of bar
                        value_label,
                        ha="left",  # Start of label at bar position
                        va="bottom",  # Bottom of text at bar top
                        fontsize=7,
                        rotation=45,
                    )
                else:
                    ax.text(
                        x_pos[axis_idx] + offset,
                        val_data["delta"],  # Position at bottom of bar
                        value_label,
                        ha="right",  # End of label at bar position
                        va="top",  # Top of text at bar bottom
                        fontsize=7,
                        rotation=45,
                    )
            elif abs(val_data["delta"]) <= 0.005:
                # Label for zero or near-zero bars
                ax.text(
                    x_pos[axis_idx] + offset,
                    0.01,  # Slightly above baseline
                    value_label,
                    ha="left",
                    va="bottom",
                    fontsize=7,
                    rotation=45,
                    style='italic',  # Make zero values visually distinct
                )
    
    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Factor Axis", fontsize=11)
    ax.set_ylabel(f"Δ{metric.upper()} from Baseline", fontsize=11)
    
    # Add baseline value to title
    metric_label = metric.upper() if metric != "map" else "mAP"
    title = f"{model_name}\nBaseline {metric_label} = {baseline_value:.4f}"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([ax.replace("_", " ").title() for ax in axes_names], rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)
    
    # Adjust y-axis limits to give more space for labels
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    ax.set_ylim(y_min - 0.15 * y_range, y_max + 0.2 * y_range)  # More space above and below for labels


def plot_all_metrics_factor_contributions(
    results: Dict,
    iou_threshold: float = 0.5,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (20, 18),
) -> plt.Figure:
    """
    Plot factor contributions for all metrics (mAP, F1, Precision, Recall) in a 4x1 grid (one row per metric).
    
    Args:
        results: Results dictionary from factor analysis
        iou_threshold: IoU threshold for metrics
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    metrics = [
        ("map", "mAP"),
        ("f1", "F1"),
        ("precision", "Precision"),
        ("recall", "Recall"),
    ]
    
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    
    for idx, (metric_key, metric_label) in enumerate(metrics):
        ax = axes[idx]
        _plot_single_factor_contributions(
            ax,
            results,
            metric_key,
            iou_threshold,
            f"{metric_label} (Δ from Baseline)",
            show_legend=False,  # No legend in multi-metric plot
        )
    
    model_name = results.get("model", "Unknown Model")
    fig.suptitle(
        f"Factor Contributions - All Metrics ({model_name}, IoU={iou_threshold})",
        fontsize=16,
        fontweight="bold",
    )
    
    # Adjust y-axis limits for each subplot to give more space for labels
    for ax in axes:
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.15 * y_range, y_max + 0.2 * y_range)  # More space above and below for labels
    
    # Adjust margins to prevent label cutoff - leave more room on all sides
    plt.tight_layout(rect=[0.08, 0.08, 0.92, 0.92])
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.5)
        print(f"Saved all-metrics plot to {save_path}")
    
    return fig
