"""Visualization for factor analysis results."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns

from agvfm.config.experiments import FACTOR_AXES


# ---------------------------------------------------------------------------
# Emoji handling
# ---------------------------------------------------------------------------

# Map each emoji character to a short readable description.
# These match the values defined in FACTOR_AXES emoji axis.
_EMOJI_DESCRIPTIONS = {
    "🌸": "(cherry blossom)",
    "🌺": "(hibiscus)",
    "🌻": "(sunflower)",
    "🌼": "(blossom)",
    "💐": "(bouquet)",
    "🌷": "(tulip)",
}


def _sanitize_label(text: str) -> str:
    """
    Return a matplotlib-safe label string.

    If the text is a single emoji, replace it with its parenthesised
    description (e.g. "🌸" → "(cherry blossom)").  If the text contains
    an emoji as a suffix (e.g. "a yellow flower 🌸"), replace the emoji
    in-place so the rest of the label is preserved.
    Falls back to the original string for any character not in the lookup.
    """
    result = text
    for emoji, desc in _EMOJI_DESCRIPTIONS.items():
        result = result.replace(emoji, desc)
    return result


# ---------------------------------------------------------------------------
# Per-axis empty-value labels
#
# For axes where the baseline / empty string value has a meaningful semantic
# interpretation, define a human-readable label to show instead of the generic
# "(baseline)" fallback.
#
# Rules:
#   grammar   → "" means no article prefix, i.e. bare-noun form
#   phenology → "" means no phenology modifier; in context the word being
#               described is simply "flower" (the noun itself)
#   All other axes → fall back to "(baseline)"
# ---------------------------------------------------------------------------
_AXIS_EMPTY_LABELS: Dict[str, str] = {
    "grammar":   "(bare noun)",
}


def _axis_value_label(axis_name: str, value: str) -> str:
    """Return the display label for a single axis value.

    Handles:
    * Non-empty values → ``_sanitize_label(value)`` (emoji substitution).
    * Empty string (``""``) → axis-specific override from
      ``_AXIS_EMPTY_LABELS`` if one exists, otherwise ``"(baseline)"``.
    """
    if value != "":
        return _sanitize_label(value)
    return _AXIS_EMPTY_LABELS.get(axis_name, "(baseline)")


# ---------------------------------------------------------------------------
# Spoke-label abbreviation table
#
# For spider/radar charts the spoke labels must fit in a small cell.  Any
# label longer than 10 characters is replaced with a shorter form.  The table
# maps the *full display label* (after emoji substitution / empty handling) to
# its abbreviated version.  Unknown labels are left as-is.
# ---------------------------------------------------------------------------
_SPOKE_ABBREVS: Dict[str, str] = {
    # taxonomy axis
    "vigna unguiculata flower": "V. unguiculata",
    "black-eyed pea flower":    "blk-eyd pea",
    "cowpea flower":            "cowpea flwr",
    "legume flower":            "legume flwr",
    "crop flower":              "crop flwr",
    "bean flower":              "bean flwr",
    "pea flower":               "pea flwr",
    # negation axis — these are the longest labels in the whole set
    "not a bud, not the green calyx, not a leaf": "n bud/calyx/leaf",
    "not a bud, not the green calyx":             "n bud/calyx",
    "not the green calyx":                        "n calyx",
    "not a leaf":                                 "n leaf",
    "not a bud":                                  "n bud",
    # anatomy axis
    "with open petals":        "w/ open petals",
    "with visible petals":     "w/ vis. petals",
    "with petals and stamens": "w/ pet+stamen",
    # grammar axis
    "a photo of a":  "photo of a",
    "close-up of a": "close-up of",
    # emoji descriptions (produced by _sanitize_label)
    "(cherry blossom)": "(cherry bl.)",
    "(hibiscus)":       "(hibiscus)",   # already short
    "(sunflower)":      "(sunflower)",  # already short
    "(blossom)":        "(blossom)",    # already short
    "(bouquet)":        "(bouquet)",    # already short
    "(tulip)":          "(tulip)",      # already short
}


def _abbreviate_spoke_label(label: str) -> str:
    """Return a shortened form of *label* suitable for spider spoke annotation.

    If the label appears in ``_SPOKE_ABBREVS``, return that abbreviation.
    Otherwise return the label unchanged (it is assumed to be short enough).
    """
    return _SPOKE_ABBREVS.get(label, label)



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
                "value": _axis_value_label(axis_name, value),
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
                "value": _axis_value_label(axis_name, value),
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




# ---------------------------------------------------------------------------
# Radar / spider chart — metric shifts from baseline
# ---------------------------------------------------------------------------

# Metric display config: (key, display label, color)
_RADAR_METRICS: List[Tuple[str, str, str]] = [
    ("map",       "mAP",       "#E63946"),   # red
    ("f1",        "F1",        "#2196F3"),   # blue
    ("precision", "Precision", "#4CAF50"),   # green
    ("recall",    "Recall",    "#FF9800"),   # orange
]

# Factor axis colour palette — one colour per axis polygon.
# Generated at call time via seaborn so the count matches active_axes.
def _axis_palette(n: int) -> List:
    return sns.color_palette("tab10", n_colors=n)


def _collect_best_delta_per_metric(
    results_dict: Dict,
    axis_name: str,
    iou_key: str,
    baseline_metrics: Dict[str, float],
) -> Dict[str, float]:
    """
    For a given factor axis, find the single OFAT variant that produces
    the largest *absolute* Δ from baseline for each metric.

    Returns ``{metric_key: best_signed_delta}``.
    """
    best: Dict[str, float] = {m_key: 0.0 for m_key, _, _ in _RADAR_METRICS}

    for combo_name, result in results_dict.items():
        if combo_name == "baseline":
            continue

        components = result.get("components", {})

        # Identify which single axis is varied in this OFAT entry
        varied_axes = []
        for ax in FACTOR_AXES:
            val = components.get(ax.name, "")
            baseline_val = ax.baseline if ax.baseline is not None else ""
            if val != baseline_val:
                varied_axes.append(ax.name)

        if len(varied_axes) != 1 or varied_axes[0] != axis_name:
            continue  # Not an OFAT entry for this axis

        for m_key, _, _ in _RADAR_METRICS:
            metric_value = (
                result.get("metrics_by_iou", {})
                .get(iou_key, {})
                .get(m_key, 0.0)
            )
            delta = metric_value - baseline_metrics[m_key]
            if abs(delta) > abs(best[m_key]):
                best[m_key] = delta

    return best


def plot_metric_shift_radar(
    results: Dict,
    iou_threshold: float = 0.5,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (8, 8),
    exclude_empty_axes: bool = True,
) -> plt.Figure:
    """Radar chart: spokes = metrics, polygons = factor axes.

    Each spoke is one metric (mAP, F1, Precision, Recall).  Each coloured
    polygon is one factor axis.  The radius of each vertex is the **best
    signed Δ from baseline** achievable by any single OFAT variant of that
    axis on that metric.

    Rings are drawn at actual (unnormalised) Δ values so the scale is
    directly interpretable.  The baseline ring (Δ = 0) is drawn as a solid
    black circle.  Negative deltas (worse than baseline) pull the polygon
    inward past the centre.

    Args:
        results:              Results dict from a factor analysis JSON file.
        iou_threshold:        IoU threshold used to extract metric values.
        save_path:            If given, save the figure here (PNG, 300 dpi).
        figsize:              Figure size.
        exclude_empty_axes:   Drop axes where every metric Δ = 0 (no data).

    Returns:
        matplotlib ``Figure``.
    """
    results_dict = results.get("results", {})
    iou_key = f"iou_{iou_threshold}"
    model_name = results.get("model", "Unknown Model")

    # ------------------------------------------------------------------ baseline
    baseline_entry = results_dict.get("baseline", {})
    if not baseline_entry:
        for result in results_dict.values():
            components = result.get("components", {})
            if all(
                components.get(ax.name, "") in ("", ax.baseline or "")
                for ax in FACTOR_AXES
            ):
                baseline_entry = result
                break

    if not baseline_entry:
        raise ValueError("No baseline entry found — cannot draw radar.")

    baseline_metrics = {
        m_key: baseline_entry.get("metrics_by_iou", {}).get(iou_key, {}).get(m_key, 0.0)
        for m_key, _, _ in _RADAR_METRICS
    }

    # ------------------------------------------------------------------ collect Δ
    # {axis_name: {metric_key: best_delta}}
    axis_deltas: Dict[str, Dict[str, float]] = {}
    for ax in FACTOR_AXES:
        d = _collect_best_delta_per_metric(
            results_dict, ax.name, iou_key, baseline_metrics
        )
        axis_deltas[ax.name] = d

    # ------------------------------------------------------------------ active axes
    active_axes = list(FACTOR_AXES)
    if exclude_empty_axes:
        active_axes = [
            ax for ax in FACTOR_AXES
            if any(abs(axis_deltas[ax.name][m_key]) > 1e-6 for m_key, _, _ in _RADAR_METRICS)
        ]

    if not active_axes:
        fig, ax_empty = plt.subplots(figsize=figsize)
        ax_empty.text(0.5, 0.5, "No OFAT data", ha="center", va="center",
                      transform=ax_empty.transAxes)
        return fig

    # ------------------------------------------------------------------ scale
    # Collect all deltas to determine ring spacing and radial limits.
    all_deltas = [
        axis_deltas[ax.name][m_key]
        for ax in active_axes
        for m_key, _, _ in _RADAR_METRICS
    ]
    d_max = max(all_deltas) if all_deltas else 0.0
    d_min = min(all_deltas) if all_deltas else 0.0

    # Choose a clean ring step: target ~4 positive rings above baseline.
    # Round up to a "nice" number (0.01, 0.02, 0.05, 0.10, …).
    raw_step = max(abs(d_max), abs(d_min)) / 4.0
    for step in [0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.50]:
        if step >= raw_step:
            ring_step = step
            break
    else:
        ring_step = round(raw_step, 2) or 0.05

    # Build rings: symmetric around 0 out to cover both extremes.
    n_rings_pos = int(np.ceil(d_max / ring_step)) if d_max > 1e-9 else 1
    n_rings_neg = int(np.ceil(abs(d_min) / ring_step)) if d_min < -1e-9 else 0
    ring_deltas = (
        [-ring_step * i for i in range(n_rings_neg, 0, -1)]
        + [0.0]
        + [ring_step * i for i in range(1, n_rings_pos + 1)]
    )

    # Map Δ → radius: shift so that Δ=0 sits at a fixed inner radius (r0).
    # This lets negative polygons extend toward the centre without going below 0.
    r0 = n_rings_neg * ring_step + ring_step * 0.8   # baseline ring radius
    def _r(delta: float) -> float:
        return r0 + delta

    r_max = r0 + n_rings_pos * ring_step + ring_step * 0.3   # plot ceiling

    # ------------------------------------------------------------------ angles
    n_metrics = len(_RADAR_METRICS)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    # ------------------------------------------------------------------ figure
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, r_max)
    ax.set_yticks([])   # we draw our own rings

    # ---- reference rings -------------------------------------------------
    theta_full = np.linspace(0, 2 * np.pi, 300)
    for d_ring in ring_deltas:
        r_ring = _r(d_ring)
        if r_ring < 0:
            continue
        is_baseline = abs(d_ring) < 1e-9
        ax.plot(
            theta_full, [r_ring] * 300,
            color="black" if is_baseline else "gray",
            linewidth=1.2 if is_baseline else 0.5,
            linestyle="-" if is_baseline else "--",
            alpha=0.5 if is_baseline else 0.35,
            zorder=1,
        )
        # Label ring on the first spoke (top)
        sign = "+" if d_ring > 0 else ("" if d_ring == 0 else "")
        lbl = "baseline (Δ=0)" if is_baseline else f"Δ={sign}{d_ring:.3f}"
        ax.text(
            angles[0], r_ring + r_max * 0.01, lbl,
            ha="center", va="bottom",
            fontsize=6.5, color="black" if is_baseline else "gray",
            fontweight="bold" if is_baseline else "normal",
        )

    # ---- factor axis polygons --------------------------------------------
    colors = _axis_palette(len(active_axes))
    for idx, fax in enumerate(active_axes):
        color = colors[idx]
        label = fax.name.replace("_", " ").title()

        radii = [_r(axis_deltas[fax.name][m_key]) for m_key, _, _ in _RADAR_METRICS]
        # Clamp to >= 0 so polygon doesn't go below the plot floor
        radii_clamped = [max(0.0, rv) for rv in radii]
        radii_closed = radii_clamped + radii_clamped[:1]

        ax.plot(
            angles_closed, radii_closed,
            color=color, linewidth=1.8,
            label=label, zorder=3,
        )
        ax.fill(
            angles_closed, radii_closed,
            color=color, alpha=0.13, zorder=2,
        )
        # Dots at each spoke
        for ang, rv in zip(angles, radii_clamped):
            ax.plot(ang, rv, "o", color=color, markersize=4, zorder=5)

    # ---- spoke labels (metric names) -------------------------------------
    ax.set_xticks(angles)
    metric_labels = [m_label for _, m_label, _ in _RADAR_METRICS]
    ax.set_xticklabels(metric_labels, fontsize=11, fontweight="bold")

    # ---- title & legend --------------------------------------------------
    ax.set_title(
        f"Metric Shifts from Baseline\n{model_name}  (IoU={iou_threshold})",
        fontsize=13,
        fontweight="bold",
        pad=22,
    )
    ax.legend(
        title="Factor Axis",
        title_fontsize=9,
        loc="upper right",
        bbox_to_anchor=(1.40, 1.15),
        fontsize=8,
        framealpha=0.85,
    )

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.3)
        print(f"Saved metric-shift radar to {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Per-axis spider charts overlaying multiple models
# ---------------------------------------------------------------------------

def _nice_radial_ticks(rmin: float, rmax: float, n_rings: int = 3) -> List[float]:
    """Return up to *n_rings* clean tick values on each side of zero.

    Tick values are rounded to the nearest "nice" step chosen from
    ``[0.05, 0.10, 0.15, 0.20, 0.25, 0.50]`` so that the displayed numbers
    are always multiples of 0.05 or 0.10 rather than arbitrary fractions.

    The step is the smallest value in the nice-step list such that
    ``step * n_rings`` covers the full positive (or negative) range.
    """
    _NICE_STEPS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.50, 1.0]
    chosen: List[float] = []

    for sign, bound in [(1, rmax), (-1, abs(rmin))]:
        if bound < 1e-6:
            continue
        # Pick smallest step so that n_rings steps cover the range
        step = _NICE_STEPS[-1]
        for s in _NICE_STEPS:
            if s * n_rings >= bound:
                step = s
                break
        # Emit ticks at 1×step, 2×step, … up to n_rings (skip if > bound+ε)
        for i in range(1, n_rings + 1):
            v = round(sign * s * i, 6)
            if sign > 0 and v > rmax + 1e-9:
                break
            if sign < 0 and v < rmin - 1e-9:
                break
            chosen.append(v)

    return sorted(chosen)


def _find_baseline_entry(results_dict: Dict) -> Optional[Dict]:
    baseline_entry = results_dict.get("baseline", {})
    if baseline_entry:
        return baseline_entry

    for result in results_dict.values():
        components = result.get("components", {})
        if all(
            components.get(ax.name, "") in ("", ax.baseline or "")
            for ax in FACTOR_AXES
        ):
            return result
    return None


def _get_ofat_variant_entry(results_dict: Dict, axis_name: str, variant_value: str) -> Optional[Dict]:
    """Find an OFAT entry that changes only axis_name to variant_value.

    Returns the result dict or None if not found.
    """
    for combo_name, result in results_dict.items():
        if combo_name == "baseline":
            continue
        components = result.get("components", {})
        # must have the axis equal to the variant
        if components.get(axis_name, None) != variant_value:
            continue

        # ensure no other axis deviates from its baseline
        valid = True
        for ax in FACTOR_AXES:
            if ax.name == axis_name:
                continue
            other_val = components.get(ax.name, "")
            other_baseline = ax.baseline if ax.baseline is not None else ""
            if other_val != other_baseline:
                valid = False
                break
        if valid:
            return result
    return None


def plot_axis_spider_comparison(
    models_results: Dict[str, Dict],
    axis_name: str,
    metrics: List[str] = ["map", "f1"],
    iou_threshold: float = 0.5,
    save_dir: Optional[Path] = None,
    figsize: Tuple[int, int] = (9, 9),
) -> List[plt.Figure]:
    """Create spider/radar charts for a single factor axis comparing multiple models.

    Args:
        models_results: mapping of model_label -> results dict (as loaded from JSON)
        axis_name:       the FACTOR_AXIS.name to visualise
        metrics:         list of metric keys to plot (each metric produces one chart)
        iou_threshold:   IoU key to use
        save_dir:        directory to save PNGs
        figsize:         figure size for each chart

    Returns:
        list of matplotlib Figures (one per metric)
    """
    # Build canonical variant list from FACTOR_AXES ordering
    axis_obj = next((ax for ax in FACTOR_AXES if ax.name == axis_name), None)
    if axis_obj is None:
        raise ValueError(f"Unknown axis: {axis_name}")

    variants = axis_obj.values
    # Sanitise labels for display — use axis-aware empty-value labels
    labels = [_axis_value_label(axis_name, v) for v in variants]

    figs = []
    iou_key = f"iou_{iou_threshold}"

    # Precompute per-model baseline metrics and per-variant metric values
    model_baselines = {}
    model_variant_metrics = {}
    for model_label, res in models_results.items():
        res_dict = res.get("results", {})
        baseline_entry = _find_baseline_entry(res_dict)
        if not baseline_entry:
            # fallback: zero baseline
            baseline_metrics = {m: 0.0 for m in metrics}
        else:
            baseline_metrics = {m: baseline_entry.get("metrics_by_iou", {}).get(iou_key, {}).get(m, 0.0) for m in metrics}
        model_baselines[model_label] = baseline_metrics

        # For each variant find OFAT entry and record absolute metric value
        variant_vals = {m: [] for m in metrics}
        for v in variants:
            entry = _get_ofat_variant_entry(res_dict, axis_name, v)
            if entry:
                for m in metrics:
                    val = entry.get("metrics_by_iou", {}).get(iou_key, {}).get(m, baseline_metrics.get(m, 0.0))
                    variant_vals[m].append(val)
            else:
                # missing -> use baseline (delta 0)
                for m in metrics:
                    variant_vals[m].append(baseline_metrics.get(m, 0.0))

        model_variant_metrics[model_label] = variant_vals

    # Plot one radar per requested metric
    for m in metrics:
        values_per_model = {}
        for model_label in models_results.keys():
            baseline_val = model_baselines[model_label].get(m, 0.0)
            abs_vals = model_variant_metrics[model_label][m]
            # convert to delta from baseline
            deltas = [v - baseline_val for v in abs_vals]
            # close the polygon
            deltas_closed = deltas + deltas[:1]
            values_per_model[model_label] = deltas_closed

        # angles
        n = len(variants)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        angles_closed = angles + angles[:1]

        fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        # radial scale: gather all deltas to set a symmetric radius
        all_deltas = [d for vals in values_per_model.values() for d in vals]
        d_max = max(all_deltas) if all_deltas else 0.0
        d_min = min(all_deltas) if all_deltas else 0.0
        pad = max(0.02, 0.1 * max(abs(d_max), abs(d_min)))
        rmax = d_max + pad
        rmin = d_min - pad
        # ensure some positive range
        if abs(rmax - rmin) < 1e-6:
            rmax += 0.05
            rmin -= 0.05
        ax.set_ylim(rmin, rmax)

        import textwrap as _textwrap
        # Abbreviate long labels first, then wrap what remains
        abbrev_labels = [_abbreviate_spoke_label(lbl) for lbl in labels]
        wrapped_labels = [_textwrap.fill(lbl, width=16) for lbl in abbrev_labels]

        palette = sns.color_palette("tab10", n_colors=len(values_per_model))
        for idx, (model_label, vals) in enumerate(values_per_model.items()):
            color = palette[idx % len(palette)]
            ax.plot(angles_closed, vals, label=model_label, color=color, linewidth=2)
            ax.fill(angles_closed, vals, color=color, alpha=0.12)

        # spoke labels — push outward with tick_params pad; use wrapped text
        ax.set_xticks(angles)
        ax.set_xticklabels(wrapped_labels, fontsize=12, fontweight="bold")
        ax.tick_params(pad=26)   # push labels well clear of the outermost ring
        # Build ~3 concentric rings per side snapped to clean 0.05/0.10 multiples
        ax.set_yticks(_nice_radial_ticks(rmin, rmax, n_rings=3))
        # Bold + large radial distance labels
        plt.setp(ax.yaxis.get_ticklabels(), fontsize=12, fontweight="bold")

        # baseline ring at 0
        if rmin <= 0 <= rmax:
            theta = np.linspace(0, 2 * np.pi, 300)
            ax.plot(theta, [0.0] * 300, color="black", linewidth=2.2, linestyle="--")

        # No title, no legend — legend only appears in the summary grid
        plt.tight_layout()
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            out_path = save_dir / f"{axis_name}_spider_{m}.png"
            fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.5)
            print(f"Saved axis spider plot: {out_path}")

        figs.append(fig)

    return figs


# ---------------------------------------------------------------------------
# 2×8 spider summary grid  (2 rows = metrics, 8 columns = FACTOR_AXES)
# ---------------------------------------------------------------------------

def plot_spider_summary_grid(
    models_results: Dict[str, Dict],
    metrics: List[str] = None,
    iou_threshold: float = 0.5,
    save_dir: Optional[Path] = None,
    cell_size: float = 4.5,
) -> plt.Figure:
    """Assemble mAP per-axis spider charts into a clean 2x4 summary figure."""
    # Always mAP only
    metrics = ["map"]

    import matplotlib.gridspec as _gridspec
    import textwrap as _textwrap

    N_COLS = 4
    axes_list = list(FACTOR_AXES)   # 8 axes
    n_grid_rows = 2                 # 2 half-rows for the 8 axes
    iou_key = f"iou_{iou_threshold}"

    # ------------------------------------------------------------------
    # Pre-compute baseline and per-variant delta for every model × axis
    # ------------------------------------------------------------------
    model_baselines: Dict[str, Dict[str, float]] = {}
    model_axis_deltas: Dict[str, Dict[str, Dict[str, List[float]]]] = {}

    for model_label, res in models_results.items():
        res_dict = res.get("results", {})
        baseline_entry = _find_baseline_entry(res_dict)
        if not baseline_entry:
            bm = {m: 0.0 for m in metrics}
        else:
            bm = {
                m: baseline_entry.get("metrics_by_iou", {}).get(iou_key, {}).get(m, 0.0)
                for m in metrics
            }
        model_baselines[model_label] = bm

        axis_deltas: Dict[str, Dict[str, List[float]]] = {}
        for ax_obj in axes_list:
            per_metric: Dict[str, List[float]] = {m: [] for m in metrics}
            for v in ax_obj.values:
                entry = _get_ofat_variant_entry(res_dict, ax_obj.name, v)
                for m in metrics:
                    if entry:
                        raw = entry.get("metrics_by_iou", {}).get(iou_key, {}).get(m, bm[m])
                    else:
                        raw = bm[m]
                    per_metric[m].append(raw - bm[m])
            axis_deltas[ax_obj.name] = per_metric
        model_axis_deltas[model_label] = axis_deltas

    # ------------------------------------------------------------------
    # Build figure
    # ------------------------------------------------------------------
    fig_w = cell_size * N_COLS
    fig_h = cell_size * n_grid_rows

    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = _gridspec.GridSpec(
        n_grid_rows,
        N_COLS,
        figure=fig,
        hspace=0.60,            # horizontal gap 
        wspace=0.40,            # vertical gap - adjusted to give breathing room
        height_ratios=[1.0] * n_grid_rows,
        bottom=0.05,
        top=0.95,               # Reclaimed the space from the removed legend
        left=0.05,
        right=0.95,
    )

    palette = sns.color_palette("tab10", n_colors=len(models_results))
    model_labels = list(models_results.keys())
    m = "map"   # single metric

    for half in range(2):           # half=0 → axes 0-3,  half=1 → axes 4-7
        grid_row = half
        axis_slice = axes_list[half * N_COLS : (half + 1) * N_COLS]

        for col_idx, ax_obj in enumerate(axis_slice):
            ax = fig.add_subplot(gs[grid_row, col_idx], projection="polar")
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)

            variants = ax_obj.values
            spoke_labels = [_axis_value_label(ax_obj.name, v) for v in variants]
            
            # Abbreviate long labels before wrapping to a tighter width (12)
            spoke_labels = [_abbreviate_spoke_label(lbl) for lbl in spoke_labels]
            wrapped_spoke_labels = [_textwrap.fill(lbl, width=12) for lbl in spoke_labels]

            n_spokes = len(variants)
            angles = np.linspace(0, 2 * np.pi, n_spokes, endpoint=False).tolist()
            angles_closed = angles + angles[:1]

            # Radial limits for this axis/metric cell
            all_d: List[float] = []
            for ml in model_labels:
                all_d.extend(model_axis_deltas[ml][ax_obj.name][m])
            d_max = max(all_d) if all_d else 0.0
            d_min = min(all_d) if all_d else 0.0
            pad = max(0.02, 0.1 * max(abs(d_max), abs(d_min), 1e-9))
            rmax = d_max + pad
            rmin = d_min - pad
            if abs(rmax - rmin) < 1e-6:
                rmax += 0.05
                rmin -= 0.05
            ax.set_ylim(rmin, rmax)

            # Draw polygons
            for idx, ml in enumerate(model_labels):
                deltas = model_axis_deltas[ml][ax_obj.name][m]
                deltas_closed = deltas + deltas[:1]
                color = palette[idx % len(palette)]
                line, = ax.plot(
                    angles_closed, deltas_closed,
                    color=color, linewidth=2, linestyle='solid',
                    label=ml,
                )
                ax.fill(angles_closed, deltas_closed, color=color, alpha=0.1) # Dropped alpha

            # Style the background grid and remove heavy outer spine
            ax.spines['polar'].set_visible(False)
            ax.xaxis.grid(True, color='grey', linestyle='dashed', alpha=0.5)
            ax.yaxis.grid(True, color='grey', linestyle='dashed', alpha=0.5)

            # Spoke labels
            ax.set_xticks(angles)
            ax.set_xticklabels(wrapped_spoke_labels, fontsize=9)
            ax.tick_params(pad=18) 

            # Baseline ring
            if rmin <= 0 <= rmax:
                theta_ring = np.linspace(0, 2 * np.pi, 300)
                ax.plot(theta_ring, [0.0] * 300,
                        color="black", linewidth=1.5, linestyle="--")

            # Clean Radial tick labels
            ticks = _nice_radial_ticks(rmin, rmax, n_rings=3)
            ax.set_yticks(ticks)
            ax.set_yticklabels([f"{t:g}" for t in ticks], color="grey", size=8)

            # Bold axis name as title
            ax.set_title(ax_obj.name.replace("_", " ").title(), weight='bold', pad=20)

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / "spider_summary_grid.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.3)
        print(f"Saved clean spider summary grid: {out_path}")

    return fig

# def plot_spider_summary_grid(
#     models_results: Dict[str, Dict],
#     metrics: List[str] = None,
#     iou_threshold: float = 0.5,
#     save_dir: Optional[Path] = None,
#     cell_size: float = 5.0,
# ) -> plt.Figure:
#     """Assemble mAP per-axis spider charts into a 2×4 summary figure.

#     Layout
#     ------
#     The 8 FACTOR_AXES are arranged across two rows of 4 columns::

#         row 0 │ axes 0-3
#         row 1 │ axes 4-7

#     Only mAP is shown (one metric, two rows).  A single shared legend is
#     placed at the bottom of the figure.  The axis name is rendered as a bold
#     subtitle below each polar cell.

#     Args:
#         models_results: mapping ``model_label -> results dict`` (same schema as
#                         :func:`plot_axis_spider_comparison`).
#         metrics:        ignored — always uses ``["map"]``.
#         iou_threshold:  IoU threshold used for metric extraction.
#         save_dir:       directory in which to save
#                         ``spider_summary_grid.png`` (skipped if ``None``).
#         cell_size:      width **and** height of each individual polar cell in
#                         inches.  Increase if spoke labels still overlap.

#     Returns:
#         The assembled :class:`matplotlib.figure.Figure`.
#     """
#     # Always mAP only
#     metrics = ["map"]

#     import matplotlib.gridspec as _gridspec
#     import textwrap as _textwrap

#     N_COLS = 4
#     axes_list = list(FACTOR_AXES)   # 8 axes
#     n_grid_rows = 2                 # 2 half-rows for the 8 axes
#     iou_key = f"iou_{iou_threshold}"

#     # ------------------------------------------------------------------
#     # Pre-compute baseline and per-variant delta for every model × axis
#     # ------------------------------------------------------------------
#     model_baselines: Dict[str, Dict[str, float]] = {}
#     model_axis_deltas: Dict[str, Dict[str, Dict[str, List[float]]]] = {}

#     for model_label, res in models_results.items():
#         res_dict = res.get("results", {})
#         baseline_entry = _find_baseline_entry(res_dict)
#         if not baseline_entry:
#             bm = {m: 0.0 for m in metrics}
#         else:
#             bm = {
#                 m: baseline_entry.get("metrics_by_iou", {}).get(iou_key, {}).get(m, 0.0)
#                 for m in metrics
#             }
#         model_baselines[model_label] = bm

#         axis_deltas: Dict[str, Dict[str, List[float]]] = {}
#         for ax_obj in axes_list:
#             per_metric: Dict[str, List[float]] = {m: [] for m in metrics}
#             for v in ax_obj.values:
#                 entry = _get_ofat_variant_entry(res_dict, ax_obj.name, v)
#                 for m in metrics:
#                     if entry:
#                         raw = entry.get("metrics_by_iou", {}).get(iou_key, {}).get(m, bm[m])
#                     else:
#                         raw = bm[m]
#                     per_metric[m].append(raw - bm[m])
#             axis_deltas[ax_obj.name] = per_metric
#         model_axis_deltas[model_label] = axis_deltas

#     # ------------------------------------------------------------------
#     # Build figure
#     # ------------------------------------------------------------------
#     fig_w = cell_size * N_COLS
#     fig_h = cell_size * n_grid_rows      # no legend strip needed

#     fig = plt.figure(figsize=(fig_w, fig_h))

#     gs = _gridspec.GridSpec(
#         n_grid_rows,            # no dummy legend row
#         N_COLS,
#         figure=fig,
#         hspace=0.65,            # vertical gap — room for outer spoke labels
#         wspace=0.55,            # horizontal gap
#         height_ratios=[1.0] * n_grid_rows,
#         bottom=0.02,
#         top=0.98,
#         left=0.04,
#         right=0.96,
#     )

#     palette = sns.color_palette("tab10", n_colors=len(models_results))
#     model_labels = list(models_results.keys())
#     m = "map"   # single metric

#     legend_handles: List = []
#     legend_texts:   List[str] = []

#     for half in range(2):           # half=0 → axes 0-3,  half=1 → axes 4-7
#         grid_row = half
#         axis_slice = axes_list[half * N_COLS : (half + 1) * N_COLS]

#         for col_idx, ax_obj in enumerate(axis_slice):
#             ax = fig.add_subplot(gs[grid_row, col_idx], projection="polar")
#             ax.set_theta_offset(np.pi / 2)
#             ax.set_theta_direction(-1)

#             variants = ax_obj.values
#             spoke_labels = [_axis_value_label(ax_obj.name, v) for v in variants]
#             # Abbreviate long labels before wrapping
#             spoke_labels = [_abbreviate_spoke_label(lbl) for lbl in spoke_labels]
#             wrapped_spoke_labels = [_textwrap.fill(lbl, width=16) for lbl in spoke_labels]

#             n_spokes = len(variants)
#             angles = np.linspace(0, 2 * np.pi, n_spokes, endpoint=False).tolist()
#             angles_closed = angles + angles[:1]

#             # Radial limits for this axis/metric cell
#             all_d: List[float] = []
#             for ml in model_labels:
#                 all_d.extend(model_axis_deltas[ml][ax_obj.name][m])
#             d_max = max(all_d) if all_d else 0.0
#             d_min = min(all_d) if all_d else 0.0
#             pad = max(0.02, 0.1 * max(abs(d_max), abs(d_min), 1e-9))
#             rmax = d_max + pad
#             rmin = d_min - pad
#             if abs(rmax - rmin) < 1e-6:
#                 rmax += 0.05
#                 rmin -= 0.05
#             ax.set_ylim(rmin, rmax)

#             # Draw polygons
#             for idx, ml in enumerate(model_labels):
#                 deltas = model_axis_deltas[ml][ax_obj.name][m]
#                 deltas_closed = deltas + deltas[:1]
#                 color = palette[idx % len(palette)]
#                 line, = ax.plot(
#                     angles_closed, deltas_closed,
#                     color=color, linewidth=1.8,
#                     label=ml,
#                 )
#                 ax.fill(angles_closed, deltas_closed, color=color, alpha=0.12)

#                 # Collect legend artists once only
#                 if half == 0 and col_idx == 0:
#                     legend_handles.append(line)
#                     legend_texts.append(ml)

#             # Spoke labels
#             ax.set_xticks(angles)
#             ax.set_xticklabels(wrapped_spoke_labels, fontsize=10, fontweight="bold")
#             ax.tick_params(pad=22)   # push labels well clear of the outermost ring

#             # Baseline ring
#             if rmin <= 0 <= rmax:
#                 theta_ring = np.linspace(0, 2 * np.pi, 300)
#                 ax.plot(theta_ring, [0.0] * 300,
#                         color="black", linewidth=2.0, linestyle="--")

#             # Radial tick labels — 3 rings per side, values snapped to
#             # clean 0.05 / 0.10 multiples so numbers are always readable.
#             ax.set_yticks(_nice_radial_ticks(rmin, rmax, n_rings=3))
#             # Bold + large radial distance labels
#             plt.setp(ax.yaxis.get_ticklabels(), fontsize=11, fontweight="bold")

#             # Bold axis name subtitle below the polar cell — omitted to
#             # maximise space; the grid layout itself communicates axis identity.

#     # Legend and axis labels removed — maximise cell area.

#     if save_dir:
#         save_dir = Path(save_dir)
#         save_dir.mkdir(parents=True, exist_ok=True)
#         out_path = save_dir / "spider_summary_grid.png"
#         fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.3)
#         print(f"Saved spider summary grid: {out_path}")

#     return fig
