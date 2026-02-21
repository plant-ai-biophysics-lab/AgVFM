"""Visualization for Phase 2 combination and absorber results."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_combination_performance(
    results: Dict,
    metric: str = "map",
    iou_threshold: float = 0.5,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 8),
    baseline_config: str = "C3",
) -> plt.Figure:
    """
    Plot combination performance ranked by metric value.
    
    Args:
        results: Results dictionary from combination tests
        metric: Metric to plot ('map', 'f1', 'precision', 'recall')
        iou_threshold: IoU threshold for metrics
        save_path: Optional path to save figure
        figsize: Figure size
        baseline_config: Name of baseline config (default: "C3")
    
    Returns:
        matplotlib Figure
    """
    results_dict = results.get("results", {})
    
    if not results_dict:
        raise ValueError("No results found - cannot generate plot")
    
    # Get baseline value
    baseline_result = results_dict.get(baseline_config, {})
    baseline_value = (
        baseline_result.get("metrics_by_iou", {})
        .get(f"iou_{iou_threshold}", {})
        .get(metric, 0.0)
    )
    
    # Extract all configs and their metrics
    config_data = []
    for config_name, config_result in results_dict.items():
        metric_value = (
            config_result.get("metrics_by_iou", {})
            .get(f"iou_{iou_threshold}", {})
            .get(metric, 0.0)
        )
        delta = metric_value - baseline_value
        
        # Get description for better labels
        description = config_result.get("description", "")
        prompt = config_result.get("prompt", "")
        
        config_data.append({
            "name": config_name,
            "value": metric_value,
            "delta": delta,
            "description": description,
            "prompt": prompt,
        })
    
    # Sort by metric value (descending)
    config_data.sort(key=lambda x: x["value"], reverse=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare data for plotting
    names = [d["name"] for d in config_data]
    values = [d["value"] for d in config_data]
    deltas = [d["delta"] for d in config_data]
    
    # Color bars: positive delta = green, negative = red, zero = gray
    colors = [
        "#2ecc71" if d > 0 else "#e74c3c" if d < 0 else "#95a5a6"
        for d in deltas
    ]
    
    # Create bar plot
    bars = ax.barh(range(len(names)), values, color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)
    
    # Add baseline line
    ax.axvline(baseline_value, color="black", linestyle="--", linewidth=2, label=f"Baseline ({baseline_config})")
    
    # Add value labels on bars
    for i, (bar, val, delta) in enumerate(zip(bars, values, deltas)):
        # Position label at the end of the bar
        label_x = val + (max(values) - min(values)) * 0.01
        delta_str = f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}"
        ax.text(label_x, i, f"{val:.3f}\n({delta_str})", 
                va="center", ha="left", fontsize=9)
    
    # Set y-axis labels
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()  # Top to bottom
    
    # Labels and title
    metric_label = metric.upper() if metric == "map" else metric.capitalize()
    ax.set_xlabel(f"{metric_label}@IoU={iou_threshold}", fontsize=12, fontweight="bold")
    ax.set_title(f"Combination Performance: {metric_label}@IoU={iou_threshold}\n"
                 f"Baseline: {baseline_config} ({baseline_value:.4f})", 
                 fontsize=14, fontweight="bold", pad=20)
    
    # Grid
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    
    # Legend
    ax.legend(loc="lower right", fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.5)
        print(f"  Saved: {save_path}")
    
    return fig


def plot_absorber_comparison(
    combination_results: Dict,
    absorber_results: Dict,
    metric: str = "f1",
    iou_threshold: float = 0.5,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """
    Compare text negation (C1) vs absorber architecture configs.
    
    Args:
        combination_results: Results from combination tests
        absorber_results: Results from absorber tests
        metric: Metric to compare ('map', 'f1', 'precision', 'recall')
        iou_threshold: IoU threshold for metrics
        save_path: Optional path to save figure
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    comb_dict = combination_results.get("results", {})
    abs_dict = absorber_results.get("results", {})
    
    # Get C1 (text negation) and C2 (no negation) for comparison
    c1_result = comb_dict.get("C1", {})
    c2_result = comb_dict.get("C2", {})
    
    # Get absorber configs
    absorber_configs = ["H3b", "H2a", "H3c", "H5"]
    
    # Extract metrics
    configs = []
    values = []
    labels = []
    colors = []
    
    # Add C2 (baseline without negation)
    if c2_result:
        c2_value = (
            c2_result.get("metrics_by_iou", {})
            .get(f"iou_{iou_threshold}", {})
            .get(metric, 0.0)
        )
        configs.append("C2")
        values.append(c2_value)
        labels.append("C2: No negation")
        colors.append("#3498db")  # Blue
    
    # Add C1 (text negation)
    if c1_result:
        c1_value = (
            c1_result.get("metrics_by_iou", {})
            .get(f"iou_{iou_threshold}", {})
            .get(metric, 0.0)
        )
        configs.append("C1")
        values.append(c1_value)
        labels.append("C1: Text negation")
        colors.append("#e67e22")  # Orange
    
    # Add absorber configs
    for abs_name in absorber_configs:
        abs_result = abs_dict.get(abs_name, {})
        if abs_result:
            abs_value = (
                abs_result.get("metrics_by_iou", {})
                .get(f"iou_{iou_threshold}", {})
                .get(metric, 0.0)
            )
            configs.append(abs_name)
            values.append(abs_value)
            desc = abs_result.get("description", abs_name)
            labels.append(f"{abs_name}: {desc}")
            colors.append("#2ecc71")  # Green
    
    if not configs:
        raise ValueError("No results found for comparison")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bar plot
    bars = ax.bar(range(len(configs)), values, color=colors, alpha=0.7, edgecolor="black", linewidth=1)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    
    # Set x-axis labels
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, fontsize=10, rotation=45, ha="right")
    
    # Labels and title
    metric_label = metric.upper() if metric == "map" else metric.capitalize()
    ax.set_ylabel(f"{metric_label}@IoU={iou_threshold}", fontsize=12, fontweight="bold")
    ax.set_title(f"Text Negation vs Absorber Architecture: {metric_label}@IoU={iou_threshold}",
                 fontsize=14, fontweight="bold", pad=20)
    
    # Grid
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#3498db", label="No negation (C2)"),
        Patch(facecolor="#e67e22", label="Text negation (C1)"),
        Patch(facecolor="#2ecc71", label="Absorber architecture"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.5)
        print(f"  Saved: {save_path}")
    
    return fig


def plot_precision_recall_tradeoff(
    combination_results: Dict,
    absorber_results: Optional[Dict] = None,
    iou_threshold: float = 0.5,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 10),
) -> plt.Figure:
    """
    Plot Precision-Recall scatter with F1 iso-lines.
    
    Args:
        combination_results: Results from combination tests
        absorber_results: Optional results from absorber tests
        iou_threshold: IoU threshold for metrics
        save_path: Optional path to save figure
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    comb_dict = combination_results.get("results", {})
    
    # Extract precision and recall for all configs
    configs = []
    precisions = []
    recalls = []
    f1_scores = []
    labels = []
    
    for config_name, config_result in comb_dict.items():
        metrics = config_result.get("metrics_by_iou", {}).get(f"iou_{iou_threshold}", {})
        precision = metrics.get("precision", 0.0)
        recall = metrics.get("recall", 0.0)
        f1 = metrics.get("f1", 0.0)
        
        configs.append(config_name)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        labels.append(config_name)
    
    # Add absorber configs if provided
    if absorber_results:
        abs_dict = absorber_results.get("results", {})
        for config_name, config_result in abs_dict.items():
            metrics = config_result.get("metrics_by_iou", {}).get(f"iou_{iou_threshold}", {})
            precision = metrics.get("precision", 0.0)
            recall = metrics.get("recall", 0.0)
            f1 = metrics.get("f1", 0.0)
            
            configs.append(config_name)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
            labels.append(config_name)
    
    if not configs:
        raise ValueError("No results found - cannot generate plot")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot F1 iso-lines
    p_range = np.linspace(0.01, 1.0, 100)
    for f1_target in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        r_range = f1_target * 2 * p_range / (2 * p_range - f1_target + 1e-10)
        r_range = np.clip(r_range, 0, 1)
        valid = (r_range > 0) & (r_range <= 1) & (p_range > 0) & (p_range <= 1)
        if np.any(valid):
            ax.plot(p_range[valid], r_range[valid], '--', color='gray', alpha=0.3, linewidth=0.5)
            # Add F1 label
            if np.any(valid):
                mid_idx = len(p_range[valid]) // 2
                ax.text(p_range[valid][mid_idx], r_range[valid][mid_idx], 
                       f'F1={f1_target:.1f}', fontsize=8, alpha=0.5, rotation=-45)
    
    # Color points: C1 (text negation) = orange, C2 (no negation) = blue, absorbers = green, others = gray
    colors = []
    for config_name in configs:
        if config_name == "C1":
            colors.append("#e67e22")  # Orange
        elif config_name == "C2":
            colors.append("#3498db")  # Blue
        elif config_name.startswith("H"):
            colors.append("#2ecc71")  # Green
        else:
            colors.append("#95a5a6")  # Gray
    
    # Plot points
    scatter = ax.scatter(precisions, recalls, c=colors, s=100, alpha=0.7, 
                        edgecolors="black", linewidths=1, zorder=5)
    
    # Add labels for key configs
    key_configs = ["C1", "C2", "C3", "H3b", "H2a", "H3c", "H5"]
    for i, (config_name, prec, rec) in enumerate(zip(configs, precisions, recalls)):
        if config_name in key_configs:
            ax.annotate(config_name, (prec, rec), 
                       xytext=(5, 5), textcoords="offset points",
                       fontsize=9, fontweight="bold",
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    
    # Labels and title
    ax.set_xlabel("Precision@IoU=0.5", fontsize=12, fontweight="bold")
    ax.set_ylabel("Recall@IoU=0.5", fontsize=12, fontweight="bold")
    ax.set_title("Precision-Recall Tradeoff\n(F1 iso-lines shown)", 
                 fontsize=14, fontweight="bold", pad=20)
    
    # Set limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    
    # Grid
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#3498db", label="C2: No negation"),
        Patch(facecolor="#e67e22", label="C1: Text negation"),
        Patch(facecolor="#2ecc71", label="Absorber architecture"),
        Patch(facecolor="#95a5a6", label="Other combinations"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.5)
        print(f"  Saved: {save_path}")
    
    return fig


def plot_all_metrics_combinations(
    results: Dict,
    iou_threshold: float = 0.5,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (20, 18),
    baseline_config: str = "C3",
) -> plt.Figure:
    """
    Plot all metrics (mAP, F1, Precision, Recall) in a 2x2 grid.
    
    Args:
        results: Results dictionary from combination tests
        iou_threshold: IoU threshold for metrics
        save_path: Optional path to save figure
        figsize: Figure size
        baseline_config: Name of baseline config
    
    Returns:
        matplotlib Figure
    """
    results_dict = results.get("results", {})
    
    if not results_dict:
        raise ValueError("No results found - cannot generate plot")
    
    # Get baseline values for all metrics
    baseline_result = results_dict.get(baseline_config, {})
    baseline_values = {}
    for metric in ["map", "f1", "precision", "recall"]:
        baseline_values[metric] = (
            baseline_result.get("metrics_by_iou", {})
            .get(f"iou_{iou_threshold}", {})
            .get(metric, 0.0)
        )
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    metrics = [("map", "mAP"), ("f1", "F1"), ("precision", "Precision"), ("recall", "Recall")]
    
    for idx, (metric, label) in enumerate(metrics):
        ax = axes[idx]
        baseline_value = baseline_values[metric]
        
        # Extract config data for this metric
        config_data = []
        for config_name, config_result in results_dict.items():
            metric_value = (
                config_result.get("metrics_by_iou", {})
                .get(f"iou_{iou_threshold}", {})
                .get(metric, 0.0)
            )
            delta = metric_value - baseline_value
            
            config_data.append({
                "name": config_name,
                "value": metric_value,
                "delta": delta,
            })
        
        # Sort by metric value (descending)
        config_data.sort(key=lambda x: x["value"], reverse=True)
        
        # Prepare data for plotting
        names = [d["name"] for d in config_data]
        values = [d["value"] for d in config_data]
        deltas = [d["delta"] for d in config_data]
        
        # Color bars: positive delta = green, negative = red, zero = gray
        colors = [
            "#2ecc71" if d > 0 else "#e74c3c" if d < 0 else "#95a5a6"
            for d in deltas
        ]
        
        # Create bar plot
        bars = ax.barh(range(len(names)), values, color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)
        
        # Add baseline line
        ax.axvline(baseline_value, color="black", linestyle="--", linewidth=2)
        
        # Add value labels on bars
        for i, (bar, val, delta) in enumerate(zip(bars, values, deltas)):
            label_x = val + (max(values) - min(values)) * 0.01 if max(values) > min(values) else val + 0.01
            delta_str = f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}"
            ax.text(label_x, i, f"{val:.3f}\n({delta_str})", 
                    va="center", ha="left", fontsize=8)
        
        # Set y-axis labels
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis()  # Top to bottom
        
        # Labels and title
        ax.set_xlabel(f"{label}@IoU={iou_threshold}", fontsize=11, fontweight="bold")
        ax.set_title(f"{label}@IoU={iou_threshold}\nBaseline: {baseline_value:.4f}", 
                     fontsize=12, fontweight="bold")
        
        # Grid
        ax.grid(axis="x", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
    
    plt.suptitle("Phase 2: Combination Performance - All Metrics", 
                 fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.5)
        print(f"  Saved: {save_path}")
    
    return fig
