"""Visualization for Phase 3 confidence threshold sweep results."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10


def load_confidence_sweep_results(results_file: Path) -> Dict:
    """Load confidence sweep results from JSON file."""
    import json
    
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    with open(results_file, "r") as f:
        return json.load(f)


def extract_sweep_data(results: Dict) -> Dict[str, Dict]:
    """
    Extract sweep data from results for plotting.
    
    Returns:
        Dictionary mapping config_name to {
            'thresholds': [list of thresholds],
            'maps': [list of mAP values],
            'f1s': [list of F1 values],
            'precisions': [list of precision values],
            'recalls': [list of recall values],
            'total_predictions': [list of prediction counts],
            'description': str,
            'prompt': str,
        }
    """
    results_dict = results.get("results", {})
    sweep_data = {}
    
    for config_name, config_data in results_dict.items():
        results_by_conf = config_data.get("results_by_conf", {})
        
        if not results_by_conf:
            continue
        
        # Extract metrics for each threshold
        thresholds = []
        maps = []
        f1s = []
        precisions = []
        recalls = []
        total_preds = []
        
        for thresh_str, result in sorted(results_by_conf.items(), key=lambda x: float(x[0])):
            thresholds.append(float(thresh_str))
            m = result.get("metrics", {})
            maps.append(m.get("map", 0.0))
            f1s.append(m.get("f1", 0.0))
            precisions.append(m.get("precision", 0.0))
            recalls.append(m.get("recall", 0.0))
            total_preds.append(result.get("total_predictions", 0))
        
        sweep_data[config_name] = {
            "thresholds": thresholds,
            "maps": maps,
            "f1s": f1s,
            "precisions": precisions,
            "recalls": recalls,
            "total_predictions": total_preds,
            "description": config_data.get("description", ""),
            "prompt": config_data.get("prompt", ""),
        }
    
    return sweep_data


def plot_performance_curves(
    sweep_data: Dict[str, Dict],
    metrics: List[str] = ["map", "f1", "precision", "recall"],
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (16, 10),
    model_name: str = "",
) -> plt.Figure:
    """
    Plot performance curves (mAP, F1, Precision, Recall) vs confidence threshold.
    
    Args:
        sweep_data: Dictionary from extract_sweep_data()
        metrics: List of metrics to plot
        save_path: Optional path to save figure
        figsize: Figure size
        model_name: Model name for title
    
    Returns:
        matplotlib Figure
    """
    n_metrics = len(metrics)
    n_configs = len(sweep_data)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    # Color palette for configs
    colors = sns.color_palette("husl", n_configs)
    
    metric_labels = {
        "map": "mAP@0.5",
        "f1": "F1 Score",
        "precision": "Precision",
        "recall": "Recall",
    }
    
    metric_keys = {
        "map": "maps",
        "f1": "f1s",
        "precision": "precisions",
        "recall": "recalls",
    }
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        metric_key = metric_keys[metric]
        metric_label = metric_labels[metric]
        
        # Plot each config
        for config_idx, (config_name, data) in enumerate(sweep_data.items()):
            thresholds = data["thresholds"]
            values = data[metric_key]
            color = colors[config_idx]
            
            # Find best point
            best_idx = np.argmax(values)
            best_thresh = thresholds[best_idx]
            best_val = values[best_idx]
            
            # Plot curve
            ax.plot(thresholds, values, marker="o", markersize=4, linewidth=2,
                   label=config_name, color=color, alpha=0.8)
            
            # Mark best point
            ax.scatter([best_thresh], [best_val], s=100, color=color, 
                      edgecolors="black", linewidths=2, zorder=5, marker="*")
        
        # Add vertical lines at pre-declared thresholds
        predeclared = [0.05, 0.10, 0.20, 0.30, 0.50]
        for thresh in predeclared:
            ax.axvline(thresh, color="gray", linestyle="--", alpha=0.3, linewidth=1)
        
        # Labels and formatting
        ax.set_xlabel("Confidence Threshold", fontsize=11, fontweight="bold")
        ax.set_ylabel(metric_label, fontsize=11, fontweight="bold")
        ax.set_title(f"{metric_label} vs Confidence Threshold", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        ax.set_xlim(0, 0.95)
    
    # Overall title
    model_title = f"{model_name} - " if model_name else ""
    fig.suptitle(f"{model_title}Performance Curves Across Confidence Thresholds", 
                 fontsize=14, fontweight="bold", y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", facecolor="white")
        print(f"Saved: {save_path}")
    
    return fig


def plot_precision_recall_tradeoff(
    sweep_data: Dict[str, Dict],
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 8),
    model_name: str = "",
) -> plt.Figure:
    """
    Plot Precision-Recall tradeoff with F1 iso-lines.
    
    Args:
        sweep_data: Dictionary from extract_sweep_data()
        save_path: Optional path to save figure
        figsize: Figure size
        model_name: Model name for title
    
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color palette
    n_configs = len(sweep_data)
    colors = sns.color_palette("husl", n_configs)
    
    # Plot F1 iso-lines
    f1_levels = [0.2, 0.3, 0.4, 0.5, 0.6]
    precision_range = np.linspace(0.1, 0.9, 100)
    for f1 in f1_levels:
        recall = precision_range * f1 / (2 * precision_range - f1)
        recall = np.clip(recall, 0, 1)
        valid = (recall >= 0) & (recall <= 1) & (precision_range > f1/2)
        if np.any(valid):
            ax.plot(recall[valid], precision_range[valid], 
                   color="gray", linestyle="--", alpha=0.3, linewidth=1)
            # Add F1 label
            mid_idx = len(valid[valid]) // 2
            if mid_idx > 0:
                ax.text(recall[valid][mid_idx], precision_range[valid][mid_idx], 
                       f"F1={f1:.1f}", fontsize=8, alpha=0.5, rotation=-45)
    
    # Plot each config
    for config_idx, (config_name, data) in enumerate(sweep_data.items()):
        precisions = data["precisions"]
        recalls = data["recalls"]
        thresholds = data["thresholds"]
        color = colors[config_idx]
        
        # Plot curve
        ax.plot(recalls, precisions, marker="o", markersize=6, linewidth=2,
               label=config_name, color=color, alpha=0.8)
        
        # Mark key thresholds
        key_thresholds = [0.05, 0.10, 0.20, 0.30, 0.50]
        for thresh in key_thresholds:
            if thresh in thresholds:
                idx = thresholds.index(thresh)
                ax.scatter([recalls[idx]], [precisions[idx]], s=80, color=color,
                          edgecolors="black", linewidths=1.5, zorder=5)
                ax.annotate(f"{thresh:.2f}", (recalls[idx], precisions[idx]),
                           xytext=(5, 5), textcoords="offset points", fontsize=8)
        
        # Mark best F1 point
        f1s = data["f1s"]
        best_f1_idx = np.argmax(f1s)
        ax.scatter([recalls[best_f1_idx]], [precisions[best_f1_idx]], s=150,
                  color=color, edgecolors="black", linewidths=2, zorder=6, marker="*")
    
    ax.set_xlabel("Recall", fontsize=12, fontweight="bold")
    ax.set_ylabel("Precision", fontsize=12, fontweight="bold")
    model_title = f"{model_name} - " if model_name else ""
    ax.set_title(f"{model_title}Precision-Recall Tradeoff\n(Curves show confidence threshold sweep)", 
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", facecolor="white")
        print(f"Saved: {save_path}")
    
    return fig


def plot_detection_volume(
    sweep_data: Dict[str, Dict],
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 8),
    model_name: str = "",
    n_images: int = 178,
) -> plt.Figure:
    """
    Plot detection volume (predictions per image) vs confidence threshold.
    
    Args:
        sweep_data: Dictionary from extract_sweep_data()
        save_path: Optional path to save figure
        figsize: Figure size
        model_name: Model name for title
        n_images: Number of images (for per-image calculation)
    
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color palette
    n_configs = len(sweep_data)
    colors = sns.color_palette("husl", n_configs)
    
    # Plot each config
    for config_idx, (config_name, data) in enumerate(sweep_data.items()):
        thresholds = data["thresholds"]
        total_preds = data["total_predictions"]
        preds_per_image = [p / n_images for p in total_preds]
        color = colors[config_idx]
        
        ax.plot(thresholds, preds_per_image, marker="o", markersize=5, linewidth=2,
               label=config_name, color=color, alpha=0.8)
    
    # Add ground truth reference line
    # Assuming ~6.4 objects per image (from previous analysis)
    gt_per_image = 6.4
    ax.axhline(gt_per_image, color="red", linestyle="--", linewidth=2, 
              label=f"GT density (~{gt_per_image:.1f}/image)", alpha=0.7)
    
    # Add vertical lines at key thresholds
    key_thresholds = [0.05, 0.10, 0.20, 0.30, 0.50]
    for thresh in key_thresholds:
        ax.axvline(thresh, color="gray", linestyle="--", alpha=0.3, linewidth=1)
    
    ax.set_xlabel("Confidence Threshold", fontsize=12, fontweight="bold")
    ax.set_ylabel("Predictions per Image", fontsize=12, fontweight="bold")
    model_title = f"{model_name} - " if model_name else ""
    ax.set_title(f"{model_title}Detection Volume vs Confidence Threshold", 
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)
    ax.set_xlim(0, 0.95)
    ax.set_yscale("log")
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", facecolor="white")
        print(f"Saved: {save_path}")
    
    return fig


def plot_detection_volume_linear(
    sweep_data: Dict[str, Dict],
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 8),
    model_name: str = "",
    n_images: int = 178,
) -> plt.Figure:
    """
    Plot detection volume (predictions per image) vs confidence threshold (linear scale).
    
    This is a non-log-transformed version of plot_detection_volume for better
    visualization of the relationship at lower thresholds.
    
    Args:
        sweep_data: Dictionary from extract_sweep_data()
        save_path: Optional path to save figure
        figsize: Figure size
        model_name: Model name for title
        n_images: Number of images (for per-image calculation)
    
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color palette
    n_configs = len(sweep_data)
    colors = sns.color_palette("husl", n_configs)
    
    # Plot each config
    for config_idx, (config_name, data) in enumerate(sweep_data.items()):
        thresholds = data["thresholds"]
        total_preds = data["total_predictions"]
        preds_per_image = [p / n_images for p in total_preds]
        color = colors[config_idx]
        
        ax.plot(thresholds, preds_per_image, marker="o", markersize=5, linewidth=2,
               label=config_name, color=color, alpha=0.8)
    
    # Add ground truth reference line
    # Assuming ~6.4 objects per image (from previous analysis)
    gt_per_image = 6.4
    ax.axhline(gt_per_image, color="red", linestyle="--", linewidth=2, 
              label=f"GT density (~{gt_per_image:.1f}/image)", alpha=0.7)
    
    # Add vertical lines at key thresholds
    key_thresholds = [0.05, 0.10, 0.20, 0.30, 0.50]
    for thresh in key_thresholds:
        ax.axvline(thresh, color="gray", linestyle="--", alpha=0.3, linewidth=1)
    
    ax.set_xlabel("Confidence Threshold", fontsize=12, fontweight="bold")
    ax.set_ylabel("Predictions per Image", fontsize=12, fontweight="bold")
    model_title = f"{model_name} - " if model_name else ""
    ax.set_title(f"{model_title}Detection Volume vs Confidence Threshold (Linear Scale)",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)
    ax.set_xlim(0, 0.95)
    # Note: No log scale - linear y-axis
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", facecolor="white")
        print(f"Saved: {save_path}")
    
    return fig


def plot_sensitivity_analysis(
    sweep_data: Dict[str, Dict],
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 8),
    model_name: str = "",
) -> plt.Figure:
    """
    Plot sensitivity analysis showing F1 change around optimal threshold.
    
    Args:
        sweep_data: Dictionary from extract_sweep_data()
        save_path: Optional path to save figure
        figsize: Figure size
        model_name: Model name for title
    
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Color palette
    n_configs = len(sweep_data)
    colors = sns.color_palette("husl", n_configs)
    
    # Left plot: F1 curves with optimal points
    ax1 = axes[0]
    for config_idx, (config_name, data) in enumerate(sweep_data.items()):
        thresholds = data["thresholds"]
        f1s = data["f1s"]
        color = colors[config_idx]
        
        # Find optimal
        best_idx = np.argmax(f1s)
        best_thresh = thresholds[best_idx]
        best_f1 = f1s[best_idx]
        
        # Plot curve
        ax1.plot(thresholds, f1s, marker="o", markersize=4, linewidth=2,
                label=config_name, color=color, alpha=0.8)
        
        # Mark optimal
        ax1.scatter([best_thresh], [best_f1], s=150, color=color,
                   edgecolors="black", linewidths=2, zorder=5, marker="*")
        
        # Add sensitivity region (±0.05)
        if best_idx > 0 and best_idx < len(thresholds) - 1:
            lower_idx = max(0, best_idx - 1)
            upper_idx = min(len(thresholds) - 1, best_idx + 1)
            ax1.axvspan(thresholds[lower_idx], thresholds[upper_idx], 
                       alpha=0.2, color=color)
    
    ax1.set_xlabel("Confidence Threshold", fontsize=11, fontweight="bold")
    ax1.set_ylabel("F1 Score", fontsize=11, fontweight="bold")
    ax1.set_title("F1 Score vs Confidence Threshold\n(★ = Optimal, shaded = ±0.05 region)", 
                  fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", fontsize=9)
    ax1.set_xlim(0, 0.95)
    
    # Right plot: Sensitivity metrics
    ax2 = axes[1]
    config_names = []
    optimal_thresholds = []
    optimal_f1s = []
    sensitivities = []
    
    for config_name, data in sweep_data.items():
        thresholds = data["thresholds"]
        f1s = data["f1s"]
        
        best_idx = np.argmax(f1s)
        best_f1 = f1s[best_idx]
        
        # Calculate sensitivity
        if best_idx > 0 and best_idx < len(f1s) - 1:
            f1_lower = f1s[best_idx - 1]
            f1_higher = f1s[best_idx + 1]
            sensitivity = max(abs(best_f1 - f1_lower), abs(best_f1 - f1_higher))
        else:
            sensitivity = 0.0
        
        config_names.append(config_name)
        optimal_thresholds.append(thresholds[best_idx])
        optimal_f1s.append(best_f1)
        sensitivities.append(sensitivity)
    
    # Bar plot of sensitivities
    bars = ax2.barh(config_names, sensitivities, color=colors[:n_configs], alpha=0.7, edgecolor="black")
    
    # Add value labels
    for bar, sens in zip(bars, sensitivities):
        width = bar.get_width()
        label = "Low" if sens < 0.02 else "Moderate" if sens < 0.05 else "High"
        ax2.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
               f"{sens:.4f} ({label})", va="center", ha="left", fontsize=9)
    
    ax2.set_xlabel("F1 Change with ±0.05 Threshold", fontsize=11, fontweight="bold")
    ax2.set_title("Threshold Sensitivity Analysis", fontsize=12, fontweight="bold")
    ax2.grid(axis="x", alpha=0.3)
    ax2.axvline(0.02, color="green", linestyle="--", alpha=0.5, label="Low threshold")
    ax2.axvline(0.05, color="orange", linestyle="--", alpha=0.5, label="Moderate threshold")
    
    # Overall title
    model_title = f"{model_name} - " if model_name else ""
    fig.suptitle(f"{model_title}Sensitivity Analysis", fontsize=14, fontweight="bold")
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", facecolor="white")
        print(f"Saved: {save_path}")
    
    return fig


def plot_config_comparison(
    sweep_data: Dict[str, Dict],
    metric: str = "f1",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 8),
    model_name: str = "",
) -> plt.Figure:
    """
    Plot comparison of all configs for a single metric.
    
    Args:
        sweep_data: Dictionary from extract_sweep_data()
        metric: Metric to compare ('map', 'f1', 'precision', 'recall')
        save_path: Optional path to save figure
        figsize: Figure size
        model_name: Model name for title
    
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color palette
    n_configs = len(sweep_data)
    colors = sns.color_palette("husl", n_configs)
    
    metric_keys = {
        "map": "maps",
        "f1": "f1s",
        "precision": "precisions",
        "recall": "recalls",
    }
    
    metric_labels = {
        "map": "mAP@0.5",
        "f1": "F1 Score",
        "precision": "Precision",
        "recall": "Recall",
    }
    
    metric_key = metric_keys[metric]
    metric_label = metric_labels[metric]
    
    # Plot each config
    for config_idx, (config_name, data) in enumerate(sweep_data.items()):
        thresholds = data["thresholds"]
        values = data[metric_key]
        color = colors[config_idx]
        
        # Find best point
        best_idx = np.argmax(values)
        best_thresh = thresholds[best_idx]
        best_val = values[best_idx]
        
        # Plot curve
        ax.plot(thresholds, values, marker="o", markersize=5, linewidth=2.5,
               label=f"{config_name} (best: {best_val:.4f} @ {best_thresh:.2f})", 
               color=color, alpha=0.8)
        
        # Mark best point
        ax.scatter([best_thresh], [best_val], s=120, color=color,
                  edgecolors="black", linewidths=2, zorder=5, marker="*")
    
    # Add vertical lines at key thresholds
    key_thresholds = [0.05, 0.10, 0.20, 0.30, 0.50]
    for thresh in key_thresholds:
        ax.axvline(thresh, color="gray", linestyle="--", alpha=0.3, linewidth=1)
    
    ax.set_xlabel("Confidence Threshold", fontsize=12, fontweight="bold")
    ax.set_ylabel(metric_label, fontsize=12, fontweight="bold")
    model_title = f"{model_name} - " if model_name else ""
    ax.set_title(f"{model_title}Config Comparison: {metric_label}", 
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)
    ax.set_xlim(0, 0.95)
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", facecolor="white")
        print(f"Saved: {save_path}")
    
    return fig
