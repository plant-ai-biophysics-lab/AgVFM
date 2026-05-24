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
    top_n: Optional[int] = 20,
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
        top_n: If set, show only the top N configs by metric value.
               The baseline is always included regardless of its rank.
               Set to None to show all configs.
    
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
    
    # Keep only top_n entries, always preserving the baseline
    if top_n is not None and len(config_data) > top_n:
        top_entries = config_data[:top_n]
        top_names   = {d["name"] for d in top_entries}
        # Re-insert baseline at its sorted position if it was trimmed
        if baseline_config not in top_names:
            baseline_entry = next(
                (d for d in config_data if d["name"] == baseline_config), None
            )
            if baseline_entry is not None:
                top_entries.append(baseline_entry)
                top_entries.sort(key=lambda x: x["value"], reverse=True)
        config_data = top_entries
    
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
    n_shown  = len(config_data)
    n_total  = len(results_dict)
    top_note = f" (top {n_shown} of {n_total})" if top_n is not None and n_shown < n_total else ""
    ax.set_title(f"Combination Performance: {metric_label}@IoU={iou_threshold}{top_note}\n"
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

    # ---------------------------------------------------------------
    # Reference baselines drawn from combination_results.
    # "No negation" = comb_clr_spp_anat_gram  (color+species+anatomy+grammar,
    #                 same base as absorber configs, no negation token)
    # "Text negation" = best single-negation combo present in the dict;
    #                   prefer bud+calyx, fall back to any neg variant.
    # ---------------------------------------------------------------
    _NO_NEG_KEY = "comb_clr_spp_anat_gram"
    _NEG_PRIORITY = [
        "comb_clr_spp_anat_gram_neg_bud_calyx",
        "comb_clr_spp_anat_gram_neg_calyx",
        "comb_clr_spp_anat_gram_neg_bud",
        "comb_clr_spp_anat_gram_neg_leaf",
    ]
    no_neg_result = comb_dict.get(_NO_NEG_KEY, {})
    neg_result = next(
        (comb_dict[k] for k in _NEG_PRIORITY if k in comb_dict),
        {},
    )
    neg_key_used = next((k for k in _NEG_PRIORITY if k in comb_dict), None)

    # Absorber entries: all keys starting with "abs_" in absorber_results
    absorber_keys = sorted(k for k in abs_dict if k.startswith("abs_"))

    # Extract metrics
    configs = []
    values = []
    labels = []
    colors = []

    def _get_val(result_dict):
        return (
            result_dict.get("metrics_by_iou", {})
            .get(f"iou_{iou_threshold}", {})
            .get(metric, 0.0)
        )

    # Add no-negation baseline
    if no_neg_result:
        configs.append(_NO_NEG_KEY)
        values.append(_get_val(no_neg_result))
        labels.append(f"{_NO_NEG_KEY}: {no_neg_result.get('description', 'No negation')}")
        colors.append("#3498db")  # Blue

    # Add text-negation reference
    if neg_result and neg_key_used:
        configs.append(neg_key_used)
        values.append(_get_val(neg_result))
        labels.append(f"{neg_key_used}: {neg_result.get('description', 'Text negation')}")
        colors.append("#e67e22")  # Orange

    # Add all absorber configs
    for abs_name in absorber_keys:
        abs_result = abs_dict[abs_name]
        configs.append(abs_name)
        values.append(_get_val(abs_result))
        desc = abs_result.get("description", abs_name)
        labels.append(f"{abs_name}: {desc}")
        colors.append("#2ecc71")  # Green

    if not configs:
        raise ValueError(
            "No results found for absorber comparison. "
            f"Expected keys like '{_NO_NEG_KEY}' in combination_results and "
            f"'abs_*' keys in absorber_results. "
            f"Got combination keys: {list(comb_dict.keys())[:10]}, "
            f"absorber keys: {list(abs_dict.keys())[:10]}"
        )
    
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
        Patch(facecolor="#3498db", label="No negation (baseline)"),
        Patch(facecolor="#e67e22", label="Text negation"),
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
    top_n: Optional[int] = 20,
) -> plt.Figure:
    """
    Plot all metrics (mAP, F1, Precision, Recall) in a 2x2 grid.
    
    Args:
        results: Results dictionary from combination tests
        iou_threshold: IoU threshold for metrics
        save_path: Optional path to save figure
        figsize: Figure size
        baseline_config: Name of baseline config
        top_n: If set, each subplot shows only the top N configs by that
               metric. The baseline is always included regardless of rank.
               Set to None to show all configs.
    
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
        
        # Keep only top_n entries, always preserving the baseline
        n_total_cfg = len(config_data)
        if top_n is not None and n_total_cfg > top_n:
            top_entries = config_data[:top_n]
            top_names   = {d["name"] for d in top_entries}
            if baseline_config not in top_names:
                baseline_entry = next(
                    (d for d in config_data if d["name"] == baseline_config), None
                )
                if baseline_entry is not None:
                    top_entries.append(baseline_entry)
                    top_entries.sort(key=lambda x: x["value"], reverse=True)
            config_data = top_entries
        
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
        n_shown_cfg = len(config_data)
        top_note = (
            f" (top {n_shown_cfg} of {n_total_cfg})"
            if top_n is not None and n_shown_cfg < n_total_cfg
            else ""
        )
        ax.set_title(f"{label}@IoU={iou_threshold}{top_note}\nBaseline: {baseline_value:.4f}", 
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


# ---------------------------------------------------------------------------
# Phase 2 per-component spider / radar charts (multi-model overlay)
# ---------------------------------------------------------------------------

# Tags that appear in combination config key names, with display labels.
# A config "contains" a component if its key includes the tag as a word token
# (split by underscores).
_COMBO_COMPONENTS: List[Tuple[str, str]] = [
    ("spp",  "Species/\nTaxonomy"),
    ("clr",  "Color"),
    ("anat", "Anatomy"),
    ("gram", "Grammar"),
    ("neg",  "Negation"),
]


def _key_has_tag(key: str, tag: str) -> bool:
    """Return True if *tag* appears as a token in the underscore-split *key*."""
    return tag in key.split("_")


def plot_combo_component_spider(
    models_results: Dict[str, Dict],
    metrics: List[str] = None,
    iou_threshold: float = 0.5,
    baseline_config: str = "comb_species",
    save_dir: Optional[Path] = None,
    figsize: Tuple[int, int] = (9, 9),
) -> List[plt.Figure]:
    """Spider / radar chart for Phase 2: spokes = prompt components.

    For each component (species, color, anatomy, grammar, negation) the
    radial value is the **mean metric delta** (vs. ``baseline_config``) over
    all combination configs whose key contains that component tag.

    One figure is produced per requested metric; all figures overlay all
    supplied models as differently coloured polygons.

    Args:
        models_results:   ``{model_label: results_dict}`` where each
                          ``results_dict`` is loaded directly from a Phase 2
                          JSON file.
        metrics:          Metric keys to plot (default: ``["map", "f1"]``).
        iou_threshold:    IoU threshold for metric extraction.
        baseline_config:  Key of the baseline configuration (default:
                          ``"comb_species"``).
        save_dir:         Directory to write PNGs.  Filenames are
                          ``combo_component_spider_{metric}.png``.
        figsize:          Figure size.

    Returns:
        List of ``matplotlib.Figure`` objects (one per metric).
    """
    import textwrap as _textwrap

    if metrics is None:
        metrics = ["map", "f1"]

    iou_key = f"iou_{iou_threshold}"
    figs: List[plt.Figure] = []

    # ------------------------------------------------------------------
    # Pre-compute per-model baseline and per-component mean deltas
    # ------------------------------------------------------------------
    model_baselines: Dict[str, Dict[str, float]] = {}
    model_component_deltas: Dict[str, Dict[str, Dict[str, float]]] = {}
    # model_component_deltas[model_label][metric][component_tag] = mean_delta

    for model_label, res in models_results.items():
        res_dict = res.get("results", {})

        # Baseline value
        baseline_entry = res_dict.get(baseline_config, {})
        baseline_metrics: Dict[str, float] = {}
        for m in metrics:
            baseline_metrics[m] = (
                baseline_entry.get("metrics_by_iou", {})
                .get(iou_key, {})
                .get(m, 0.0)
            )
        model_baselines[model_label] = baseline_metrics

        # Per-component mean delta over matching configs
        comp_deltas: Dict[str, Dict[str, float]] = {}
        for tag, _label in _COMBO_COMPONENTS:
            matching_keys = [
                k for k in res_dict
                if k != baseline_config and _key_has_tag(k, tag)
            ]
            per_metric: Dict[str, float] = {}
            for m in metrics:
                if matching_keys:
                    vals = [
                        res_dict[k]
                        .get("metrics_by_iou", {})
                        .get(iou_key, {})
                        .get(m, baseline_metrics[m])
                        for k in matching_keys
                    ]
                    per_metric[m] = float(np.mean(vals)) - baseline_metrics[m]
                else:
                    per_metric[m] = 0.0
            comp_deltas[tag] = per_metric
        model_component_deltas[model_label] = comp_deltas

    # ------------------------------------------------------------------
    # One figure per metric
    # ------------------------------------------------------------------
    n_comps = len(_COMBO_COMPONENTS)
    angles = np.linspace(0, 2 * np.pi, n_comps, endpoint=False).tolist()
    angles_closed = angles + angles[:1]
    spoke_labels = [_textwrap.fill(lbl, width=12) for _, lbl in _COMBO_COMPONENTS]
    tags = [tag for tag, _ in _COMBO_COMPONENTS]

    palette = sns.color_palette("tab10", n_colors=len(models_results))

    for m in metrics:
        # Gather all deltas to size the radial axis
        all_deltas = [
            model_component_deltas[ml][tag][m]
            for ml in models_results
            for tag in tags
        ]
        d_max = max(all_deltas) if all_deltas else 0.0
        d_min = min(all_deltas) if all_deltas else 0.0
        pad = max(0.005, 0.15 * max(abs(d_max), abs(d_min)))
        rmax = d_max + pad
        rmin = d_min - pad
        if abs(rmax - rmin) < 1e-6:
            rmax += 0.05
            rmin -= 0.05

        fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_ylim(rmin, rmax)

        # Baseline ring at Δ = 0
        if rmin <= 0.0 <= rmax:
            theta_full = np.linspace(0, 2 * np.pi, 300)
            ax.plot(theta_full, [0.0] * 300,
                    color="black", linewidth=1.0, linestyle="--", alpha=0.6)

        for idx, (model_label, _) in enumerate(models_results.items()):
            color = palette[idx % len(palette)]
            deltas = [model_component_deltas[model_label][tag][m] for tag in tags]
            deltas_closed = deltas + deltas[:1]
            ax.plot(angles_closed, deltas_closed,
                    label=model_label, color=color, linewidth=2, zorder=3)
            ax.fill(angles_closed, deltas_closed,
                    color=color, alpha=0.12, zorder=2)
            for ang, dv in zip(angles, deltas):
                ax.plot(ang, dv, "o", color=color, markersize=5, zorder=5)

        ax.set_xticks(angles)
        ax.set_xticklabels(spoke_labels, fontsize=8)
        ax.tick_params(pad=20)

        # No title — will be added in paper
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.05, 1.05),
            fontsize=8,
            framealpha=0.9,
        )

        plt.tight_layout()

        if save_dir is not None:
            out = Path(save_dir)
            out.mkdir(parents=True, exist_ok=True)
            out_path = out / f"combo_component_spider_{m}.png"
            fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.5)
            print(f"  Saved: {out_path}")

        figs.append(fig)

    return figs


# ---------------------------------------------------------------------------
# mAP summary grid for Phase 2 prompt components  (1 row × 5 cols)
# ---------------------------------------------------------------------------

def plot_combo_summary_grid(
    models_results: Dict[str, Dict],
    iou_threshold: float = 0.5,
    baseline_config: str = "comb_species",
    save_dir: Optional[Path] = None,
    cell_size: float = 4.0,
) -> plt.Figure:
    """One-row spider summary grid for the 5 Phase 2 prompt components (mAP).

    Layout
    ------
    Single row of 5 polar cells, one per :data:`_COMBO_COMPONENTS` entry.
    Each cell shows delta-from-baseline mAP for every model as an overlaid
    polygon.  A single shared legend is placed at the bottom of the figure.
    Component labels are bold subtitles below each cell.

    Args:
        models_results:  ``{model_label: results_dict}`` loaded from Phase 2 JSON.
        iou_threshold:   IoU threshold for metric extraction.
        baseline_config: Results key used as baseline (default: ``"comb_species"``).
        save_dir:        Directory to save ``combo_summary_grid.png``.
        cell_size:       Width and height of each polar cell in inches.

    Returns:
        The assembled :class:`matplotlib.figure.Figure`.
    """
    import matplotlib.gridspec as _gridspec
    import textwrap as _textwrap

    m = "map"
    iou_key = f"iou_{iou_threshold}"
    tags = [tag for tag, _ in _COMBO_COMPONENTS]
    display_labels = [lbl for _, lbl in _COMBO_COMPONENTS]
    N_COLS = len(_COMBO_COMPONENTS)   # 5

    # ------------------------------------------------------------------
    # Pre-compute per-model baseline and per-component deltas (same logic
    # as plot_combo_component_spider but for mAP only)
    # ------------------------------------------------------------------
    model_baselines: Dict[str, float] = {}
    model_component_deltas: Dict[str, List[float]] = {}

    for model_label, res in models_results.items():
        res_dict = res.get("results", {})
        baseline_entry = res_dict.get(baseline_config, {})
        bv = (
            baseline_entry.get("metrics_by_iou", {})
            .get(iou_key, {})
            .get(m, 0.0)
        )
        model_baselines[model_label] = bv

        deltas: List[float] = []
        for tag in tags:
            matching = [
                k for k in res_dict
                if k != baseline_config and _key_has_tag(k, tag)
            ]
            if matching:
                vals = [
                    res_dict[k].get("metrics_by_iou", {}).get(iou_key, {}).get(m, bv)
                    for k in matching
                ]
                deltas.append(float(np.mean(vals)) - bv)
            else:
                deltas.append(0.0)
        model_component_deltas[model_label] = deltas

    # ------------------------------------------------------------------
    # Build figure
    # ------------------------------------------------------------------
    fig_w = cell_size * N_COLS
    fig_h = cell_size + 0.8          # single data row + legend strip

    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = _gridspec.GridSpec(
        2,                            # 1 data row + 1 dummy legend anchor row
        N_COLS,
        figure=fig,
        hspace=0.30,
        wspace=0.50,
        height_ratios=[1.0, 0.001],
        bottom=0.07,
        top=0.97,
        left=0.03,
        right=0.97,
    )

    palette = sns.color_palette("tab10", n_colors=len(models_results))
    model_labels = list(models_results.keys())

    legend_handles: List = []
    legend_texts:   List[str] = []

    # ------------------------------------------------------------------
    # Per-component variant keys: union across all models so spokes are
    # consistent regardless of which model was evaluated on which configs.
    # ------------------------------------------------------------------
    component_variant_keys: Dict[str, List[str]] = {}
    for tag in tags:
        all_keys: List[str] = []
        for res in models_results.values():
            res_dict = res.get("results", {})
            for k in res_dict:
                if k != baseline_config and _key_has_tag(k, tag) and k not in all_keys:
                    all_keys.append(k)
        component_variant_keys[tag] = sorted(all_keys)

    model_per_comp_deltas: Dict[str, Dict[str, List[float]]] = {}
    for model_label, res in models_results.items():
        res_dict = res.get("results", {})
        bv = model_baselines[model_label]
        per_comp: Dict[str, List[float]] = {}
        for tag in tags:
            keys = component_variant_keys[tag]
            per_comp[tag] = [
                res_dict.get(k, {}).get("metrics_by_iou", {}).get(iou_key, {}).get(m, bv) - bv
                for k in keys
            ]
        model_per_comp_deltas[model_label] = per_comp

    for col_idx, (tag, disp_lbl) in enumerate(_COMBO_COMPONENTS):
        variant_keys = component_variant_keys[tag]
        n_spokes = len(variant_keys)

        ax = fig.add_subplot(gs[0, col_idx], projection="polar")
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        if n_spokes == 0:
            ax.set_visible(False)
            continue

        # Radial limits for this component cell
        cell_d: List[float] = [
            d
            for ml in model_labels
            for d in model_per_comp_deltas[ml][tag]
        ]
        d_max = max(cell_d) if cell_d else 0.0
        d_min = min(cell_d) if cell_d else 0.0
        pad = max(0.02, 0.1 * max(abs(d_max), abs(d_min), 1e-9))
        rmax = d_max + pad
        rmin = d_min - pad
        if abs(rmax - rmin) < 1e-6:
            rmax += 0.05
            rmin -= 0.05
        ax.set_ylim(rmin, rmax)

        angles = np.linspace(0, 2 * np.pi, n_spokes, endpoint=False).tolist()
        angles_closed = angles + angles[:1]

        # Spoke labels: shorten config key names
        spoke_labels = [_textwrap.fill(k.replace("comb_", "").replace("_", " "), width=10)
                        for k in variant_keys]

        for idx, ml in enumerate(model_labels):
            deltas = model_per_comp_deltas[ml][tag]
            deltas_closed = deltas + deltas[:1]
            color = palette[idx % len(palette)]
            line, = ax.plot(
                angles_closed, deltas_closed,
                color=color, linewidth=1.8, label=ml,
            )
            ax.fill(angles_closed, deltas_closed, color=color, alpha=0.12)

            if col_idx == 0:
                legend_handles.append(line)
                legend_texts.append(ml)

        ax.set_xticks(angles)
        ax.set_xticklabels(spoke_labels, fontsize=7)
        ax.tick_params(pad=10)

        if rmin <= 0 <= rmax:
            theta_ring = np.linspace(0, 2 * np.pi, 300)
            ax.plot(theta_ring, [0.0] * 300,
                    color="black", linewidth=0.8, linestyle="--")

        ax.yaxis.set_tick_params(labelsize=6)

        # Bold component label as subtitle
        ax.set_xlabel(
            disp_lbl.replace("\n", " "),
            fontsize=9,
            fontweight="bold",
            labelpad=16,
        )

    # ------------------------------------------------------------------
    # Shared legend at figure bottom
    # ------------------------------------------------------------------
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_texts,
            loc="lower center",
            ncol=min(len(model_labels), 6),
            fontsize=10,
            framealpha=0.9,
            bbox_to_anchor=(0.5, 0.0),
        )

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / "combo_summary_grid.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.3)
        print(f"Saved combo summary grid: {out_path}")

    return fig
