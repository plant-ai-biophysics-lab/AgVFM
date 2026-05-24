"""Visualization helpers for Phase 1 factor analysis — HuggingFace OVD models.

This module adds comparison plots that are specific to the HF model pair
(GroundingDINO + OWLv2) and an optional four-model overview combining the
YOLO World / SAM3 results with the HF results.

All single-model per-metric plots (``plot_factor_contributions``,
``plot_all_metrics_factor_contributions``) are **re-exported from**
``phase1_factor_analysis`` so callers only need to import from here.

New functions
-------------
plot_hf_model_comparison
    Side-by-side GroundingDINO vs OWLv2 factor-contribution chart for a
    single metric.  Mirrors ``plot_factor_comparison`` from the base module
    but is parameterised for two generic HF result dicts.

plot_four_model_comparison
    2×2 grid: YOLO World | SAM3 (top row), GroundingDINO | OWLv2 (bottom
    row).  Allows direct cross-paradigm comparison on the same page.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns

# Re-export reusable helpers so callers only need this module
from phase1_factor_analysis import (  # noqa: F401
    plot_factor_contributions,
    plot_all_metrics_factor_contributions,
    _plot_single_factor_contributions,
    _sanitize_label,
)


# ---------------------------------------------------------------------------
# HF model comparison (GDino vs OWLv2)
# ---------------------------------------------------------------------------

def plot_hf_model_comparison(
    gdino_results: Dict,
    owlv2_results: Dict,
    metric: str = "map",
    iou_threshold: float = 0.5,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (16, 7),
) -> plt.Figure:
    """Side-by-side factor-contribution chart for GroundingDINO vs OWLv2.

    Args:
        gdino_results:  Results dict from ``grounding_dino_factor_analysis.json``.
        owlv2_results:  Results dict from ``owlv2_factor_analysis.json``.
        metric:         Metric key — ``"map"``, ``"f1"``, ``"precision"``,
                        or ``"recall"``.
        iou_threshold:  IoU threshold used to select the metric slice.
        save_path:      If given, save the figure to this path (PNG, 300 dpi).
        figsize:        Figure width × height in inches.

    Returns:
        matplotlib ``Figure``.
    """
    fig, (ax_gdino, ax_owlv2) = plt.subplots(1, 2, figsize=figsize, sharey=True)

    gdino_name = gdino_results.get("model", "GroundingDINO")
    owlv2_name = owlv2_results.get("model", "OWLv2")

    _plot_single_factor_contributions(ax_gdino, gdino_results, metric, iou_threshold, gdino_name)
    _plot_single_factor_contributions(ax_owlv2, owlv2_results, metric, iou_threshold, owlv2_name)

    # Remove redundant y-label on right panel (shared y-axis)
    ax_owlv2.set_ylabel("")

    metric_label = "mAP" if metric == "map" else metric.upper()
    fig.suptitle(
        f"HF Models — Factor Contributions ({metric_label}@IoU={iou_threshold})",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.4)
        print(f"Saved HF comparison plot to {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Four-model comparison (YOLO World + SAM3 + GDino + OWLv2)
# ---------------------------------------------------------------------------

def plot_four_model_comparison(
    yolo_results: Dict,
    sam3_results: Dict,
    gdino_results: Dict,
    owlv2_results: Dict,
    metric: str = "map",
    iou_threshold: float = 0.5,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (20, 12),
) -> plt.Figure:
    """2×2 grid comparing all four OVD models on a single metric.

    Layout::

        ┌─────────────────┬─────────────────┐
        │   YOLO World    │      SAM3        │  ← prompt-based (existing)
        ├─────────────────┼─────────────────┤
        │  GroundingDINO  │     OWLv2        │  ← HuggingFace
        └─────────────────┴─────────────────┘

    The y-axes are **not** shared across rows because the scale ranges may
    differ substantially between the prompt-based and HF models.  Within
    each row the y-axis is shared so left/right bars are directly comparable.

    Args:
        yolo_results:   Results dict for YOLO World.
        sam3_results:   Results dict for SAM3.
        gdino_results:  Results dict for GroundingDINO.
        owlv2_results:  Results dict for OWLv2.
        metric:         Metric key (``"map"``, ``"f1"``, ``"precision"``,
                        ``"recall"``).
        iou_threshold:  IoU threshold for metric slice.
        save_path:      Optional save path (PNG, 300 dpi).
        figsize:        Figure size.

    Returns:
        matplotlib ``Figure``.
    """
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.12)

    # Row 0 — existing models (shared y within row)
    ax_yolo = fig.add_subplot(gs[0, 0])
    ax_sam3 = fig.add_subplot(gs[0, 1], sharey=ax_yolo)

    # Row 1 — HF models (shared y within row)
    ax_gdino = fig.add_subplot(gs[1, 0])
    ax_owlv2 = fig.add_subplot(gs[1, 1], sharey=ax_gdino)

    _plot_single_factor_contributions(
        ax_yolo, yolo_results, metric, iou_threshold,
        yolo_results.get("model", "YOLO World"),
    )
    _plot_single_factor_contributions(
        ax_sam3, sam3_results, metric, iou_threshold,
        sam3_results.get("model", "SAM3"),
    )
    _plot_single_factor_contributions(
        ax_gdino, gdino_results, metric, iou_threshold,
        gdino_results.get("model", "GroundingDINO"),
    )
    _plot_single_factor_contributions(
        ax_owlv2, owlv2_results, metric, iou_threshold,
        owlv2_results.get("model", "OWLv2"),
    )

    # Remove inner y-labels (right column)
    ax_sam3.set_ylabel("")
    ax_owlv2.set_ylabel("")

    # Row labels
    _add_row_label(fig, gs[0, :], "Prompt-Based Models", fontsize=11)
    _add_row_label(fig, gs[1, :], "HuggingFace Models", fontsize=11)

    metric_label = "mAP" if metric == "map" else metric.upper()
    fig.suptitle(
        f"All Models — Factor Contributions ({metric_label}@IoU={iou_threshold})",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.5)
        print(f"Saved four-model comparison plot to {save_path}")

    return fig


def _add_row_label(fig: plt.Figure, subplot_spec, label: str, fontsize: int = 10) -> None:
    """Draw a centred italic row label above a GridSpec row."""
    # Get the bounding box of the GridSpec row in figure coordinates
    ss = subplot_spec
    # Use the subplotspec geometry to place text
    x0, y0, width, height = (
        ss.get_position(fig).x0,
        ss.get_position(fig).y0,
        ss.get_position(fig).width,
        ss.get_position(fig).height,
    )
    fig.text(
        x0 + width / 2,
        y0 + height + 0.01,
        label,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        style="italic",
        color="dimgray",
    )


# ---------------------------------------------------------------------------
# All-metrics HF comparison (one page, 4 rows × 2 columns)
# ---------------------------------------------------------------------------

def plot_hf_comparison_all_metrics(
    gdino_results: Dict,
    owlv2_results: Dict,
    iou_threshold: float = 0.5,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (20, 22),
) -> plt.Figure:
    """GDino vs OWLv2 side-by-side for all four metrics (4 rows × 2 columns).

    Each row is one metric (mAP, F1, Precision, Recall).  Within a row the
    y-axis is shared so the two models are directly comparable.

    Args:
        gdino_results:  GroundingDINO results dict.
        owlv2_results:  OWLv2 results dict.
        iou_threshold:  IoU threshold.
        save_path:      Optional save path.
        figsize:        Figure size.

    Returns:
        matplotlib ``Figure``.
    """
    metrics = [
        ("map",       "mAP"),
        ("f1",        "F1"),
        ("precision", "Precision"),
        ("recall",    "Recall"),
    ]

    gdino_name = gdino_results.get("model", "GroundingDINO")
    owlv2_name = owlv2_results.get("model", "OWLv2")

    fig, axes = plt.subplots(
        len(metrics), 2,
        figsize=figsize,
        sharey="row",       # share y within each row (same metric, both models)
    )

    for row, (metric_key, metric_label) in enumerate(metrics):
        ax_l = axes[row, 0]
        ax_r = axes[row, 1]

        _plot_single_factor_contributions(
            ax_l, gdino_results, metric_key, iou_threshold,
            f"{gdino_name}\n({metric_label})",
        )
        _plot_single_factor_contributions(
            ax_r, owlv2_results, metric_key, iou_threshold,
            f"{owlv2_name}\n({metric_label})",
        )
        ax_r.set_ylabel("")   # shared y — remove duplicate label

    fig.suptitle(
        f"HF Models — All Metrics Factor Contributions (IoU={iou_threshold})",
        fontsize=16,
        fontweight="bold",
    )

    plt.tight_layout(rect=[0.05, 0.02, 0.98, 0.97])

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.5)
        print(f"Saved HF all-metrics comparison plot to {save_path}")

    return fig
