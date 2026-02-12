#!/usr/bin/env python3
"""
Visualize Phase 1 factor analysis results — all four OVD models.

Covers YOLO World, SAM3 (baseline models) **and** GroundingDINO, OWLv2
(HuggingFace models).  This script supersedes both
``visualize_phase1_results.py`` (YOLO/SAM3 only) and
``visualize_phase1_hf_results.py`` (GDino/OWLv2 only).

Supports both filename conventions produced by the two runners:
  • load_and_run.py      →  ph1_<model>_factor_analysis.json   (new)
  • run_factor_analysis* →  <model>_factor_analysis.json        (legacy)

All four models can live in **one** results directory (load_and_run.py
writes them all there by default) or be split across two directories
(legacy layout).  Use ``--results-dir`` for a unified dir, or combine
``--results-dir`` (baseline) with ``--hf-results-dir`` (HF models) for the
split layout.

Usage examples
--------------
# All four models, unified dir (load_and_run.py output):
    python experiments/scripts/visualization/visualize_phase1_results.py \\
        --results-dir experiments/results/my_run

# Legacy split layout (separate HF dir):
    python experiments/scripts/visualization/visualize_phase1_results.py \\
        --results-dir     experiments/results/phase1_factor_analysis \\
        --hf-results-dir  experiments/results/phase1_factor_analysis_hf

# Single model:
    python experiments/scripts/visualization/visualize_phase1_results.py \\
        --results-dir experiments/results/my_run --model yolo_world

# Skip spider plots:
    python experiments/scripts/visualization/visualize_phase1_results.py \\
        --results-dir experiments/results/my_run --no-spider

CLI flags
---------
--results-dir PATH     Directory for baseline (YOLO World / SAM3) JSON files.
                       Also used for HF models when --hf-results-dir is absent.
                       (default: experiments/results/phase1_factor_analysis)
--hf-results-dir PATH  Separate directory for GroundingDINO / OWLv2 JSON files.
                       If omitted, --results-dir is searched for all four models.
--output-dir PATH      Output directory for PNGs (default: <results-dir>/plots)
--model                yolo_world | sam3 | grounding_dino | owlv2 | all (default: all)
--iou T                IoU threshold for metric extraction (default: 0.5)
--metrics KEYS         Metrics to plot individually (default: map f1 precision recall)
--no-four-model        Skip the four-model comparison plots
--no-spider            Skip per-axis spider / summary-grid plots
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Path setup — __file__ is
#   experiments/scripts/visualization/visualize_phase1_results.py
# parent×4 = project root (AgVFM)
# ---------------------------------------------------------------------------
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Visualisation helpers live in the same directory as this script
viz_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(viz_dir))

from phase1_factor_analysis import (
    plot_factor_comparison,
    plot_factor_contributions,
    plot_all_metrics_factor_contributions,
    plot_metric_shift_radar,
    plot_axis_spider_comparison,
    plot_spider_summary_grid,
)

from phase1_factor_analysis_hf import (
    plot_hf_model_comparison,
    plot_hf_comparison_all_metrics,
    plot_four_model_comparison,
)

from agvfm.config.experiments import FACTOR_AXES


# ---------------------------------------------------------------------------
# Model registry — each entry lists candidate filenames, new prefix first
# ---------------------------------------------------------------------------
_MODELS = [
    {
        "key":      "yolo_world",
        "label":    "YOLO World",
        "group":    "baseline",
        "filenames": [
            "ph1_yolo_world_factor_analysis.json",   # load_and_run.py
            "yolo_world_factor_analysis.json",        # legacy run_factor_analysis.py
        ],
    },
    {
        "key":      "sam3",
        "label":    "SAM3",
        "group":    "baseline",
        "filenames": [
            "ph1_sam3_factor_analysis.json",          # load_and_run.py
            "sam3_factor_analysis.json",              # legacy run_factor_analysis.py
        ],
    },
    {
        "key":      "grounding_dino",
        "label":    "GroundingDINO",
        "group":    "hf",
        "filenames": [
            "ph1_grounding_dino_factor_analysis.json",  # load_and_run.py
            "grounding_dino_factor_analysis.json",       # legacy run_factor_analysis_hf.py
        ],
    },
    {
        "key":      "owlv2",
        "label":    "OWLv2",
        "group":    "hf",
        "filenames": [
            "ph1_owlv2_factor_analysis.json",   # load_and_run.py
            "owlv2_factor_analysis.json",        # legacy run_factor_analysis_hf.py
        ],
    },
]

_ALL_KEYS = [m["key"] for m in _MODELS]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _try_load(search_dirs: List[Path], model_info: dict) -> Optional[Dict]:
    """
    Try each candidate filename for *model_info* across all *search_dirs*.
    Returns the first match found, or None.
    """
    for fname in model_info["filenames"]:
        for d in search_dirs:
            path = d / fname
            if path.exists():
                with open(path) as fh:
                    data = json.load(fh)
                n = len(data.get("results", {}))
                print(f"✅  Loaded {model_info['label']} ({n} configs) ← {path.name}")
                return data

    tried = ", ".join(model_info["filenames"])
    dirs  = ", ".join(str(d) for d in search_dirs)
    print(f"⚠️   {model_info['label']} not found (tried: {tried} in {dirs})")
    return None


def _plot_single_model(
    results: Dict,
    model_tag: str,
    output_dir: Path,
    iou: float,
    metrics: List[str],
) -> None:
    """Generate all-metrics panel, per-metric plots, and radar for one model."""
    label = results.get("model", model_tag)
    print(f"\n  ── {label} ──")

    plot_all_metrics_factor_contributions(
        results,
        iou_threshold=iou,
        save_path=output_dir / f"{model_tag}_factor_contributions_all_metrics.png",
    )

    for metric in metrics:
        plot_factor_contributions(
            results,
            metric=metric,
            iou_threshold=iou,
            save_path=output_dir / f"{model_tag}_factor_contributions_{metric}.png",
        )

    plot_metric_shift_radar(
        results,
        iou_threshold=iou,
        save_path=output_dir / f"{model_tag}_metric_shift_radar.png",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize Phase 1 factor analysis — all four OVD models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--results-dir", default=None,
        help="Directory for YOLO World / SAM3 JSON files "
             "(also searched for HF models when --hf-results-dir is absent).",
    )
    p.add_argument(
        "--hf-results-dir", default=None,
        help="Separate directory for GroundingDINO / OWLv2 JSON files "
             "(legacy split-dir layout; omit when all models share one dir).",
    )
    p.add_argument(
        "--output-dir", default=None,
        help="Output directory for PNG files (default: <results-dir>/plots).",
    )
    p.add_argument(
        "--model", default="all",
        choices=_ALL_KEYS + ["all"],
        help="Which model(s) to visualize.",
    )
    p.add_argument(
        "--iou", type=float, default=0.5, metavar="T",
        help="IoU threshold for metric extraction.",
    )
    p.add_argument(
        "--metrics", nargs="+",
        default=["map", "f1", "precision", "recall"],
        choices=["map", "f1", "precision", "recall"],
        help="Metrics to plot individually.",
    )
    p.add_argument(
        "--no-four-model", action="store_true",
        help="Skip the four-model comparison plots.",
    )
    p.add_argument(
        "--no-spider", action="store_true",
        help="Skip per-axis spider / summary-grid plots.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ── Resolve directories ──────────────────────────────────────────────────
    baseline_dir = (
        Path(args.results_dir)
        if args.results_dir
        else project_root / "experiments" / "results" / "phase1_factor_analysis"
    )
    hf_dir = Path(args.hf_results_dir) if args.hf_results_dir else baseline_dir
    output_dir = Path(args.output_dir) if args.output_dir else baseline_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Phase 1 — Factor Analysis Visualization (all four OVD models)")
    print("=" * 70)
    print(f"Baseline dir : {baseline_dir}")
    print(f"HF dir       : {hf_dir}")
    print(f"Output dir   : {output_dir}")
    print(f"IoU          : {args.iou}")
    print(f"Metrics      : {args.metrics}")
    print()

    # ── Filter to requested model(s) ────────────────────────────────────────
    model_defs = (
        _MODELS if args.model == "all"
        else [m for m in _MODELS if m["key"] == args.model]
    )

    # ── Load all requested models ────────────────────────────────────────────
    print("Loading results …")
    loaded: Dict[str, Dict] = {}
    for mdef in model_defs:
        # Baseline models search baseline_dir; HF models also check hf_dir
        search = [baseline_dir] if mdef["group"] == "baseline" else [hf_dir, baseline_dir]
        data = _try_load(search, mdef)
        if data is not None:
            loaded[mdef["key"]] = data

    if not loaded:
        print("\n❌  No results found — nothing to plot.")
        sys.exit(1)

    # ── Per-model plots ──────────────────────────────────────────────────────
    print("\nGenerating per-model plots …")
    for mdef in model_defs:
        key = mdef["key"]
        if key not in loaded:
            continue
        _plot_single_model(loaded[key], key, output_dir, args.iou, args.metrics)

    # ── Baseline pair comparison (YOLO World vs SAM3) ────────────────────────
    yolo  = loaded.get("yolo_world")
    sam3  = loaded.get("sam3")
    gdino = loaded.get("grounding_dino")
    owlv2 = loaded.get("owlv2")

    if yolo and sam3:
        print("\nGenerating YOLO World vs SAM3 comparison plots …")
        for metric in args.metrics:
            plot_factor_comparison(
                yolo, sam3,
                metric=metric,
                iou_threshold=args.iou,
                save_path=output_dir / f"baseline_comparison_{metric}.png",
            )

    # ── HF pair comparison (GroundingDINO vs OWLv2) ──────────────────────────
    if gdino and owlv2:
        print("\nGenerating GroundingDINO vs OWLv2 comparison plots …")

        plot_hf_comparison_all_metrics(
            gdino, owlv2,
            iou_threshold=args.iou,
            save_path=output_dir / "hf_comparison_all_metrics.png",
        )

        for metric in args.metrics:
            plot_hf_model_comparison(
                gdino, owlv2,
                metric=metric,
                iou_threshold=args.iou,
                save_path=output_dir / f"hf_comparison_{metric}.png",
            )

    # ── Four-model comparison ────────────────────────────────────────────────
    if not args.no_four_model:
        if yolo and sam3 and gdino and owlv2:
            print("\nGenerating four-model comparison plots …")
            for metric in args.metrics:
                plot_four_model_comparison(
                    yolo, sam3, gdino, owlv2,
                    metric=metric,
                    iou_threshold=args.iou,
                    save_path=output_dir / f"four_model_comparison_{metric}.png",
                )
        else:
            missing = [
                label for key, label in [
                    ("yolo_world", "YOLO World"), ("sam3", "SAM3"),
                    ("grounding_dino", "GroundingDINO"), ("owlv2", "OWLv2"),
                ]
                if key not in loaded
            ]
            if missing:
                print(
                    f"\n  ⚠️  Skipping four-model comparison — missing: {', '.join(missing)}"
                )

    # ── Spider / radar comparison plots ─────────────────────────────────────
    if not args.no_spider:
        # Build label → results map from whatever was loaded
        label_map = {
            mdef["label"]: loaded[mdef["key"]]
            for mdef in _MODELS
            if mdef["key"] in loaded
        }

        if len(label_map) >= 2:
            print("\nGenerating per-axis spider comparison plots …")
            for axis in FACTOR_AXES:
                plot_axis_spider_comparison(
                    label_map,
                    axis.name,
                    metrics=["map", "f1"],
                    iou_threshold=args.iou,
                    save_dir=output_dir,
                    figsize=(9, 9),
                )

            print("Generating spider summary grid …")
            plot_spider_summary_grid(
                label_map,
                metrics=["map", "f1"],
                iou_threshold=args.iou,
                save_dir=output_dir,
            )
        else:
            print("\n  ⚠️  Only one model loaded — skipping spider comparison (need ≥2).")

    print()
    print("=" * 70)
    print("✅  All plots saved.")
    print(f"📊  Output directory: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
