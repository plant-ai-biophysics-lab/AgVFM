#!/usr/bin/env python3
"""
Visualize Phase 1 factor analysis results — HuggingFace OVD models.

Produces the same set of plots as ``visualize_phase1_results.py`` for
GroundingDINO and OWLv2, plus HF-specific comparison plots and an optional
four-model overview that includes YOLO World and SAM3.

Usage examples
--------------
# Both HF models only (default HF results dir):
    python experiments/scripts/visualization/visualize_phase1_hf_results.py

# Explicit HF results directory:
    python experiments/scripts/visualization/visualize_phase1_hf_results.py \\
        --hf-results-dir /path/to/phase1_factor_analysis_hf

# Include YOLO World / SAM3 for four-model comparison:
    python experiments/scripts/visualization/visualize_phase1_hf_results.py \\
        --baseline-results-dir experiments/results/phase1_factor_analysis

# All options:
    python experiments/scripts/visualization/visualize_phase1_hf_results.py \\
        --hf-results-dir     experiments/results/phase1_factor_analysis_hf \\
        --baseline-results-dir experiments/results/phase1_factor_analysis \\
        --output-dir         experiments/results/phase1_factor_analysis_hf/plots \\
        --iou                0.5 \\
        --metrics            map f1 precision recall

CLI flags
---------
--hf-results-dir        Directory containing grounding_dino_factor_analysis.json
                        and owlv2_factor_analysis.json.
                        (default: experiments/results/phase1_factor_analysis_hf
                                  relative to project root)
--baseline-results-dir  Directory containing yolo_world_factor_analysis.json
                        and sam3_factor_analysis.json.  Required only for the
                        four-model comparison plots; silently skipped if absent.
                        (default: experiments/results/phase1_factor_analysis)
--output-dir            Where to write PNG files.
                        (default: <hf-results-dir>/plots)
--iou                   IoU threshold to use for all plots (default: 0.5)
--metrics               One or more metric keys to plot individually.
                        (default: map f1 precision recall)
--no-four-model         Skip the four-model comparison even if baseline results
                        are available.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Path setup — __file__ is
#   experiments/scripts/visualization/visualize_phase1_hf_results.py
# parent×4 = project root (AgVFM)
# ---------------------------------------------------------------------------
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Visualisation modules live in the same directory as this script
viz_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(viz_dir))

# Reused single-model helpers (identical schema → no adaptation needed)
from phase1_factor_analysis import (
    plot_factor_contributions,
    plot_all_metrics_factor_contributions,
    plot_metric_shift_radar,
    plot_axis_spider_comparison,
    plot_spider_summary_grid,
)

# HF-specific comparison helpers
from phase1_factor_analysis_hf import (
    plot_hf_model_comparison,
    plot_hf_comparison_all_metrics,
    plot_four_model_comparison,
)

# Factor axis definitions (to iterate when making spider charts)
from agvfm.config.experiments import FACTOR_AXES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Optional[Dict]:
    """Load a JSON file; return None (with a warning) if missing."""
    if not path.exists():
        print(f"⚠️  Not found — skipping: {path}")
        return None
    with open(path) as fh:
        data = json.load(fh)
    n = len(data.get("results", {}))
    model = data.get("model", path.stem)
    print(f"✅  Loaded {model} results ({n} configurations) from {path.name}")
    return data


def _plot_single_model(
    results: Dict,
    model_tag: str,
    output_dir: Path,
    iou: float,
    metrics: List[str],
) -> None:
    """Generate per-model individual-metric, all-metrics, and radar plots."""
    print(f"\n  ── {results.get('model', model_tag)} ──")

    # All-metrics combined panel
    plot_all_metrics_factor_contributions(
        results,
        iou_threshold=iou,
        save_path=output_dir / f"{model_tag}_factor_contributions_all_metrics.png",
    )

    # Per-metric individual plots
    for metric in metrics:
        plot_factor_contributions(
            results,
            metric=metric,
            iou_threshold=iou,
            save_path=output_dir / f"{model_tag}_factor_contributions_{metric}.png",
        )

    # Radar chart — metric shifts from baseline
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
        description="Visualize Phase 1 factor analysis for HuggingFace OVD models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--hf-results-dir",
        default=None,
        help="Directory with grounding_dino_factor_analysis.json and "
             "owlv2_factor_analysis.json.",
    )
    p.add_argument(
        "--baseline-results-dir",
        default=None,
        help="Directory with yolo_world_factor_analysis.json and "
             "sam3_factor_analysis.json (used for four-model comparison).",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for PNG files (default: <hf-results-dir>/plots).",
    )
    p.add_argument(
        "--iou",
        type=float,
        default=0.5,
        metavar="T",
        help="IoU threshold for metric extraction.",
    )
    p.add_argument(
        "--metrics",
        nargs="+",
        default=["map", "f1", "precision", "recall"],
        choices=["map", "f1", "precision", "recall"],
        help="Metrics to plot individually.",
    )
    p.add_argument(
        "--no-four-model",
        action="store_true",
        help="Skip four-model comparison plots.",
    )
    p.add_argument(
        "--no-spider",
        action="store_true",
        help="Skip per-axis spider/radar comparison plots (default: generate).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Resolve directories
    # ------------------------------------------------------------------
    if args.hf_results_dir:
        hf_results_dir = Path(args.hf_results_dir)
    else:
        hf_results_dir = (
            project_root / "experiments" / "results" / "phase1_factor_analysis_hf"
        )

    if args.baseline_results_dir:
        baseline_results_dir = Path(args.baseline_results_dir)
    else:
        baseline_results_dir = (
            project_root / "experiments" / "results" / "phase1_factor_analysis"
        )

    output_dir = Path(args.output_dir) if args.output_dir else hf_results_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Phase 1 (HF) — Factor Analysis Visualization")
    print("=" * 70)
    print(f"HF results dir      : {hf_results_dir}")
    print(f"Baseline results dir: {baseline_results_dir}")
    print(f"Output dir          : {output_dir}")
    print(f"IoU threshold       : {args.iou}")
    print(f"Metrics             : {args.metrics}")
    print()

    # ------------------------------------------------------------------
    # Load HF results
    # ------------------------------------------------------------------
    print("Loading HF model results …")
    gdino_results = _load_json(hf_results_dir / "grounding_dino_factor_analysis.json")
    owlv2_results = _load_json(hf_results_dir / "owlv2_factor_analysis.json")

    if not gdino_results and not owlv2_results:
        print("\n❌  No HF results found — nothing to plot.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Per-model plots (reusing phase1_factor_analysis functions directly)
    # ------------------------------------------------------------------
    print("\nGenerating per-model plots …")

    if gdino_results:
        _plot_single_model(gdino_results, "grounding_dino", output_dir, args.iou, args.metrics)

    if owlv2_results:
        _plot_single_model(owlv2_results, "owlv2", output_dir, args.iou, args.metrics)

    # ------------------------------------------------------------------
    # HF model comparison (GDino vs OWLv2)
    # ------------------------------------------------------------------
    if gdino_results and owlv2_results:
        print("\nGenerating HF model comparison plots …")

        # All-metrics side-by-side (single large figure)
        plot_hf_comparison_all_metrics(
            gdino_results,
            owlv2_results,
            iou_threshold=args.iou,
            save_path=output_dir / "hf_comparison_all_metrics.png",
        )

        # Per-metric side-by-side
        for metric in args.metrics:
            plot_hf_model_comparison(
                gdino_results,
                owlv2_results,
                metric=metric,
                iou_threshold=args.iou,
                save_path=output_dir / f"hf_comparison_{metric}.png",
            )

    # ------------------------------------------------------------------
    # Four-model comparison (YOLO World + SAM3 + GDino + OWLv2)
    # ------------------------------------------------------------------
    if not args.no_four_model:
        print("\nAttempting four-model comparison …")
        yolo_results = _load_json(baseline_results_dir / "yolo_world_factor_analysis.json")
        sam3_results  = _load_json(baseline_results_dir / "sam3_factor_analysis.json")

        if yolo_results and sam3_results and gdino_results and owlv2_results:
            print("Generating four-model comparison plots …")
            for metric in args.metrics:
                plot_four_model_comparison(
                    yolo_results,
                    sam3_results,
                    gdino_results,
                    owlv2_results,
                    metric=metric,
                    iou_threshold=args.iou,
                    save_path=output_dir / f"four_model_comparison_{metric}.png",
                )
        else:
            missing = []
            if not yolo_results:
                missing.append("YOLO World")
            if not sam3_results:
                missing.append("SAM3")
            if not gdino_results:
                missing.append("GroundingDINO")
            if not owlv2_results:
                missing.append("OWLv2")
            print(
                f"  ⚠️  Skipping four-model comparison — missing: {', '.join(missing)}\n"
                f"       Pass --baseline-results-dir to include YOLO World / SAM3."
            )

    # ------------------------------------------------------------------
    # Per-axis spider/radar charts overlaying available models
    # ------------------------------------------------------------------
    if not args.no_spider:
        print("\nGenerating per-axis spider/radar comparison plots …")

        # Load baseline YOLO/SAM results if available (may be None)
        yolo_results = _load_json(baseline_results_dir / "yolo_world_factor_analysis.json")
        sam3_results = _load_json(baseline_results_dir / "sam3_factor_analysis.json")

        # Build models map: prefer readable labels
        models_map = {}
        if yolo_results:
            models_map["YOLO World"] = yolo_results
        if sam3_results:
            models_map["SAM3"] = sam3_results
        if gdino_results:
            models_map["GroundingDINO"] = gdino_results
        if owlv2_results:
            models_map["OWLv2"] = owlv2_results

        if models_map:
            # Create spider charts for each axis and metric (map, f1)
            for axis in FACTOR_AXES:
                # Use output_dir defined earlier
                plot_axis_spider_comparison(
                    models_map,
                    axis.name,
                    metrics=["map", "f1"],
                    iou_threshold=args.iou,
                    save_dir=output_dir,
                    figsize=(9, 9),
                )

            # 2×8 summary grid: all axes × both metrics in one figure
            plot_spider_summary_grid(
                models_map,
                metrics=["map", "f1"],
                iou_threshold=args.iou,
                save_dir=output_dir,
            )
        else:
            print("⚠️  No model results available for spider comparison — skipping.")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("✅  All plots saved.")
    print(f"📊  Output directory: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
