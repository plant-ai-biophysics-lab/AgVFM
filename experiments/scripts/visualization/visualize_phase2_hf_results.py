#!/usr/bin/env python3
"""
Visualize Phase 2 combination results — HuggingFace OVD models.

Produces the same set of plots as ``visualize_phase2_results.py`` for
GroundingDINO and OWLv2, plus an optional side-by-side HF model comparison.

Note: Absorber-architecture plots are not produced here — absorbers require
YOLO World's multi-class API and are not available for HF text-grounded models.

Usage examples
--------------
# Default HF results directory:
    python experiments/scripts/visualization/visualize_phase2_hf_results.py

# Explicit results directory:
    python experiments/scripts/visualization/visualize_phase2_hf_results.py \\
        --hf-results-dir experiments/results/phase2_combinations_hf

# Include YOLO World / SAM3 for four-model P-R scatter comparison:
    python experiments/scripts/visualization/visualize_phase2_hf_results.py \\
        --baseline-results-dir experiments/results/phase2_combinations

# Limit each bar chart to the top 10 prompts:
    python experiments/scripts/visualization/visualize_phase2_hf_results.py \\
        --top-n 10

# Show all prompts (no truncation):
    python experiments/scripts/visualization/visualize_phase2_hf_results.py \\
        --top-n 0

CLI flags
---------
--hf-results-dir        Directory with grounding_dino_combinations.json
                        and owlv2_combinations.json.
                        (default: experiments/results/phase2_combinations_hf)
--baseline-results-dir  Directory with yolo_world_combinations.json and
                        sam3_combinations.json.  Used only for the four-model
                        P-R scatter; silently skipped if absent.
                        (default: experiments/results/phase2_combinations)
--output-dir            Where to write PNG files.
                        (default: <hf-results-dir>/plots)
--top-n N               Show only the top N prompts per ranked bar chart
                        (baseline always kept regardless of rank).
                        Set to 0 to show all. (default: 20)
--iou T                 IoU threshold for all metric extraction (default: 0.5)
--baseline-config KEY   Results key treated as the baseline in bar charts.
                        (default: comb_species)
--no-hf-comparison      Skip the side-by-side GDino vs OWLv2 comparison plots.
--no-four-model         Skip the four-model P-R scatter even when baseline
                        results are available.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Path setup — __file__ is
#   experiments/scripts/visualization/visualize_phase2_hf_results.py
# parent×4 = project root (AgVFM)
# ---------------------------------------------------------------------------
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Visualisation modules live in the same directory as this script
viz_dir_path = Path(__file__).resolve().parent
sys.path.insert(0, str(viz_dir_path))

from phase2_combinations import (
    plot_all_metrics_combinations,
    plot_combination_performance,
    plot_precision_recall_tradeoff,
    plot_combo_component_spider,
    plot_combo_summary_grid,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Optional[Dict]:
    """Load a results JSON; return None (with a warning) if missing or corrupt."""
    if not path.exists():
        print(f"⚠️  Not found — skipping: {path}")
        return None
    try:
        with open(path) as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"⚠️  Could not parse {path.name}: {exc}")
        return None
    n = len(data.get("results", {}))
    model = data.get("model", path.stem)
    print(f"✅  Loaded {model} ({n} configurations) from {path.name}")
    return data


def _plot_single_model(
    results: Dict,
    model_tag: str,
    output_dir: Path,
    iou: float,
    baseline_config: str,
    top_n: Optional[int],
) -> None:
    """
    Generate per-model visualizations:
      • all-metrics 2×2 grid
      • individual metric bar charts (mAP, F1, Precision, Recall)
      • precision-recall scatter
    """
    model_name = results.get("model", model_tag)
    print(f"\n  ── {model_name} ──")

    # 1. All-metrics 2×2 grid
    plot_all_metrics_combinations(
        results,
        iou_threshold=iou,
        save_path=output_dir / f"{model_tag}_combinations_all_metrics.png",
        baseline_config=baseline_config,
        top_n=top_n,
    )

    # 2. Individual metric bar charts
    for metric, label in [
        ("map",       "mAP"),
        ("f1",        "F1"),
        ("precision", "Precision"),
        ("recall",    "Recall"),
    ]:
        print(f"    {label} …")
        plot_combination_performance(
            results,
            metric=metric,
            iou_threshold=iou,
            save_path=output_dir / f"{model_tag}_combinations_{metric}.png",
            baseline_config=baseline_config,
            top_n=top_n,
        )

    # 3. Precision-Recall scatter (shows all points; no top_n)
    plot_precision_recall_tradeoff(
        results,
        absorber_results=None,
        iou_threshold=iou,
        save_path=output_dir / f"{model_tag}_precision_recall_tradeoff.png",
    )


def _plot_hf_comparison(
    gdino_results: Dict,
    owlv2_results: Dict,
    output_dir: Path,
    iou: float,
    baseline_config: str,
    top_n: Optional[int],
) -> None:
    """
    Side-by-side GDino vs OWLv2 bar charts for each metric.

    Each pair shares a y-axis so the comparison is fair.  We build a combined
    results dict containing only configs present in *both* models, ranked by
    the average of the two metric values, then plot them as adjacent bars.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    gdino_res = gdino_results.get("results", {})
    owlv2_res = owlv2_results.get("results", {})
    common_keys = sorted(set(gdino_res) & set(owlv2_res))

    if not common_keys:
        print("  ⚠️  No common configuration keys between GDino and OWLv2 — skipping comparison.")
        return

    _COLOR_GDINO = "#5B8DB8"   # steel blue
    _COLOR_OWLV2 = "#E07B39"   # warm orange

    iou_key = f"iou_{iou}"

    metrics_meta = [
        ("map",       "mAP"),
        ("f1",        "F1"),
        ("precision", "Precision"),
        ("recall",    "Recall"),
    ]

    for metric, label in metrics_meta:
        # Collect values for both models across common keys
        entries = []
        for key in common_keys:
            g_val = (
                gdino_res[key]
                .get("metrics_by_iou", {})
                .get(iou_key, {})
                .get(metric, 0.0)
            )
            o_val = (
                owlv2_res[key]
                .get("metrics_by_iou", {})
                .get(iou_key, {})
                .get(metric, 0.0)
            )
            entries.append({"key": key, "gdino": g_val, "owlv2": o_val,
                            "avg": (g_val + o_val) / 2.0})

        # Sort by average value descending; always keep baseline at top
        entries.sort(key=lambda x: x["avg"], reverse=True)

        if top_n is not None and len(entries) > top_n:
            top_entries = entries[:top_n]
            top_keys = {e["key"] for e in top_entries}
            if baseline_config not in top_keys:
                baseline_entry = next(
                    (e for e in entries if e["key"] == baseline_config), None
                )
                if baseline_entry is not None:
                    top_entries.append(baseline_entry)
                    top_entries.sort(key=lambda x: x["avg"], reverse=True)
            entries = top_entries

        names = [e["key"] for e in entries]
        gdino_vals = [e["gdino"] for e in entries]
        owlv2_vals = [e["owlv2"] for e in entries]

        # Baseline reference lines
        g_base = gdino_res.get(baseline_config, {}).get("metrics_by_iou", {}).get(iou_key, {}).get(metric, 0.0)
        o_base = owlv2_res.get(baseline_config, {}).get("metrics_by_iou", {}).get(iou_key, {}).get(metric, 0.0)

        n = len(names)
        bar_w = 0.38
        x = np.arange(n)

        n_shown  = len(entries)
        n_total  = len(common_keys)
        top_note = f" (top {n_shown} of {n_total})" if top_n is not None and n_shown < n_total else ""

        fig, ax = plt.subplots(figsize=(max(10, n * 1.3), 7))

        bars_g = ax.barh(x + bar_w / 2, gdino_vals, bar_w,
                         color=_COLOR_GDINO, alpha=0.8, label="GroundingDINO",
                         edgecolor="black", linewidth=0.4)
        bars_o = ax.barh(x - bar_w / 2, owlv2_vals, bar_w,
                         color=_COLOR_OWLV2, alpha=0.8, label="OWLv2",
                         edgecolor="black", linewidth=0.4)

        val_range = max(gdino_vals + owlv2_vals) - min(gdino_vals + owlv2_vals + [0])
        offset = val_range * 0.01 if val_range > 0 else 0.005
        for bar, val, base in zip(bars_g, gdino_vals, [g_base] * n):
            delta = val - base
            dstr = f"{delta:+.3f}" if abs(delta) > 1e-5 else ""
            txt = f"{val:.3f}" + (f"\n({dstr})" if dstr else "")
            ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
                    txt, va="center", ha="left", fontsize=7)
        for bar, val, base in zip(bars_o, owlv2_vals, [o_base] * n):
            delta = val - base
            dstr = f"{delta:+.3f}" if abs(delta) > 1e-5 else ""
            txt = f"{val:.3f}" + (f"\n({dstr})" if dstr else "")
            ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
                    txt, va="center", ha="left", fontsize=7)

        ax.axvline(g_base, color=_COLOR_GDINO, linestyle="--", linewidth=1.2,
                   alpha=0.7, label=f"GDino baseline ({g_base:.3f})")
        ax.axvline(o_base, color=_COLOR_OWLV2, linestyle="--", linewidth=1.2,
                   alpha=0.7, label=f"OWLv2 baseline ({o_base:.3f})")

        ax.set_yticks(x)
        ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel(f"{label}@IoU={iou}", fontsize=12, fontweight="bold")
        ax.set_title(
            f"HF Model Comparison: {label}@IoU={iou}{top_note}\n"
            f"Baseline: {baseline_config}",
            fontsize=14, fontweight="bold", pad=16,
        )
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(axis="x", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

        plt.tight_layout()
        save_path = output_dir / f"hf_comparison_{metric}.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.4)
        plt.close(fig)
        print(f"    Saved: {save_path}")


def _plot_four_model_pr(
    gdino_results: Dict,
    owlv2_results: Dict,
    yolo_results: Dict,
    sam3_results: Dict,
    iou: float,
    output_dir: Path,
) -> None:
    """
    Precision-Recall scatter showing all four models on the same axes.
    Points are coloured by model; config names are shown for key/extreme points.
    """
    import matplotlib.pyplot as plt

    _PALETTE = {
        "GroundingDINO": "#5B8DB8",
        "OWLv2":         "#E07B39",
        "YOLO World":    "#4CAF50",
        "SAM3":          "#9C27B0",
    }

    iou_key = f"iou_{iou}"

    fig, ax = plt.subplots(figsize=(9, 9))

    # Draw F1 iso-lines
    import numpy as np
    p_range = np.linspace(0.01, 1.0, 200)
    for f1_target in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        with np.errstate(divide="ignore", invalid="ignore"):
            r_range = f1_target * 2 * p_range / (2 * p_range - f1_target + 1e-10)
        r_range = np.clip(r_range, 0, 1)
        valid = (r_range > 0) & (r_range <= 1) & (p_range > 0) & (p_range <= 1)
        if np.any(valid):
            ax.plot(p_range[valid], r_range[valid], "--", color="gray",
                    alpha=0.25, linewidth=0.6)
            mid = len(p_range[valid]) // 2
            ax.text(p_range[valid][mid], r_range[valid][mid],
                    f"F1={f1_target:.1f}", fontsize=7, alpha=0.4, rotation=-40)

    model_results = {
        "GroundingDINO": gdino_results,
        "OWLv2":         owlv2_results,
        "YOLO World":    yolo_results,
        "SAM3":          sam3_results,
    }

    for model_label, res in model_results.items():
        if res is None:
            continue
        color = _PALETTE[model_label]
        res_dict = res.get("results", {})
        precs, recs = [], []
        for cfg_name, cfg_val in res_dict.items():
            m = cfg_val.get("metrics_by_iou", {}).get(iou_key, {})
            precs.append(m.get("precision", 0.0))
            recs.append(m.get("recall", 0.0))
        ax.scatter(precs, recs, c=color, s=60, alpha=0.6,
                   edgecolors="black", linewidths=0.5,
                   label=model_label, zorder=4)

    ax.set_xlabel("Precision", fontsize=12, fontweight="bold")
    ax.set_ylabel("Recall",    fontsize=12, fontweight="bold")
    ax.set_title(
        f"Four-Model Precision-Recall (Phase 2 Combinations)\nIoU={iou}",
        fontsize=13, fontweight="bold", pad=16,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.legend(fontsize=10, loc="lower left")

    plt.tight_layout()
    save_path = output_dir / "four_model_precision_recall_tradeoff.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.4)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize Phase 2 combination results for HuggingFace OVD models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--hf-results-dir",
        default=None,
        metavar="DIR",
        help="Directory with grounding_dino_combinations.json and "
             "owlv2_combinations.json. "
             "(default: experiments/results/phase2_combinations_hf)",
    )
    p.add_argument(
        "--baseline-results-dir",
        default=None,
        metavar="DIR",
        help="Directory with yolo_world_combinations.json and "
             "sam3_combinations.json. Used only for the four-model P-R scatter. "
             "(default: experiments/results/phase2_combinations)",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help="Output directory for PNG files. "
             "(default: <hf-results-dir>/plots)",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=20,
        metavar="N",
        help="Show only the top N prompts per ranked bar chart "
             "(baseline always included). Set to 0 to show all. (default: 20)",
    )
    p.add_argument(
        "--iou",
        type=float,
        default=0.5,
        metavar="T",
        help="IoU threshold for metric extraction.",
    )
    p.add_argument(
        "--baseline-config",
        default="comb_species",
        metavar="KEY",
        help="Results key treated as the baseline in bar charts.",
    )
    p.add_argument(
        "--no-hf-comparison",
        action="store_true",
        help="Skip the side-by-side GDino vs OWLv2 comparison bar charts.",
    )
    p.add_argument(
        "--no-four-model",
        action="store_true",
        help="Skip the four-model P-R scatter even when baseline results exist.",
    )
    p.add_argument(
        "--no-spider",
        action="store_true",
        help="Skip per-component spider/radar comparison plots (default: generate).",
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
    hf_results_dir = (
        Path(args.hf_results_dir)
        if args.hf_results_dir
        else project_root / "experiments" / "results" / "phase2_combinations_hf"
    )
    baseline_results_dir = (
        Path(args.baseline_results_dir)
        if args.baseline_results_dir
        else project_root / "experiments" / "results" / "phase2_combinations"
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else hf_results_dir / "plots"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    top_n = args.top_n if args.top_n > 0 else None

    print("=" * 70)
    print("Phase 2 (HF) — Combination Results Visualization")
    print("=" * 70)
    print(f"HF results dir      : {hf_results_dir}")
    print(f"Baseline results dir: {baseline_results_dir}")
    print(f"Output dir          : {output_dir}")
    print(f"IoU threshold       : {args.iou}")
    print(f"Baseline config     : {args.baseline_config}")
    print(f"Top-N per chart     : {top_n if top_n is not None else 'all'}")
    print()

    # ------------------------------------------------------------------
    # Load HF results
    # ------------------------------------------------------------------
    print("Loading HF model combination results …")
    gdino_results = _load_json(hf_results_dir / "grounding_dino_combinations.json")
    owlv2_results = _load_json(hf_results_dir / "owlv2_combinations.json")

    if not gdino_results and not owlv2_results:
        print("\n❌  No HF combination results found — nothing to plot.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Per-model plots
    # ------------------------------------------------------------------
    print("\nGenerating per-model plots …")

    if gdino_results:
        _plot_single_model(
            gdino_results,
            model_tag="grounding_dino",
            output_dir=output_dir,
            iou=args.iou,
            baseline_config=args.baseline_config,
            top_n=top_n,
        )

    if owlv2_results:
        _plot_single_model(
            owlv2_results,
            model_tag="owlv2",
            output_dir=output_dir,
            iou=args.iou,
            baseline_config=args.baseline_config,
            top_n=top_n,
        )

    # ------------------------------------------------------------------
    # HF side-by-side comparison (GDino vs OWLv2)
    # ------------------------------------------------------------------
    if not args.no_hf_comparison and gdino_results and owlv2_results:
        print("\nGenerating HF model comparison plots …")
        _plot_hf_comparison(
            gdino_results=gdino_results,
            owlv2_results=owlv2_results,
            output_dir=output_dir,
            iou=args.iou,
            baseline_config=args.baseline_config,
            top_n=top_n,
        )

    # ------------------------------------------------------------------
    # Four-model P-R scatter (optional; needs baseline results)
    # ------------------------------------------------------------------
    if not args.no_four_model:
        print("\nAttempting four-model P-R scatter …")
        yolo_results = _load_json(baseline_results_dir / "yolo_world_combinations.json")
        sam3_results  = _load_json(baseline_results_dir / "sam3_combinations.json")

        have_all = gdino_results and owlv2_results and yolo_results and sam3_results
        if have_all:
            _plot_four_model_pr(
                gdino_results=gdino_results,
                owlv2_results=owlv2_results,
                yolo_results=yolo_results,
                sam3_results=sam3_results,
                iou=args.iou,
                output_dir=output_dir,
            )
        else:
            missing = [
                name for name, res in [
                    ("GroundingDINO", gdino_results),
                    ("OWLv2",         owlv2_results),
                    ("YOLO World",    yolo_results),
                    ("SAM3",          sam3_results),
                ]
                if not res
            ]
            print(
                f"  ⚠️  Skipping four-model scatter — missing: {', '.join(missing)}\n"
                f"       Pass --baseline-results-dir to include YOLO World / SAM3."
            )

    # ------------------------------------------------------------------
    # Per-component spider/radar charts — overlay all available models
    # ------------------------------------------------------------------
    if not args.no_spider:
        # Load YOLO / SAM3 if not already loaded (four-model section may have)
        # We re-use whatever is in scope; for clarity load fresh locals here.
        _yolo_for_spider = _load_json(baseline_results_dir / "yolo_world_combinations.json")
        _sam3_for_spider = _load_json(baseline_results_dir / "sam3_combinations.json")

        models_map: Dict[str, Dict] = {}
        if gdino_results:
            models_map["GroundingDINO"] = gdino_results
        if owlv2_results:
            models_map["OWLv2"] = owlv2_results
        if _yolo_for_spider:
            models_map["YOLO World"] = _yolo_for_spider
        if _sam3_for_spider:
            models_map["SAM3"] = _sam3_for_spider

        if models_map:
            print("\nGenerating per-component spider/radar comparison plots …")
            plot_combo_component_spider(
                models_map,
                metrics=["map", "f1"],
                iou_threshold=args.iou,
                baseline_config=args.baseline_config,
                save_dir=output_dir,
                figsize=(9, 9),
            )
            # mAP summary grid: all 5 components in one row
            plot_combo_summary_grid(
                models_map,
                iou_threshold=args.iou,
                baseline_config=args.baseline_config,
                save_dir=output_dir,
            )
        else:
            print("  ⚠️  No model results available for spider comparison — skipping.")

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
