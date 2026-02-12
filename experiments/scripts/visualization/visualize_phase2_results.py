#!/usr/bin/env python3
"""
Visualize Phase 2 combination and absorber results.

Usage:
    python experiments/scripts/visualization/visualize_phase2_results.py
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import from local module (experiment-specific visualization code)
from phase2_combinations import (
    plot_absorber_comparison,
    plot_all_metrics_combinations,
    plot_combination_performance,
    plot_precision_recall_tradeoff,
    plot_combo_component_spider,
    plot_combo_summary_grid,
)


def main():
    parser = argparse.ArgumentParser(description="Visualize Phase 2 combination and absorber results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory containing phase2 results JSON files "
             "(default: experiments/results/phase2_combinations relative to project root)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Show only the top N prompts per plot (baseline always included). "
             "Set to 0 to show all. (default: 20)",
    )
    parser.add_argument(
        "--no-spider",
        action="store_true",
        help="Skip per-component spider/radar comparison plots (default: generate).",
    )
    args = parser.parse_args()

    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        results_dir = project_root / "experiments" / "results" / "phase2_combinations"

    top_n = args.top_n if args.top_n > 0 else None

    output_dir = results_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    yolo_comb_file = results_dir / "yolo_world_combinations.json"
    yolo_abs_file = results_dir / "yolo_world_absorbers.json"
    sam3_comb_file = results_dir / "sam3_combinations.json"
    
    yolo_comb_results = None
    yolo_abs_results = None
    sam3_comb_results = None
    
    if yolo_comb_file.exists():
        with open(yolo_comb_file) as f:
            yolo_comb_results = json.load(f)
        print(f"✅ Loaded YOLO World combination results ({len(yolo_comb_results.get('results', {}))} configs)")
    else:
        print(f"⚠️  YOLO World combination results not found: {yolo_comb_file}")
    
    if yolo_abs_file.exists():
        with open(yolo_abs_file) as f:
            yolo_abs_results = json.load(f)
        print(f"✅ Loaded YOLO World absorber results ({len(yolo_abs_results.get('results', {}))} configs)")
    else:
        print(f"⚠️  YOLO World absorber results not found: {yolo_abs_file}")
    
    if sam3_comb_file.exists():
        with open(sam3_comb_file) as f:
            sam3_comb_results = json.load(f)
        print(f"✅ Loaded SAM3 combination results ({len(sam3_comb_results.get('results', {}))} configs)")
    else:
        print(f"⚠️  SAM3 combination results not found: {sam3_comb_file}")
    
    if not yolo_comb_results and not sam3_comb_results:
        print("\n❌ No combination results found. Cannot generate plots.")
        return
    
    print("\n" + "="*80)
    print("Generating Phase 2 Visualizations")
    print("="*80)
    
    # Generate YOLO World visualizations
    if yolo_comb_results:
        print("\n" + "-"*80)
        print("YOLO World Visualizations")
        print("-"*80)
        
        # 1. All metrics combination performance (2x2 grid)
        print("\n1. Generating all-metrics combination plot...")
        plot_all_metrics_combinations(
            yolo_comb_results,
            iou_threshold=0.5,
            save_path=output_dir / "yolo_world_combinations_all_metrics.png",
            baseline_config="comb_species",
            top_n=top_n,
        )
        
        # 2. Individual metric plots for combinations
        print("\n2. Generating individual metric plots...")
        for metric, label in [("map", "mAP"), ("f1", "F1"), ("precision", "Precision"), ("recall", "Recall")]:
            print(f"   {label}...")
            plot_combination_performance(
                yolo_comb_results,
                metric=metric,
                iou_threshold=0.5,
                save_path=output_dir / f"yolo_world_combinations_{metric}.png",
                baseline_config="comb_species",
                top_n=top_n,
            )
        
        # 3. Absorber comparison (if absorber results available)
        if yolo_abs_results:
            abs_keys = [k for k in yolo_abs_results.get("results", {}) if k.startswith("abs_")]
            if not abs_keys:
                print("\n3. Skipping absorber comparison — no 'abs_*' keys found in absorber results.")
            else:
                print("\n3. Generating absorber comparison plots...")
                for metric, label in [("f1", "F1"), ("map", "mAP"), ("precision", "Precision"), ("recall", "Recall")]:
                    print(f"   {label}...")
                    plot_absorber_comparison(
                        yolo_comb_results,
                        yolo_abs_results,
                        metric=metric,
                        iou_threshold=0.5,
                        save_path=output_dir / f"yolo_world_absorber_comparison_{metric}.png",
                    )
        
        # 4. Precision-Recall tradeoff plot
        print("\n4. Generating Precision-Recall tradeoff plot...")
        plot_precision_recall_tradeoff(
            yolo_comb_results,
            absorber_results=yolo_abs_results,
            iou_threshold=0.5,
            save_path=output_dir / "yolo_world_precision_recall_tradeoff.png",
        )
    
    # Generate SAM3 visualizations
    if sam3_comb_results:
        print("\n" + "-"*80)
        print("SAM3 Visualizations")
        print("-"*80)
        
        # 1. All metrics combination performance (2x2 grid)
        print("\n1. Generating all-metrics combination plot...")
        plot_all_metrics_combinations(
            sam3_comb_results,
            iou_threshold=0.5,
            save_path=output_dir / "sam3_combinations_all_metrics.png",
            baseline_config="comb_species",
            top_n=top_n,
        )
        
        # 2. Individual metric plots for combinations
        print("\n2. Generating individual metric plots...")
        for metric, label in [("map", "mAP"), ("f1", "F1"), ("precision", "Precision"), ("recall", "Recall")]:
            print(f"   {label}...")
            plot_combination_performance(
                sam3_comb_results,
                metric=metric,
                iou_threshold=0.5,
                save_path=output_dir / f"sam3_combinations_{metric}.png",
                baseline_config="comb_species",
                top_n=top_n,
            )
        
        # 3. Precision-Recall tradeoff plot
        print("\n3. Generating Precision-Recall tradeoff plot...")
        plot_precision_recall_tradeoff(
            sam3_comb_results,
            absorber_results=None,  # SAM3 doesn't have absorber results
            iou_threshold=0.5,
            save_path=output_dir / "sam3_precision_recall_tradeoff.png",
        )

    # Per-component spider/radar charts overlaying YOLO World and SAM3
    if not args.no_spider:
        models_map = {}
        if yolo_comb_results:
            models_map["YOLO World"] = yolo_comb_results
        if sam3_comb_results:
            models_map["SAM3"] = sam3_comb_results

        if models_map:
            print("\n" + "-"*80)
            print("Spider / Radar Charts — prompt component contributions")
            print("-"*80)
            plot_combo_component_spider(
                models_map,
                metrics=["map", "f1"],
                iou_threshold=0.5,
                baseline_config="comb_species",
                save_dir=output_dir,
                figsize=(9, 9),
            )
            # mAP summary grid: all 5 components in one row
            plot_combo_summary_grid(
                models_map,
                iou_threshold=0.5,
                baseline_config="comb_species",
                save_dir=output_dir,
            )
        else:
            print("\n⚠️  No model results available for spider comparison — skipping.")

    print("\n" + "="*80)
    print("✅ All visualizations generated!")
    print(f"📊 Plots saved to: {output_dir}")
    print("="*80)

if __name__ == "__main__":
    main()
