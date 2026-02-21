#!/usr/bin/env python3
"""
Visualize Phase 1 factor analysis results.

Usage:
    python experiments/scripts/visualization/visualize_phase1_results.py
"""

import json
import sys
from pathlib import Path

# Add project root to path
# __file__ is experiments/scripts/visualization/visualize_phase1_results.py
# So parent.parent.parent.parent is the actual project root (AgVFM2)
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import from local module (experiment-specific visualization code)
from phase1_factor_analysis import (
    plot_factor_comparison,
    plot_factor_contributions,
    plot_all_metrics_factor_contributions,
)


def main():
    results_dir = project_root / "experiments" / "results" / "phase1_factor_analysis"
    output_dir = results_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    yolo_file = results_dir / "yolo_world_factor_analysis.json"
    sam3_file = results_dir / "sam3_factor_analysis.json"
    
    yolo_results = None
    sam3_results = None
    
    if yolo_file.exists():
        with open(yolo_file) as f:
            yolo_results = json.load(f)
        print(f"✅ Loaded YOLO World results ({len(yolo_results.get('results', {}))} configs)")
    else:
        print(f"⚠️  YOLO World results not found: {yolo_file}")
    
    if sam3_file.exists():
        with open(sam3_file) as f:
            sam3_results = json.load(f)
        print(f"✅ Loaded SAM3 results ({len(sam3_results.get('results', {}))} configs)")
    else:
        print(f"⚠️  SAM3 results not found: {sam3_file}")
    
    # Generate plots
    if yolo_results:
        print("\nGenerating YOLO World plots...")
        
        # All metrics in one plot (mAP, F1, Precision, Recall)
        plot_all_metrics_factor_contributions(
            yolo_results,
            iou_threshold=0.5,
            save_path=output_dir / "yolo_world_factor_contributions_all_metrics.png",
        )
        
        # Individual metric plots (for detailed viewing)
        for metric, label in [("map", "mAP"), ("f1", "F1"), ("precision", "Precision"), ("recall", "Recall")]:
            plot_factor_contributions(
                yolo_results,
                metric=metric,
                iou_threshold=0.5,
                save_path=output_dir / f"yolo_world_factor_contributions_{metric}.png",
            )
    
    if sam3_results:
        print("\nGenerating SAM3 plots...")
        
        # All metrics in one plot (mAP, F1, Precision, Recall)
        plot_all_metrics_factor_contributions(
            sam3_results,
            iou_threshold=0.5,
            save_path=output_dir / "sam3_factor_contributions_all_metrics.png",
        )
        
        # Individual metric plots (for detailed viewing)
        for metric, label in [("map", "mAP"), ("f1", "F1"), ("precision", "Precision"), ("recall", "Recall")]:
            plot_factor_contributions(
                sam3_results,
                metric=metric,
                iou_threshold=0.5,
                save_path=output_dir / f"sam3_factor_contributions_{metric}.png",
            )
    
    if yolo_results and sam3_results:
        print("\nGenerating comparison plots...")
        
        # Side-by-side comparison for mAP
        plot_factor_comparison(
            yolo_results,
            sam3_results,
            metric="map",
            iou_threshold=0.5,
            save_path=output_dir / "factor_comparison_map.png",
        )
        
        # Side-by-side comparison for F1
        plot_factor_comparison(
            yolo_results,
            sam3_results,
            metric="f1",
            iou_threshold=0.5,
            save_path=output_dir / "factor_comparison_f1.png",
        )
    
    print(f"\n✅ All plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
