#!/usr/bin/env python3
"""
Generate visualizations for Phase 3 confidence threshold sweep results.

This script generates plots for both YOLO World and SAM3 confidence sweep results:
- Performance curves (mAP, F1, Precision, Recall vs confidence threshold)
- Precision-Recall tradeoff
- Detection volume analysis
- Sensitivity analysis
- Config comparison plots

Usage:
    python experiments/scripts/visualization/visualize_phase3_results.py
    python experiments/scripts/visualization/visualize_phase3_results.py --model yolo_world
    python experiments/scripts/visualization/visualize_phase3_results.py --model sam3
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.scripts.visualization.phase3_confidence_sweeps import (
    extract_sweep_data,
    load_confidence_sweep_results,
    plot_config_comparison,
    plot_detection_volume,
    plot_detection_volume_linear,
    plot_performance_curves,
    plot_precision_recall_tradeoff,
    plot_sensitivity_analysis,
)


def generate_visualizations(
    model_name: str,
    results_file: Path,
    output_dir: Path,
    n_images: int = 178,
):
    """Generate all visualizations for a model's confidence sweep results."""
    print(f"\n{'='*80}")
    print(f"Generating Phase 3 Visualizations: {model_name}")
    print(f"{'='*80}")
    
    # Load results
    print(f"\nLoading results from: {results_file}")
    results = load_confidence_sweep_results(results_file)
    
    # Extract sweep data
    sweep_data = extract_sweep_data(results)
    print(f"Found {len(sweep_data)} configs: {list(sweep_data.keys())}")
    
    if not sweep_data:
        print(f"⚠️  No sweep data found in {results_file}. Skipping.")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Format model name for titles
    model_title = model_name.replace("_", " ").title()
    
    # 1. Performance curves (all metrics)
    print("\n1. Generating performance curves...")
    fig = plot_performance_curves(
        sweep_data=sweep_data,
        metrics=["map", "f1", "precision", "recall"],
        save_path=output_dir / f"{model_name}_performance_curves.png",
        model_name=model_title,
    )
    plt.close(fig)
    
    # 2. Precision-Recall tradeoff
    print("2. Generating Precision-Recall tradeoff plot...")
    fig = plot_precision_recall_tradeoff(
        sweep_data=sweep_data,
        save_path=output_dir / f"{model_name}_precision_recall_tradeoff.png",
        model_name=model_title,
    )
    plt.close(fig)
    
    # 3. Detection volume (log scale)
    print("3. Generating detection volume plot (log scale)...")
    fig = plot_detection_volume(
        sweep_data=sweep_data,
        save_path=output_dir / f"{model_name}_detection_volume.png",
        model_name=model_title,
        n_images=n_images,
    )
    plt.close(fig)
    
    # 3b. Detection volume (linear scale)
    print("3b. Generating detection volume plot (linear scale)...")
    fig = plot_detection_volume_linear(
        sweep_data=sweep_data,
        save_path=output_dir / f"{model_name}_detection_volume_linear.png",
        model_name=model_title,
        n_images=n_images,
    )
    plt.close(fig)
    
    # 4. Sensitivity analysis
    print("4. Generating sensitivity analysis...")
    fig = plot_sensitivity_analysis(
        sweep_data=sweep_data,
        save_path=output_dir / f"{model_name}_sensitivity_analysis.png",
        model_name=model_title,
    )
    plt.close(fig)
    
    # 5. Config comparison for each metric
    print("5. Generating config comparison plots...")
    for metric in ["map", "f1", "precision", "recall"]:
        fig = plot_config_comparison(
            sweep_data=sweep_data,
            metric=metric,
            save_path=output_dir / f"{model_name}_config_comparison_{metric}.png",
            model_name=model_title,
        )
        plt.close(fig)
    
    print(f"\n✅ All visualizations saved to: {output_dir}")
    print(f"   Generated {6 + len(['map', 'f1', 'precision', 'recall'])} plots")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Phase 3 confidence sweep visualizations"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["yolo_world", "sam3", "all"],
        default="all",
        help="Model to visualize (default: all)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Results directory (default: experiments/results/phase3_confidence_sweeps)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for plots (default: results_dir/plots)",
    )
    parser.add_argument(
        "--n-images",
        type=int,
        default=178,
        help="Number of images in test set (for detection volume calculation)",
    )
    
    args = parser.parse_args()
    
    # Set up paths
    if args.results_dir is None:
        results_dir = project_root / "experiments" / "results" / "phase3_confidence_sweeps"
    else:
        results_dir = args.results_dir
    
    if args.output_dir is None:
        output_dir = results_dir / "plots"
    else:
        output_dir = args.output_dir
    
    # Determine which models to process
    models_to_process = []
    if args.model == "all":
        models_to_process = ["yolo_world", "sam3"]
    else:
        models_to_process = [args.model]
    
    # Generate visualizations for each model
    for model_name in models_to_process:
        results_file = results_dir / f"{model_name}_confidence_sweeps.json"
        
        if not results_file.exists():
            print(f"⚠️  Results file not found: {results_file}")
            print(f"   Skipping {model_name}")
            continue
        
        try:
            generate_visualizations(
                model_name=model_name,
                results_file=results_file,
                output_dir=output_dir,
                n_images=args.n_images,
            )
        except Exception as e:
            print(f"❌ Error generating visualizations for {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*80}")
    print("Phase 3 Visualization Complete")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
