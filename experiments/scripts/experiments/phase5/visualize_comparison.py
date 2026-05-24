#!/usr/bin/env python3
"""
Compare Phase 5 results across different selection methods.

Generates comparison plots for:
- Metrics (F1, Precision, Recall) across methods
- Retention rates
- Empty image percentages
- Summary statistics
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import re

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
script_path = Path(__file__).resolve()
parts = script_path.parts
agvfm_idx = [i for i, part in enumerate(parts) if part == 'AgVFM2']
if agvfm_idx:
    project_root = Path(*parts[:agvfm_idx[0] + 1])
else:
    project_root = script_path.parents[6]
sys.path.insert(0, str(project_root))

from agvfm.models.yolo_world import YOLOWorldModel
from agvfm.data.labels import load_ground_truth
from agvfm.evaluation.metrics import compute_metrics_at_iou


def get_data_paths():
    """Get data paths using same logic as Phase 4."""
    import os
    script_path = Path(__file__).resolve()
    parts = script_path.parts
    agvfm_idx = [i for i, part in enumerate(parts) if part == 'AgVFM2']
    if agvfm_idx:
        project_root = Path(*parts[:agvfm_idx[0] + 1])
    else:
        project_root = script_path.parents[6]
    
    # Check DATA_ROOT environment variable first
    data_root = os.environ.get("DATA_ROOT")
    if data_root:
        agvfm_data = Path(data_root) / "_data" / "T4_REAL_FLOWER_SOBJ.v18-train_2023_test_location_genotype"
        if agvfm_data.exists():
            images_dir = agvfm_data / "test" / "images"
            labels_dir = agvfm_data / "test" / "labels"
            return images_dir, labels_dir
    
    # Try AgVFM2 first
    agvfm2_data = project_root / "_data" / "T4_REAL_FLOWER_SOBJ.v18-train_2023_test_location_genotype"
    if agvfm2_data.exists():
        images_dir = agvfm2_data / "test" / "images"
        labels_dir = agvfm2_data / "test" / "labels"
        return images_dir, labels_dir
    
    # Try direct path
    direct_path = Path("/data2/jmearles/AgVFM2/_data/T4_REAL_FLOWER_SOBJ.v18-train_2023_test_location_genotype")
    if direct_path.exists():
        images_dir = direct_path / "test" / "images"
        labels_dir = direct_path / "test" / "labels"
        return images_dir, labels_dir
    
    # Try AgVFM (sibling directory) as fallback
    agvfm_data = project_root.parent / "AgVFM" / "_data" / "T4_REAL_FLOWER_SOBJ.v18-train_2023_test_location_genotype"
    if agvfm_data.exists():
        images_dir = agvfm_data / "test" / "images"
        labels_dir = agvfm_data / "test" / "labels"
        return images_dir, labels_dir
    
    raise FileNotFoundError("Could not find data directory")


sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300


def load_results(results_file: Path) -> Dict:
    """Load results from JSON file."""
    with open(results_file, 'r') as f:
        return json.load(f)


def extract_method_name(filename: str) -> str:
    """Extract method name from filename."""
    # Handle baseline confidence threshold
    if "baseline_confidence" in filename:
        return "Baseline (conf=0.35)"
    
    # Remove prefix and suffix
    name = filename.replace("semantic_filtering_results_", "").replace(".json", "")
    if name == "semantic_filtering_results" or name == "":
        return "baseline (q=0.1)"
    
    # Format nicely
    if name.startswith("q"):
        return f"BH-FDR q={name[1:]}"
    elif name.startswith("p"):
        return f"Fixed p={name[1:]}"
    elif name.startswith("top"):
        if "_p" in name:
            k, p = name.replace("top", "").split("_p")
            return f"Top-{k} (p≤{p})"
        else:
            return f"Top-{name.replace('top', '')}"
    elif name.startswith("global_q"):
        return f"Global BH q={name.replace('global_q', '')}"
    else:
        return name


def compute_metrics_for_results(results: Dict, model: YOLOWorldModel, images_dir: Path, labels_dir: Path) -> Dict:
    """Compute F1, Precision, Recall for a results file."""
    # Handle baseline confidence threshold (already has metrics)
    if 'metrics' in results and results['config'].get('method') == 'confidence_threshold':
        return results['metrics']
    
    # Handle semantic filtering results
    if 'results' not in results:
        raise ValueError(f"Results file missing 'results' key. Config: {results.get('config', {})}")
    
    results_list = results['results']
    config = results['config']
    conf_floor = config.get('conf_floor', 0.01)
    
    # Collect ground truth and predictions
    list_gt_xyxy = []
    list_pred_kept_xyxy = []
    list_pred_kept_conf = []
    
    for result in results_list:
        image_path = Path(result['image_path'])
        
        # Load ground truth
        gt_boxes = load_ground_truth(image_path, labels_dir)
        list_gt_xyxy.append(gt_boxes)
        
        # Get kept detections (after filtering)
        kept_boxes = np.array(result['kept_boxes']) if result['kept_boxes'] else np.zeros((0, 4))
        kept_confs = np.array(result['kept_confidences']) if result['kept_confidences'] else np.zeros(0)
        list_pred_kept_xyxy.append(kept_boxes)
        list_pred_kept_conf.append(kept_confs)
    
    # Compute metrics
    metrics = compute_metrics_at_iou(
        list_gt_xyxy,
        list_pred_kept_xyxy,
        list_pred_kept_conf,
        iou_threshold=0.5,
    )
    
    return metrics


def plot_metrics_comparison(all_results: List[Tuple[str, Dict]], output_dir: Path, compute_metrics: bool = False):
    """Plot F1, Precision, Recall comparison across methods."""
    print("📊 Computing metrics comparison...")
    
    method_names = []
    f1_scores = []
    precisions = []
    recalls = []
    retention_rates = []
    empty_percentages = []
    
    model = None
    images_dir = None
    labels_dir = None
    
    if compute_metrics:
        print("   Loading YOLO World model...")
        model = YOLOWorldModel()
        images_dir, labels_dir = get_data_paths()
    
    for method_name, results in all_results:
        method_names.append(method_name)
        summary = results['summary']
        
        # Handle baseline confidence threshold (different structure)
        if results['config'].get('method') == 'confidence_threshold':
            total_candidates = summary['total_candidates']
            total_kept = summary['total_kept']
            retention_rate = 1.0  # All candidates are kept (no filtering)
            retention_rates.append(retention_rate)
            empty_percentage = summary.get('empty_percentage', 0.0)
            empty_percentages.append(empty_percentage)
        else:
            # Handle semantic filtering results
            results_list = results['results']
            
            # Compute retention rate
            total_candidates = summary['total_candidates']
            total_kept = summary['total_kept']
            retention_rate = total_kept / total_candidates if total_candidates > 0 else 0.0
            retention_rates.append(retention_rate)
            
            # Compute empty percentage
            empty_count = sum(1 for r in results_list if r['num_kept'] == 0)
            empty_percentage = 100 * empty_count / len(results_list) if len(results_list) > 0 else 0.0
            empty_percentages.append(empty_percentage)
        
        # Compute metrics if requested
        if compute_metrics:
            try:
                metrics = compute_metrics_for_results(results, model, images_dir, labels_dir)
                f1_scores.append(metrics['f1'])
                precisions.append(metrics['precision'])
                recalls.append(metrics['recall'])
            except Exception as e:
                print(f"   ⚠️  Error computing metrics for {method_name}: {e}")
                f1_scores.append(0.0)
                precisions.append(0.0)
                recalls.append(0.0)
        else:
            f1_scores.append(0.0)
            precisions.append(0.0)
            recalls.append(0.0)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Metrics comparison (if computed)
    if compute_metrics and any(f > 0 for f in f1_scores):
        ax = axes[0, 0]
        x = np.arange(len(method_names))
        width = 0.25
        
        ax.bar(x - width, f1_scores, width, label='F1', color='blue', alpha=0.8)
        ax.bar(x, precisions, width, label='Precision', color='green', alpha=0.8)
        ax.bar(x + width, recalls, width, label='Recall', color='orange', alpha=0.8)
        
        ax.set_xlabel('Method', fontsize=10, fontweight='bold')
        ax.set_ylabel('Score', fontsize=10, fontweight='bold')
        ax.set_title('Detection Metrics Comparison (IoU=0.5)', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(method_names, rotation=45, ha='right', fontsize=8)
        ax.set_ylim([0, 1.1])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (f1, p, r) in enumerate(zip(f1_scores, precisions, recalls)):
            if f1 > 0:
                ax.text(i - width, f1 + 0.02, f'{f1:.3f}', ha='center', va='bottom', fontsize=7)
                ax.text(i, p + 0.02, f'{p:.3f}', ha='center', va='bottom', fontsize=7)
                ax.text(i + width, r + 0.02, f'{r:.3f}', ha='center', va='bottom', fontsize=7)
    else:
        axes[0, 0].text(0.5, 0.5, 'Metrics not computed\n(use --compute-metrics)', 
                       ha='center', va='center', transform=axes[0, 0].transAxes, fontsize=12)
        axes[0, 0].set_title('Detection Metrics Comparison', fontsize=12, fontweight='bold')
    
    # Plot 2: Retention rates
    ax = axes[0, 1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(method_names)))
    bars = ax.bar(range(len(method_names)), retention_rates, color=colors, alpha=0.8, edgecolor='black')
    ax.set_xlabel('Method', fontsize=10, fontweight='bold')
    ax.set_ylabel('Retention Rate', fontsize=10, fontweight='bold')
    ax.set_title('Retention Rate Comparison\n(Kept / Candidates)', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(method_names)))
    ax.set_xticklabels(method_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylim([0, max(retention_rates) * 1.2 if retention_rates else 1.0])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (bar, rate) in enumerate(zip(bars, retention_rates)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{rate:.3f}',
               ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Plot 3: Empty image percentages
    ax = axes[1, 0]
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(method_names)))
    bars = ax.bar(range(len(method_names)), empty_percentages, color=colors, alpha=0.8, edgecolor='black')
    ax.set_xlabel('Method', fontsize=10, fontweight='bold')
    ax.set_ylabel('% Images with 0 Detections', fontsize=10, fontweight='bold')
    ax.set_title('Empty Image Percentage', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(method_names)))
    ax.set_xticklabels(method_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylim([0, 100])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (bar, pct) in enumerate(zip(bars, empty_percentages)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{pct:.1f}%',
               ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Plot 4: Summary table
    ax = axes[1, 1]
    ax.axis('tight')
    ax.axis('off')
    
    # Create table data
    table_data = []
    for i, (name, results) in enumerate(all_results):
        summary = results['summary']
        row = [
            method_names[i],
            f"{summary['total_kept']}/{summary['total_candidates']}",
            f"{retention_rates[i]:.3f}",
            f"{empty_percentages[i]:.1f}%",
        ]
        if compute_metrics and f1_scores[i] > 0:
            row.extend([f"{f1_scores[i]:.3f}", f"{precisions[i]:.3f}", f"{recalls[i]:.3f}"])
        else:
            row.extend(["N/A", "N/A", "N/A"])
        table_data.append(row)
    
    headers = ['Method', 'Kept/Candidates', 'Retention', 'Empty %']
    if compute_metrics:
        headers.extend(['F1', 'Precision', 'Recall'])
    
    # Ensure all rows have the same number of columns as headers
    num_cols = len(headers)
    for row in table_data:
        while len(row) < num_cols:
            row.append("N/A")
        row[:num_cols] = row[:num_cols]  # Truncate if too long
    
    # Create table
    try:
        table = ax.table(
            cellText=table_data,
            colLabels=headers,
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1, 1.5)
        
        # Style header
        for i in range(len(headers)):
            cell = table[(0, i)]
            cell.set_facecolor('#40466e')
            cell.set_text_props(weight='bold', color='white')
    except Exception as e:
        # Fallback: just show text
        ax.text(0.5, 0.5, f'Table creation error: {e}', 
               ha='center', va='center', transform=ax.transAxes, fontsize=10)
    
    ax.set_title('Summary Comparison', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'yolo_world_method_comparison.png', bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: yolo_world_method_comparison.png")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare Phase 5 results across methods")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("experiments/results/phase5_semantic_filtering/results"),
        help="Directory containing result JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for plots (default: plots/ in results-dir parent)",
    )
    parser.add_argument(
        "--compute-metrics",
        action="store_true",
        help="Compute F1, Precision, Recall (requires model loading and ground truth)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="semantic_filtering_results*.json",
        help="Pattern to match result files (default: semantic_filtering_results*.json)",
    )
    
    args = parser.parse_args()
    
    if not args.results_dir.exists():
        print(f"❌ Results directory not found: {args.results_dir}")
        sys.exit(1)
    
    if args.output_dir is None:
        args.output_dir = args.results_dir.parent / "plots"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all result files
    result_files = sorted(args.results_dir.glob(args.pattern))
    
    # Also include baseline if pattern matches full files
    if "*_full.json" in args.pattern or "_full.json" in args.pattern:
        baseline_file = args.results_dir / "baseline_confidence_optimal.json"
        if baseline_file.exists() and baseline_file not in result_files:
            result_files.append(baseline_file)
            result_files = sorted(result_files)  # Re-sort
    
    if not result_files:
        print(f"❌ No result files found matching pattern: {args.pattern}")
        sys.exit(1)
    
    print(f"📊 Found {len(result_files)} result files")
    
    # Load all results
    all_results = []
    for result_file in result_files:
        method_name = extract_method_name(result_file.stem)
        print(f"   Loading: {result_file.name} → {method_name}")
        results = load_results(result_file)
        all_results.append((method_name, results))
    
    print(f"\n📈 Generating comparison visualizations...")
    plot_metrics_comparison(all_results, args.output_dir, compute_metrics=args.compute_metrics)
    
    print(f"\n✅ All visualizations saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
