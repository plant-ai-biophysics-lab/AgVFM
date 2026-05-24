#!/usr/bin/env python3
"""
Visualize Phase 5 semantic filtering results.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(project_root))

from agvfm.models.yolo_world import YOLOWorldModel
from agvfm.data.labels import load_ground_truth
from agvfm.evaluation.metrics import compute_metrics_at_iou


def get_data_paths():
    """Get data paths using same logic as Phase 4."""
    import os
    project_root = Path(__file__).resolve().parents[5]
    
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


def plot_candidate_vs_kept(results: Dict, output_dir: Path):
    """Plot candidates vs kept detections per image."""
    results_list = results['results']
    
    num_candidates = [r['num_candidates'] for r in results_list]
    num_kept = [r['num_kept'] for r in results_list]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Scatter plot
    axes[0].scatter(num_candidates, num_kept, alpha=0.6, s=50)
    axes[0].plot([0, max(num_candidates)], [0, max(num_candidates)], 'r--', alpha=0.5, label='y=x')
    axes[0].set_xlabel('Number of Candidates')
    axes[0].set_ylabel('Number Kept (after BH-FDR)')
    axes[0].set_title('Candidates vs Kept Detections')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Histogram of kept detections
    axes[1].hist(num_kept, bins=20, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Number Kept per Image')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Kept Detections')
    axes[1].axvline(np.mean(num_kept), color='r', linestyle='--', label=f'Mean: {np.mean(num_kept):.2f}')
    axes[1].axvline(np.median(num_kept), color='g', linestyle='--', label=f'Median: {np.median(num_kept):.1f}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'yolo_world_candidates_vs_kept.png', bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: yolo_world_candidates_vs_kept.png")


def plot_margin_distributions(results: Dict, output_dir: Path):
    """Plot distributions of margins for candidates and background."""
    results_list = results['results']
    
    all_candidate_margins = []
    all_bg_margins = []
    
    for r in results_list:
        if r.get('margins'):
            all_candidate_margins.extend(r['margins'])
        # Note: bg_margins not saved in current version, would need to add
    
    if not all_candidate_margins:
        print("⚠️  No margin data found")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram of candidate margins
    axes[0].hist(all_candidate_margins, bins=50, edgecolor='black', alpha=0.7, color='blue')
    axes[0].set_xlabel('Semantic Margin')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Candidate Margins')
    axes[0].axvline(np.mean(all_candidate_margins), color='r', linestyle='--', 
                     label=f'Mean: {np.mean(all_candidate_margins):.3f}')
    axes[0].axvline(np.median(all_candidate_margins), color='g', linestyle='--', 
                    label=f'Median: {np.median(all_candidate_margins):.3f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot by kept status
    kept_margins = []
    rejected_margins = []
    
    for r in results_list:
        margins = r.get('margins', [])
        num_kept = r['num_kept']
        if margins:
            if num_kept > 0:
                # Get margins for kept detections (first num_kept)
                kept_margins.extend(margins[:num_kept])
                rejected_margins.extend(margins[num_kept:])
            else:
                rejected_margins.extend(margins)
    
    if kept_margins or rejected_margins:
        data_to_plot = []
        labels = []
        if kept_margins:
            data_to_plot.append(kept_margins)
            labels.append('Kept')
        if rejected_margins:
            data_to_plot.append(rejected_margins)
            labels.append('Rejected')
        
        axes[1].boxplot(data_to_plot, tick_labels=labels)
        axes[1].set_ylabel('Semantic Margin')
        axes[1].set_title('Margins: Kept vs Rejected')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'yolo_world_margin_distributions.png', bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: yolo_world_margin_distributions.png")


def plot_p_value_distributions(results: Dict, output_dir: Path):
    """Plot distributions of p-values."""
    results_list = results['results']
    
    all_p_values = []
    kept_p_values = []
    rejected_p_values = []
    
    for r in results_list:
        p_values = r.get('p_values', [])
        num_kept = r['num_kept']
        if p_values:
            all_p_values.extend(p_values)
            if num_kept > 0:
                kept_p_values.extend(p_values[:num_kept])
                rejected_p_values.extend(p_values[num_kept:])
            else:
                rejected_p_values.extend(p_values)
    
    if not all_p_values:
        print("⚠️  No p-value data found")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram of all p-values
    axes[0].hist(all_p_values, bins=50, edgecolor='black', alpha=0.7, color='purple')
    axes[0].set_xlabel('P-Value')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of All P-Values')
    axes[0].axvline(results['config']['q_fdr'], color='r', linestyle='--', 
                    label=f"FDR threshold: {results['config']['q_fdr']}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot by kept status
    if kept_p_values or rejected_p_values:
        data_to_plot = []
        labels = []
        if kept_p_values:
            data_to_plot.append(kept_p_values)
            labels.append('Kept')
        if rejected_p_values:
            data_to_plot.append(rejected_p_values)
            labels.append('Rejected')
        
        axes[1].boxplot(data_to_plot, tick_labels=labels)
        axes[1].set_ylabel('P-Value')
        axes[1].set_title('P-Values: Kept vs Rejected')
        axes[1].axhline(results['config']['q_fdr'], color='r', linestyle='--', 
                        label=f"FDR threshold: {results['config']['q_fdr']}")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'yolo_world_pvalue_distributions.png', bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: yolo_world_pvalue_distributions.png")


def plot_summary_statistics(results: Dict, output_dir: Path):
    """Create summary statistics plot."""
    results_list = results['results']
    
    num_candidates = [r['num_candidates'] for r in results_list]
    num_kept = [r['num_kept'] for r in results_list]
    keep_rate = [kept / max(cand, 1) for cand, kept in zip(num_candidates, num_kept)]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Candidates distribution
    axes[0, 0].hist(num_candidates, bins=20, edgecolor='black', alpha=0.7, color='blue')
    axes[0, 0].set_xlabel('Number of Candidates')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title(f'Candidates Distribution\n(Mean: {np.mean(num_candidates):.1f}, Median: {np.median(num_candidates):.1f})')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Kept distribution
    axes[0, 1].hist(num_kept, bins=20, edgecolor='black', alpha=0.7, color='green')
    axes[0, 1].set_xlabel('Number Kept')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title(f'Kept Distribution\n(Mean: {np.mean(num_kept):.2f}, Median: {np.median(num_kept):.1f})')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Keep rate
    axes[1, 0].hist(keep_rate, bins=20, edgecolor='black', alpha=0.7, color='orange')
    axes[1, 0].set_xlabel('Keep Rate (Kept / Candidates)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title(f'Keep Rate Distribution\n(Mean: {np.mean(keep_rate):.3f})')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Images with 0 detections
    empty_count = sum(1 for r in results_list if r['num_kept'] == 0)
    non_empty_count = len(results_list) - empty_count
    
    axes[1, 1].pie([empty_count, non_empty_count], 
                   labels=[f'Empty ({empty_count})', f'Non-Empty ({non_empty_count})'],
                   autopct='%1.1f%%', startangle=90)
    axes[1, 1].set_title('Images with Detections')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'yolo_world_summary_statistics.png', bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: yolo_world_summary_statistics.png")


def plot_metrics_before_after(results: Dict, output_dir: Path):
    """Plot F1, Precision, and Recall before and after filtering."""
    print("📊 Computing metrics before and after filtering...")
    
    # Load model
    print("   Loading YOLO World model...")
    model = YOLOWorldModel()
    
    # Get data paths
    images_dir, labels_dir = get_data_paths()
    
    results_list = results['results']
    config = results['config']
    conf_floor = config['conf_floor']
    
    # Collect ground truth and predictions
    list_gt_xyxy = []
    list_pred_candidates_xyxy = []  # Before filtering
    list_pred_candidates_conf = []
    list_pred_kept_xyxy = []  # After filtering
    list_pred_kept_conf = []
    
    print(f"   Processing {len(results_list)} images...")
    for idx, result in enumerate(results_list):
        if (idx + 1) % 20 == 0:
            print(f"      [{idx + 1}/{len(results_list)}]...")
        
        image_path = Path(result['image_path'])
        
        # Load ground truth
        gt_boxes = load_ground_truth(image_path, labels_dir)
        list_gt_xyxy.append(gt_boxes)
        
        # Regenerate candidate detections (before filtering)
        candidate_boxes, candidate_confs = model.predict(
            image_path=image_path,
            prompt="cowpea flower",
            conf_threshold=conf_floor,
        )
        list_pred_candidates_xyxy.append(candidate_boxes)
        list_pred_candidates_conf.append(candidate_confs)
        
        # Get kept detections (after filtering)
        kept_boxes = np.array(result['kept_boxes']) if result['kept_boxes'] else np.zeros((0, 4))
        kept_confs = np.array(result['kept_confidences']) if result['kept_confidences'] else np.zeros(0)
        list_pred_kept_xyxy.append(kept_boxes)
        list_pred_kept_conf.append(kept_confs)
    
    print("   Computing metrics at IoU=0.5...")
    
    # Evaluate candidates (before filtering)
    metrics_before = compute_metrics_at_iou(
        list_gt_xyxy,
        list_pred_candidates_xyxy,
        list_pred_candidates_conf,
        iou_threshold=0.5,
    )
    
    # Evaluate kept (after filtering)
    metrics_after = compute_metrics_at_iou(
        list_gt_xyxy,
        list_pred_kept_xyxy,
        list_pred_kept_conf,
        iou_threshold=0.5,
    )
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    metrics_names = ['F1', 'Precision', 'Recall']
    before_values = [
        metrics_before['f1'],
        metrics_before['precision'],
        metrics_before['recall'],
    ]
    after_values = [
        metrics_after['f1'],
        metrics_after['precision'],
        metrics_after['recall'],
    ]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, before_values, width, label='Before Filtering (Candidates)', 
                   color='lightblue', edgecolor='black', alpha=0.8)
    bars2 = ax.bar(x + width/2, after_values, width, label='After Filtering (Kept)', 
                   color='lightgreen', edgecolor='black', alpha=0.8)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Detection Metrics: Before vs After Semantic Filtering', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.set_ylim([0, 1.1])
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add summary text
    summary_text = (
        f"Before: {metrics_before['total_tp']} TP, {metrics_before['total_fp']} FP, {metrics_before['total_fn']} FN\n"
        f"After:  {metrics_after['total_tp']} TP, {metrics_after['total_fp']} FP, {metrics_after['total_fn']} FN\n"
        f"Total GT: {metrics_before['n_gt_total']} objects"
    )
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', 
            facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'yolo_world_metrics_before_after.png', bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: yolo_world_metrics_before_after.png")
    print(f"   Before: F1={metrics_before['f1']:.3f}, P={metrics_before['precision']:.3f}, R={metrics_before['recall']:.3f}")
    print(f"   After:  F1={metrics_after['f1']:.3f}, P={metrics_after['precision']:.3f}, R={metrics_after['recall']:.3f}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize Phase 5 results")
    parser.add_argument(
        "--results-file",
        type=Path,
        default=Path("experiments/results/phase5_semantic_filtering/semantic_filtering_results.json"),
        help="Path to results JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for plots (default: same as results file directory)",
    )
    
    args = parser.parse_args()
    
    if not args.results_file.exists():
        print(f"❌ Results file not found: {args.results_file}")
        sys.exit(1)
    
    if args.output_dir is None:
        args.output_dir = args.results_file.parent / "plots"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📊 Loading results from: {args.results_file}")
    results = load_results(args.results_file)
    
    print(f"📈 Generating visualizations...")
    plot_candidate_vs_kept(results, args.output_dir)
    plot_margin_distributions(results, args.output_dir)
    plot_p_value_distributions(results, args.output_dir)
    plot_summary_statistics(results, args.output_dir)
    plot_metrics_before_after(results, args.output_dir)
    
    print(f"\n✅ All visualizations saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
