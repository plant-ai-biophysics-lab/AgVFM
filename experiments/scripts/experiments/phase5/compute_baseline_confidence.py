#!/usr/bin/env python3
"""
Compute baseline metrics using optimal confidence threshold from Phase 3.

This creates a baseline comparison for Phase 5 semantic filtering methods.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import argparse

import numpy as np

# Add project root to path
script_path = Path(__file__).resolve()
parts = script_path.parts
agvfm_idx = [i for i, part in enumerate(parts) if part == 'AgVFM2']
if agvfm_idx:
    project_root = Path(*parts[:agvfm_idx[0] + 1])
else:
    project_root = script_path.parents[5]
sys.path.insert(0, str(project_root))

from agvfm.models.yolo_world import YOLOWorldModel
from agvfm.data.labels import get_image_paths, load_ground_truth
from agvfm.evaluation.metrics import compute_metrics_at_iou


def get_data_paths():
    """Get data paths using same logic as Phase 5."""
    import os
    script_path = Path(__file__).resolve()
    parts = script_path.parts
    agvfm_idx = [i for i, part in enumerate(parts) if part == 'AgVFM2']
    if agvfm_idx:
        project_root = Path(*parts[:agvfm_idx[0] + 1])
    else:
        project_root = script_path.parents[5]
    
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


def find_optimal_threshold_from_phase3(phase3_results_file: Path) -> float:
    """Find optimal confidence threshold from Phase 3 results (highest F1)."""
    with open(phase3_results_file, 'r') as f:
        data = json.load(f)
    
    best_f1 = 0
    best_threshold = 0.3  # Default
    
    for config_name, config_data in data['results'].items():
        results_by_conf = config_data['results_by_conf']
        for conf_str, result in results_by_conf.items():
            f1 = result['metrics']['f1']
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(conf_str)
    
    return best_threshold


def compute_baseline_metrics(
    conf_threshold: float,
    prompt: str = "a single yellow cowpea flower with open petals, not a bud, not the green calyx",
) -> Dict:
    """Compute baseline metrics using confidence threshold only."""
    print(f"🔧 Loading YOLO World model...")
    model = YOLOWorldModel()
    
    images_dir, labels_dir = get_data_paths()
    image_paths = get_image_paths(images_dir)
    
    print(f"📊 Processing {len(image_paths)} images with conf_threshold={conf_threshold}...")
    
    list_gt_xyxy = []
    list_pred_xyxy = []
    list_pred_conf = []
    
    for idx, img_path in enumerate(image_paths, 1):
        if idx % 20 == 0:
            print(f"   [{idx}/{len(image_paths)}]...")
        
        # Load ground truth
        gt_boxes = load_ground_truth(img_path, labels_dir)
        list_gt_xyxy.append(gt_boxes)
        
        # Run prediction
        pred_boxes, pred_confs = model.predict(
            img_path,
            prompt,
            conf_threshold=conf_threshold,
        )
        
        list_pred_xyxy.append(pred_boxes)
        list_pred_conf.append(pred_confs)
    
    # Compute metrics
    print(f"📈 Computing metrics at IoU=0.5...")
    metrics = compute_metrics_at_iou(
        list_gt_xyxy,
        list_pred_xyxy,
        list_pred_conf,
        iou_threshold=0.5,
    )
    
    # Compute summary statistics
    total_candidates = sum(len(boxes) for boxes in list_pred_xyxy)
    total_kept = total_candidates  # All candidates are kept (no filtering)
    avg_kept_per_image = total_kept / len(image_paths) if len(image_paths) > 0 else 0.0
    empty_count = sum(1 for boxes in list_pred_xyxy if len(boxes) == 0)
    empty_percentage = 100 * empty_count / len(image_paths) if len(image_paths) > 0 else 0.0
    
    # Convert numpy types to Python types for JSON serialization
    metrics_serializable = {
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "map": float(metrics["map"]),
        "f1": float(metrics["f1"]),
        "total_tp": int(metrics["total_tp"]),
        "total_fp": int(metrics["total_fp"]),
        "total_fn": int(metrics["total_fn"]),
        "n_images": int(metrics["n_images"]),
        "n_gt_total": int(metrics["n_gt_total"]),
        "n_pred_total": int(metrics.get("n_pred_total", total_candidates)),
    }
    
    return {
        "config": {
            "model": "yolo_world",
            "method": "confidence_threshold",
            "conf_threshold": conf_threshold,
            "prompt": prompt,
        },
        "summary": {
            "num_images": len(image_paths),
            "total_candidates": total_candidates,
            "total_kept": total_kept,
            "avg_kept_per_image": float(avg_kept_per_image),
            "empty_percentage": float(empty_percentage),
        },
        "metrics": metrics_serializable,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute baseline metrics using confidence threshold")
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=None,
        help="Confidence threshold (if not provided, will find optimal from Phase 3)",
    )
    parser.add_argument(
        "--phase3-results",
        type=Path,
        default=Path("experiments/results/phase3_confidence_sweeps/yolo_world_confidence_sweeps.json"),
        help="Path to Phase 3 results file",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("experiments/results/phase5_semantic_filtering/results/baseline_confidence_optimal.json"),
        help="Output JSON file",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a single yellow cowpea flower with open petals, not a bud, not the green calyx",
        help="Text prompt for YOLO World",
    )
    
    args = parser.parse_args()
    
    # Find optimal threshold if not provided
    if args.conf_threshold is None:
        if not args.phase3_results.exists():
            print(f"❌ Phase 3 results file not found: {args.phase3_results}")
            print("   Please provide --conf-threshold or ensure Phase 3 results exist")
            sys.exit(1)
        
        print(f"📊 Finding optimal threshold from Phase 3 results...")
        optimal_threshold = find_optimal_threshold_from_phase3(args.phase3_results)
        print(f"   Optimal threshold: {optimal_threshold} (highest F1)")
    else:
        optimal_threshold = args.conf_threshold
        print(f"📊 Using provided threshold: {optimal_threshold}")
    
    # Compute baseline metrics
    results = compute_baseline_metrics(optimal_threshold, prompt=args.prompt)
    
    # Save results
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Baseline metrics computed!")
    print(f"   Conf threshold: {optimal_threshold}")
    print(f"   F1: {results['metrics']['f1']:.3f}")
    print(f"   Precision: {results['metrics']['precision']:.3f}")
    print(f"   Recall: {results['metrics']['recall']:.3f}")
    print(f"   Total predictions: {results['summary']['total_kept']}")
    print(f"   Avg per image: {results['summary']['avg_kept_per_image']:.2f}")
    print(f"   Empty images: {results['summary']['empty_percentage']:.1f}%")
    print(f"\n💾 Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
