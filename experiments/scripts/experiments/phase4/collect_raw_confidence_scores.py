#!/usr/bin/env python3
"""
Collect raw confidence scores for distribution analysis.

This script runs inference at a very low confidence threshold to collect
all detections, then extracts their confidence scores for distribution analysis.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(project_root))

from agvfm.config.experiments import PHASE3_CONFIGS

# Focus on best configs only (by index in PHASE3_CONFIGS)
# PHASE3_CONFIGS structure: [YOLO World configs (0-2), SAM3 configs (3-5)]
BEST_CONFIG_INDICES = {
    "yolo_world": 0,  # C2_not_bud_calyx - Best YOLO World: mAP 0.4123
    "sam3": 3,  # C2_not_bud - Best SAM3: mAP 0.5407
}
from agvfm.data import get_image_paths, load_ground_truth
from agvfm.models.sam3 import SAM3Model
from agvfm.models.yolo_world import YOLOWorldModel
from agvfm.experiments.evaluator import Evaluator


def collect_raw_confidence_scores(
    evaluator: Evaluator,
    prompt: str,
    absorber_classes: List[str] = None,
    target_indices: List[int] = None,
    min_conf_threshold: float = 0.01,
) -> np.ndarray:
    """
    Collect all raw confidence scores by running inference at very low threshold.
    
    Args:
        evaluator: Evaluator instance
        prompt: Text prompt
        absorber_classes: Optional absorber classes
        target_indices: Optional target indices
        min_conf_threshold: Minimum confidence threshold to collect all detections
        
    Returns:
        Array of all confidence scores
    """
    # Set very low threshold to collect all detections
    evaluator.conf_threshold = min_conf_threshold
    
    all_confidences = []
    
    print(f"Collecting raw confidence scores for prompt: {prompt}")
    if absorber_classes:
        print(f"  Absorber classes: {absorber_classes}")
        print(f"  Target indices: {target_indices}")
    
    # Process images
    for img_idx, img_path in enumerate(evaluator.image_paths):
        if (img_idx + 1) % 50 == 0:
            print(f"  Processing image {img_idx + 1}/{len(evaluator.image_paths)}...")
        
        # Get predictions
        if absorber_classes:
            # Multi-class prediction for absorber architecture
            if hasattr(evaluator.model, 'predict_multi_class'):
                boxes, confidences = evaluator.model.predict_multi_class(
                    image_path=img_path,
                    class_names=[prompt] + absorber_classes,
                    target_indices=target_indices,
                    conf_threshold=min_conf_threshold,
                )
            else:
                # Fallback: single class
                boxes, confidences = evaluator.model.predict(
                    image_path=img_path,
                    prompt=prompt,
                    conf_threshold=min_conf_threshold,
                )
        else:
            # Single class prediction
            boxes, confidences = evaluator.model.predict(
                image_path=img_path,
                prompt=prompt,
                conf_threshold=min_conf_threshold,
            )
        
        # Collect confidences
        if len(confidences) > 0:
            all_confidences.extend(confidences.tolist())
    
    return np.array(all_confidences)


def main():
    """Main function to collect raw confidence scores for Phase 4 analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Collect raw confidence scores for distribution analysis"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["yolo_world", "sam3", "all"],
        default="all",
        help="Model to analyze (default: all - processes best configs only)",
    )
    parser.add_argument(
        "--all-configs",
        action="store_true",
        help="Process all Phase 3 configs instead of just best ones",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Root directory containing _data/ subdirectory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results/phase4_unlabeled_threshold_selection"),
        help="Output directory for raw confidence scores",
    )
    parser.add_argument(
        "--min-conf",
        type=float,
        default=0.01,
        help="Minimum confidence threshold to collect all detections",
    )
    
    args = parser.parse_args()
    
    # Find data directory (using same logic as Phase 3)
    import os
    project_root = Path(__file__).resolve().parents[5]
    
    # Check DATA_ROOT environment variable first
    data_root = os.environ.get("DATA_ROOT")
    if data_root:
        agvfm_data = Path(data_root) / "_data" / "T4_REAL_FLOWER_SOBJ.v18-train_2023_test_location_genotype"
        if agvfm_data.exists():
            images_dir = agvfm_data / "test" / "images"
            labels_dir = agvfm_data / "test" / "labels"
        else:
            images_dir = None
            labels_dir = None
    else:
        images_dir = None
        labels_dir = None
    
    # Try AgVFM2 first
    if images_dir is None or not images_dir.exists():
        agvfm2_data = project_root / "_data" / "T4_REAL_FLOWER_SOBJ.v18-train_2023_test_location_genotype"
        if agvfm2_data.exists():
            images_dir = agvfm2_data / "test" / "images"
            labels_dir = agvfm2_data / "test" / "labels"
    
    # Try direct path
    if images_dir is None or not images_dir.exists():
        direct_path = Path("/data2/jmearles/AgVFM2/_data/T4_REAL_FLOWER_SOBJ.v18-train_2023_test_location_genotype")
        if direct_path.exists():
            images_dir = direct_path / "test" / "images"
            labels_dir = direct_path / "test" / "labels"
    
    # Try AgVFM (sibling directory) as fallback
    if images_dir is None or not images_dir.exists():
        agvfm_data = project_root.parent / "AgVFM" / "_data" / "T4_REAL_FLOWER_SOBJ.v18-train_2023_test_location_genotype"
        if agvfm_data.exists():
            images_dir = agvfm_data / "test" / "images"
            labels_dir = agvfm_data / "test" / "labels"
    
    # Override with --data-root if provided
    if args.data_root:
        agvfm_data = Path(args.data_root) / "_data" / "T4_REAL_FLOWER_SOBJ.v18-train_2023_test_location_genotype"
        if agvfm_data.exists():
            images_dir = agvfm_data / "test" / "images"
            labels_dir = agvfm_data / "test" / "labels"
    
    if not images_dir.exists():
        print(f"❌ Error: Images directory not found: {images_dir}")
        sys.exit(1)
    
    if not labels_dir.exists():
        print(f"❌ Error: Labels directory not found: {labels_dir}")
        sys.exit(1)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get configs to process
    models_to_process = []
    if args.model in ["yolo_world", "all"]:
        models_to_process.append("yolo_world")
    if args.model in ["sam3", "all"]:
        models_to_process.append("sam3")
    
    # Process each model
    for model_name in models_to_process:
        print(f"\n{'='*60}")
        print(f"Processing {model_name.upper()}")
        print(f"{'='*60}")
        
        # Load model
        if model_name == "yolo_world":
            model = YOLOWorldModel()
        else:
            model = SAM3Model()
        
        # Create evaluator
        evaluator = Evaluator(
            model=model,
            images_dir=images_dir,
            labels_dir=labels_dir,
            conf_threshold=args.min_conf,
        )
        
        # Get configs for this model
        if args.all_configs:
            # Process all Phase 3 configs for this model
            if model_name == "yolo_world":
                configs = PHASE3_CONFIGS[:3]  # First 3 are YOLO World
            else:
                configs = PHASE3_CONFIGS[3:]  # Last 3 are SAM3
            print(f"📋 Processing all {len(configs)} Phase 3 configs for {model_name}")
        else:
            # Focus on best config only
            if model_name not in BEST_CONFIG_INDICES:
                print(f"⚠️  No best config defined for {model_name}")
                continue
            
            config_idx = BEST_CONFIG_INDICES[model_name]
            configs = [PHASE3_CONFIGS[config_idx]]
            print(f"📌 Focusing on best config: {configs[0].name} ({configs[0].description})")
        
        if not configs:
            print(f"⚠️  No configs found for {model_name}")
            continue
        
        # Process each config
        for config in configs:
            config_name = config.name
            prompt = config.prompt
            absorber_classes = getattr(config, "absorber_classes", None)
            target_indices = getattr(config, "target_indices", None)
            
            print(f"\n📊 Collecting confidence scores for: {config_name}")
            
            # Collect raw confidence scores
            try:
                confidences = collect_raw_confidence_scores(
                    evaluator=evaluator,
                    prompt=prompt,
                    absorber_classes=absorber_classes,
                    target_indices=target_indices,
                    min_conf_threshold=args.min_conf,
                )
                
                # Save results
                output_file = args.output_dir / f"{model_name}_{config_name}_raw_confidence.json"
                
                result = {
                    "config_name": config_name,
                    "model": model_name,
                    "prompt": prompt,
                    "absorber_classes": absorber_classes,
                    "target_indices": target_indices,
                    "min_conf_threshold": args.min_conf,
                    "n_detections": len(confidences),
                    "confidences": confidences.tolist(),
                    "statistics": {
                        "mean": float(np.mean(confidences)),
                        "std": float(np.std(confidences)),
                        "median": float(np.median(confidences)),
                        "min": float(np.min(confidences)),
                        "max": float(np.max(confidences)),
                        "q25": float(np.percentile(confidences, 25)),
                        "q75": float(np.percentile(confidences, 75)),
                    },
                }
                
                with open(output_file, "w") as f:
                    json.dump(result, f, indent=2)
                
                print(f"✅ Saved {len(confidences)} confidence scores to {output_file}")
                print(f"   Mean: {np.mean(confidences):.4f}, Std: {np.std(confidences):.4f}")
                
            except Exception as e:
                print(f"❌ Error processing {config_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"\n✅ Done! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
