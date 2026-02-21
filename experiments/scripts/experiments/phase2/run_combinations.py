#!/usr/bin/env python3
"""
Run Phase 2: Combination Tests & Absorber Architecture

Systematic combinations, negation strategies, and absorber architecture tests.

Usage:
    python experiments/scripts/experiments/phase2/run_combinations.py --model yolo_world
    python experiments/scripts/experiments/phase2/run_combinations.py --model sam3
    python experiments/scripts/experiments/phase2/run_combinations.py --model all
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Force unbuffered output for real-time logging
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
sys.stderr.reconfigure(line_buffering=True) if hasattr(sys.stderr, 'reconfigure') else None

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from agvfm.config.experiments import (
    ABSORBER_CONFIGS,
    COMBINATION_CONFIGS,
    MULTICLASS_CONFIGS,
)
from agvfm.data.labels import get_image_paths
from agvfm.experiments import Evaluator
from agvfm.models import SAM3Model, YOLOWorldModel
from agvfm.utils import setup_logging, log_experiment_start, log_experiment_complete


def get_data_paths() -> Dict[str, Path]:
    """Get dataset paths. Uses full test set (158 images) to match Phase 1."""
    # Check DATA_ROOT environment variable first
    data_root = os.environ.get("DATA_ROOT")
    if data_root:
        agvfm_data = Path(data_root) / "_data" / "T4_REAL_FLOWER_SOBJ.v18-train_2023_test_location_genotype"
        if agvfm_data.exists():
            test_images_dir = agvfm_data / "test" / "images"
            test_labels_dir = agvfm_data / "test" / "labels"
            dev_manifest = agvfm_data / "dev" / "manifest.txt"
            
            return {
                "images_dir": test_images_dir,
                "labels_dir": test_labels_dir,
                "dev_manifest": dev_manifest,
            }
    
    # Try AgVFM2 first (project_root is AgVFM2)
    agvfm2_data = project_root / "_data" / "T4_REAL_FLOWER_SOBJ.v18-train_2023_test_location_genotype"
    if agvfm2_data.exists():
        test_images_dir = agvfm2_data / "test" / "images"
        test_labels_dir = agvfm2_data / "test" / "labels"
        dev_manifest = agvfm2_data / "dev" / "manifest.txt"
        
        return {
            "images_dir": test_images_dir,
            "labels_dir": test_labels_dir,
            "dev_manifest": dev_manifest,
        }
    
    # Try direct path to AgVFM2
    direct_path = Path("/data2/jmearles/AgVFM2/_data/T4_REAL_FLOWER_SOBJ.v18-train_2023_test_location_genotype")
    if direct_path.exists():
        test_images_dir = direct_path / "test" / "images"
        test_labels_dir = direct_path / "test" / "labels"
        dev_manifest = direct_path / "dev" / "manifest.txt"
        
        return {
            "images_dir": test_images_dir,
            "labels_dir": test_labels_dir,
            "dev_manifest": dev_manifest,
        }
    
    # Try AgVFM (sibling directory) as fallback
    agvfm_data = project_root.parent / "AgVFM" / "_data" / "T4_REAL_FLOWER_SOBJ.v18-train_2023_test_location_genotype"
    if agvfm_data.exists():
        test_images_dir = agvfm_data / "test" / "images"
        test_labels_dir = agvfm_data / "test" / "labels"
        dev_manifest = agvfm_data / "dev" / "manifest.txt"
        
        return {
            "images_dir": test_images_dir,
            "labels_dir": test_labels_dir,
            "dev_manifest": dev_manifest,
        }
    
    raise FileNotFoundError(
        f"Dataset not found. Tried:\n"
        f"  - DATA_ROOT: {data_root or 'not set'}\n"
        f"  - {agvfm2_data}\n"
        f"  - {direct_path}\n"
        f"  - {agvfm_data}"
    )


def load_results(results_file: Path) -> Dict:
    """Load existing results from JSON file."""
    if results_file.exists():
        try:
            with open(results_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError) as e:
            # File is corrupted - backup it and start fresh
            print(f"⚠️  Warning: Results file is corrupted ({e})")
            backup_file = results_file.with_suffix('.json.backup')
            print(f"  Backing up corrupted file to: {backup_file}")
            import shutil
            shutil.copy2(results_file, backup_file)
            print(f"  Starting with fresh results file")
            return {"results": {}}
    return {"results": {}}


def convert_to_native_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    import numpy as np
    
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_native_types(item) for item in obj)
    else:
        return obj


def save_results(results_file: Path, results: Dict):
    """Save results to JSON file."""
    results_file.parent.mkdir(parents=True, exist_ok=True)
    # Convert numpy types to native Python types for JSON serialization
    results_serializable = convert_to_native_types(results)
    with open(results_file, "w") as f:
        json.dump(results_serializable, f, indent=2)


def run_combinations(
    model_name: str,
    evaluator: Evaluator,
    results_file: Path,
    resume: bool = True,
):
    """Run combination tests."""
    print(f"\n{'='*80}")
    print(f"Running Combination Tests: {model_name}")
    print(f"{'='*80}")
    
    results = load_results(results_file) if resume else {"results": {}}
    
    total_configs = len(COMBINATION_CONFIGS)
    completed = len([k for k in results.get("results", {}) if k.startswith("C")])
    
    print(f"\nTotal configs: {total_configs}")
    print(f"Already completed: {completed}")
    print(f"Remaining: {total_configs - completed}\n")
    
    for idx, config in enumerate(COMBINATION_CONFIGS, 1):
        config_key = config.name
        
        # Skip if already completed
        if resume and config_key in results.get("results", {}):
            print(f"[{idx}/{total_configs}] Skipping {config_key} (already completed)")
            continue
        
        print(f"\n[{idx}/{total_configs}] Testing: {config_key}")
        print(f"  Prompt: {config.prompt}")
        print(f"  Description: {config.description}")
        
        try:
            result = evaluator.evaluate_prompt(
                prompt=config.prompt,
                iou_thresholds=[0.3, 0.5],
            )
            
            # Add config metadata
            result["config_name"] = config.name
            result["description"] = config.description
            
            # Store result
            if "results" not in results:
                results["results"] = {}
            results["results"][config_key] = result
            
            # Save incrementally
            save_results(results_file, results)
            
            # Print summary
            map_05 = result["metrics_by_iou"]["iou_0.5"]["map"]
            f1_05 = result["metrics_by_iou"]["iou_0.5"]["f1"]
            print(f"  ✓ mAP@0.5: {map_05:.4f}, F1: {f1_05:.4f}")
            print(f"  💾 Saved to {results_file}")
            print(f"  Progress: {idx}/{total_configs} configurations complete")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*80}")
    print(f"Combination Tests Complete: {model_name}")
    print(f"{'='*80}")


def run_absorbers(
    model_name: str,
    evaluator: Evaluator,
    results_file: Path,
    resume: bool = True,
):
    """Run absorber architecture tests (YOLO World only)."""
    if model_name.lower() != "yolo_world":
        print(f"\nSkipping absorber tests for {model_name} (YOLO World only)")
        return
    
    print(f"\n{'='*80}")
    print(f"Running Absorber Architecture Tests: {model_name}")
    print(f"{'='*80}")
    
    results = load_results(results_file) if resume else {"results": {}}
    
    total_configs = len(ABSORBER_CONFIGS)
    completed = len([k for k in results.get("results", {}) if k.startswith("H")])
    
    print(f"\nTotal configs: {total_configs}")
    print(f"Already completed: {completed}")
    print(f"Remaining: {total_configs - completed}\n")
    
    for idx, config in enumerate(ABSORBER_CONFIGS, 1):
        config_key = config.name
        
        # Skip if already completed
        if resume and config_key in results.get("results", {}):
            print(f"[{idx}/{total_configs}] Skipping {config_key} (already completed)")
            continue
        
        print(f"\n[{idx}/{total_configs}] Testing: {config_key}")
        print(f"  Target prompt: {config.prompt}")
        print(f"  Absorber classes: {config.absorber_classes}")
        print(f"  Target indices: {config.target_indices}")
        print(f"  Description: {config.description}")
        
        try:
            result = evaluator.evaluate_prompt(
                prompt=config.prompt,
                absorber_classes=config.absorber_classes,
                target_indices=config.target_indices,
                iou_thresholds=[0.3, 0.5],
            )
            
            # Add config metadata
            result["config_name"] = config.name
            result["description"] = config.description
            
            # Store result
            if "results" not in results:
                results["results"] = {}
            results["results"][config_key] = result
            
            # Save incrementally
            save_results(results_file, results)
            
            # Print summary
            map_05 = result["metrics_by_iou"]["iou_0.5"]["map"]
            f1_05 = result["metrics_by_iou"]["iou_0.5"]["f1"]
            print(f"  ✓ mAP@0.5: {map_05:.4f}, F1: {f1_05:.4f}")
            print(f"  💾 Saved to {results_file}")
            print(f"  Progress: {idx}/{total_configs} configurations complete")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*80}")
    print(f"Absorber Tests Complete: {model_name}")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="Run Phase 2: Combination Tests & Absorber Architecture")
    parser.add_argument(
        "--model",
        type=str,
        choices=["yolo_world", "sam3", "all"],
        default="all",
        help="Model to test (default: all)",
    )
    parser.add_argument(
        "--full-test-set",
        action="store_true",
        help="Use full test set (158 images, includes dev images)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume from existing results (start fresh)",
    )
    parser.add_argument(
        "--yolo-weights",
        type=str,
        default="model_weights/yolo_world/yolov8x-worldv2.pt",
        help="Path to YOLO World weights",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Root directory containing _data folder",
    )
    parser.add_argument(
        "--combinations-only",
        action="store_true",
        help="Only run combination tests (skip absorbers and multi-class)",
    )
    
    args = parser.parse_args()
    
    # Get data paths
    if args.data_root:
        os.environ["DATA_ROOT"] = args.data_root
    
    data_paths = get_data_paths()
    
    # Set up results directory
    results_dir = project_root / "experiments" / "results" / "phase2_combinations"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    logs_dir = results_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    models_to_run = []
    if args.model == "all":
        models_to_run = ["yolo_world", "sam3"]
    else:
        models_to_run = [args.model]
    
    for model_name in models_to_run:
        # Set up model-specific logging
        model_prefix = model_name.replace("_", "-")
        log_file = setup_logging(
            logs_dir,
            f"{model_prefix}_run",
        )
        
        # Get test set size
        test_images = get_image_paths(data_paths["images_dir"])
        test_set_size = len(test_images)
        
        logger = logging.getLogger(__name__)
        log_experiment_start(
            experiment_name=f"Phase 2: {model_name}",
            config={
                "model": model_name,
                "test_set_size": test_set_size,
                "full_test_set": args.full_test_set,
            },
        )
        
        # Initialize model
        if model_name == "yolo_world":
            model = YOLOWorldModel(weights_path=args.yolo_weights)
            batch_size = 15
        elif model_name == "sam3":
            model = SAM3Model(model_id="facebook/sam3")
            batch_size = 1  # SAM3 batch processing has issues with attention mechanism, use sequential processing
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Initialize evaluator
        evaluator = Evaluator(
            model=model,
            images_dir=data_paths["images_dir"],
            labels_dir=data_paths["labels_dir"],
            conf_threshold=0.1,
            batch_size=batch_size,
        )
        
        # Run combination tests
        combinations_file = results_dir / f"{model_name}_combinations.json"
        run_combinations(
            model_name=model_name,
            evaluator=evaluator,
            results_file=combinations_file,
            resume=not args.no_resume,
        )
        
        # Run absorber tests (YOLO World only)
        if not args.combinations_only and model_name == "yolo_world":
            absorbers_file = results_dir / f"{model_name}_absorbers.json"
            run_absorbers(
                model_name=model_name,
                evaluator=evaluator,
                results_file=absorbers_file,
                resume=not args.no_resume,
            )
        
        combinations_file = results_dir / f"{model_name}_combinations.json"
        log_experiment_complete(
            experiment_name=f"Phase 2: {model_name}",
            results_file=combinations_file,
        )
    
    print(f"\n{'='*80}")
    print("Phase 2 Experiments Complete!")
    print(f"Results saved to: {results_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
