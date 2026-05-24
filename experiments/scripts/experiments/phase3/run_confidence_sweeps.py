#!/usr/bin/env python3
"""
Run Phase 3: Confidence Threshold Analysis

Systematic confidence threshold sweeps for top-performing configs from Phase 2.

Usage:
    python experiments/scripts/experiments/phase3/run_confidence_sweeps.py --model yolo_world
    python experiments/scripts/experiments/phase3/run_confidence_sweeps.py --model sam3
    python experiments/scripts/experiments/phase3/run_confidence_sweeps.py --model all
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Force unbuffered output for real-time logging
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
sys.stderr.reconfigure(line_buffering=True) if hasattr(sys.stderr, 'reconfigure') else None

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from agvfm.config.experiments import PHASE3_CONFIGS
from agvfm.data.labels import get_image_paths
from agvfm.experiments import Evaluator
from agvfm.models import SAM3Model, YOLOWorldModel
from agvfm.utils import setup_logging, log_experiment_start, log_experiment_complete


def get_data_paths() -> Dict[str, Path]:
    """Get dataset paths. Uses full test set (158 images) to match Phase 1 and Phase 2."""
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


def convert_to_native_types(obj):
    """Convert NumPy types to native Python types for JSON serialization."""
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
    return obj


def load_results(results_file: Path) -> Dict:
    """Load existing results, handling corrupted JSON gracefully."""
    if not results_file.exists():
        return {"results": {}}
    
    try:
        with open(results_file) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"⚠️  Warning: Corrupted JSON file detected: {results_file}")
        print(f"   Error: {e}")
        # Backup corrupted file
        backup_file = results_file.with_suffix(f".backup_{results_file.stat().st_mtime}.json")
        print(f"   Backing up to: {backup_file}")
        results_file.rename(backup_file)
        return {"results": {}}


def save_results(results_file: Path, results: Dict):
    """Save results with NumPy type conversion."""
    results_file.parent.mkdir(parents=True, exist_ok=True)
    results_converted = convert_to_native_types(results)
    with open(results_file, 'w') as f:
        json.dump(results_converted, f, indent=2)


def get_phase3_configs_for_model(model_name: str) -> List:
    """Get Phase 3 configs for a specific model."""
    # YOLO World: first 3 configs
    # SAM3: last 3 configs
    if model_name.lower() == "yolo_world":
        return PHASE3_CONFIGS[:3]
    elif model_name.lower() == "sam3":
        return PHASE3_CONFIGS[3:]
    else:
        return PHASE3_CONFIGS


def config_to_dict(config) -> Dict:
    """Convert PromptConfig to dictionary for easier handling."""
    return {
        "name": config.name,
        "prompt": config.prompt,
        "description": config.description,
        "absorber_classes": config.absorber_classes,
        "target_indices": config.target_indices,
    }


def run_confidence_sweep(
    model_name: str,
    evaluator: Evaluator,
    config: Dict,
    results_file: Path,
    conf_thresholds: List[float],
    resume: bool = True,
):
    """Run confidence threshold sweep for a single config."""
    config_name = config["name"]
    
    print(f"\n{'='*80}")
    print(f"Running Confidence Sweep: {model_name} - {config_name}")
    print(f"{'='*80}")
    print(f"Prompt: {config['prompt']}")
    print(f"Description: {config.get('description', 'N/A')}")
    print(f"Confidence thresholds: {len(conf_thresholds)} values ({min(conf_thresholds):.2f} to {max(conf_thresholds):.2f})")
    
    results = load_results(results_file) if resume else {"results": {}}
    
    # Check if this config is already complete or partially complete
    completed_thresholds = set()
    if resume and config_name in results.get("results", {}):
        existing = results["results"][config_name]
        existing_thresholds = set(str(t) for t in existing.get("conf_thresholds", []))
        requested_thresholds = set(str(t) for t in conf_thresholds)
        results_by_conf = existing.get("results_by_conf", {})
        completed_thresholds = set(results_by_conf.keys())
        
        if existing_thresholds == requested_thresholds and len(completed_thresholds) == len(requested_thresholds):
            print(f"\n✅ Config {config_name} already complete. Skipping.")
            return
        elif completed_thresholds:
            print(f"\n📋 Resuming {config_name}: {len(completed_thresholds)}/{len(conf_thresholds)} thresholds already complete")
            # Use existing structure
            sweep_result = existing
        else:
            # Initialize new structure
            sweep_result = {
                "config_name": config_name,
                "description": config.get("description", ""),
                "prompt": config["prompt"],
                "model": model_name,
                "iou_threshold": 0.5,
                "conf_thresholds": conf_thresholds,
                "results_by_conf": {},
            }
            if "results" not in results:
                results["results"] = {}
            results["results"][config_name] = sweep_result
    else:
        # Initialize new structure
        sweep_result = {
            "config_name": config_name,
            "description": config.get("description", ""),
            "prompt": config["prompt"],
            "model": model_name,
            "iou_threshold": 0.5,
            "conf_thresholds": conf_thresholds,
            "results_by_conf": {},
        }
        if "results" not in results:
            results["results"] = {}
        results["results"][config_name] = sweep_result
    
    # Run confidence sweep with incremental saving
    try:
        # Initialize sweep result structure
        sweep_result = {
            "config_name": config_name,
            "description": config.get("description", ""),
            "prompt": config["prompt"],
            "model": model_name,
            "iou_threshold": 0.5,
            "conf_thresholds": conf_thresholds,
            "results_by_conf": {},
        }
        
        # Store initial structure
        if "results" not in results:
            results["results"] = {}
        results["results"][config_name] = sweep_result
        
        # Run each threshold individually and save incrementally
        for conf_thresh in conf_thresholds:
            conf_thresh_str = str(conf_thresh)
            
            # Skip if already completed
            if resume and conf_thresh_str in completed_thresholds:
                print(f"\n⏭️  Skipping conf_threshold={conf_thresh} (already completed)")
                continue
            
            print(f"\nEvaluating at conf_threshold={conf_thresh}...")
            evaluator.conf_threshold = conf_thresh
            
            eval_result = evaluator.evaluate_prompt(
                prompt=config["prompt"],
                absorber_classes=config.get("absorber_classes"),
                target_indices=config.get("target_indices"),
                iou_thresholds=[0.5],
            )
            
            # Store result for this threshold
            sweep_result["results_by_conf"][conf_thresh_str] = {
                "metrics": eval_result["metrics_by_iou"]["iou_0.5"],
                "counting": eval_result["counting"],
                "total_predictions": eval_result["total_predictions"],
            }
            
            # Save incrementally after each threshold
            save_results(results_file, results)
            
            # Print progress
            metrics = eval_result["metrics_by_iou"]["iou_0.5"]
            completed = len(sweep_result["results_by_conf"])
            total = len(conf_thresholds)
            print(f"  ✓ conf={conf_thresh:.2f}: mAP={metrics['map']:.4f}, F1={metrics['f1']:.4f}")
            print(f"  💾 Saved (progress: {completed}/{total} thresholds)")
        
        # Print summary
        print(f"\n✅ Confidence sweep complete for {config_name}")
        print(f"   💾 Saved to {results_file}")
        
        # Show performance at key thresholds
        key_thresholds = [0.05, 0.10, 0.20, 0.30, 0.50]
        print(f"\n   Performance at key thresholds:")
        for thresh in key_thresholds:
            if str(thresh) in sweep_result["results_by_conf"]:
                metrics = sweep_result["results_by_conf"][str(thresh)]["metrics"]
                print(f"     conf={thresh:.2f}: mAP={metrics['map']:.4f}, F1={metrics['f1']:.4f}, "
                      f"P={metrics['precision']:.4f}, R={metrics['recall']:.4f}")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    parser = argparse.ArgumentParser(description="Run Phase 3: Confidence Threshold Analysis")
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
        "--conf-min",
        type=float,
        default=0.05,
        help="Minimum confidence threshold (default: 0.05)",
    )
    parser.add_argument(
        "--conf-max",
        type=float,
        default=0.95,
        help="Maximum confidence threshold (default: 0.95)",
    )
    parser.add_argument(
        "--conf-step",
        type=float,
        default=0.05,
        help="Confidence threshold step size (default: 0.05)",
    )
    
    args = parser.parse_args()
    
    # Get data paths
    if args.data_root:
        os.environ["DATA_ROOT"] = args.data_root
    
    data_paths = get_data_paths()
    
    # Set up results directory
    results_dir = project_root / "experiments" / "results" / "phase3_confidence_sweeps"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    logs_dir = results_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate confidence thresholds
    conf_thresholds = []
    current = args.conf_min
    while current <= args.conf_max:
        conf_thresholds.append(round(current, 2))
        current += args.conf_step
    
    print(f"Confidence threshold range: {min(conf_thresholds):.2f} to {max(conf_thresholds):.2f} (step: {args.conf_step:.2f})")
    print(f"Total thresholds: {len(conf_thresholds)}")
    
    models_to_run = []
    if args.model in ["yolo_world", "all"]:
        models_to_run.append("yolo_world")
    if args.model in ["sam3", "all"]:
        models_to_run.append("sam3")
    
    for model_name in models_to_run:
        # Set up model
        if model_name == "yolo_world":
            model = YOLOWorldModel(weights_path=args.yolo_weights)
            batch_size = 15
        else:  # sam3
            model = SAM3Model(model_id="facebook/sam3")
            batch_size = 1  # Sequential processing for SAM3
        
        # Set up evaluator
        evaluator = Evaluator(
            model=model,
            images_dir=data_paths["images_dir"],
            labels_dir=data_paths["labels_dir"],
            conf_threshold=0.1,  # Default, will be overridden in sweep
            batch_size=batch_size,
        )
        
        # Get image paths (full test set if requested)
        if args.full_test_set:
            # Load dev manifest to get dev image paths
            if data_paths["dev_manifest"].exists():
                dev_images_dir = data_paths["dev_manifest"].parent / "images"
                with open(data_paths["dev_manifest"]) as f:
                    dev_image_names = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
                    dev_images = [dev_images_dir / name for name in dev_image_names if (dev_images_dir / name).exists()]
                # Combine test and dev images
                test_images = get_image_paths(data_paths["images_dir"])
                all_images = sorted(set(test_images + dev_images))
                evaluator.image_paths = all_images
                print(f"\nUsing full test set: {len(all_images)} images (test: {len(test_images)}, dev: {len(dev_images)})")
        
        # Set up logging for this model
        log_prefix = model_name.replace("_", "-")
        log_file = setup_logging(
            log_dir=logs_dir,
            experiment_name=f"{log_prefix}_confidence_sweep",
            level=logging.INFO,
        )
        
        log_experiment_start(
            experiment_name=f"Phase 3: {model_name}",
            config={
                "model": model_name,
                "conf_thresholds": conf_thresholds,
                "n_configs": len(get_phase3_configs_for_model(model_name)),
            },
        )
        
        # Get configs for this model
        configs = get_phase3_configs_for_model(model_name)
        
        # Results file
        results_file = results_dir / f"{model_name}_confidence_sweeps.json"
        
        print(f"\n{'='*80}")
        print(f"Phase 3: Confidence Threshold Analysis - {model_name}")
        print(f"{'='*80}")
        print(f"Total configs: {len(configs)}")
        print(f"Confidence thresholds: {len(conf_thresholds)} values")
        print(f"Results file: {results_file}")
        print(f"Resume: {not args.no_resume}")
        
        # Run confidence sweeps for each config
        for idx, config in enumerate(configs, 1):
            config_dict = config_to_dict(config)
            print(f"\n[{idx}/{len(configs)}] Processing: {config_dict['name']}")
            run_confidence_sweep(
                model_name=model_name,
                evaluator=evaluator,
                config=config_dict,
                results_file=results_file,
                conf_thresholds=conf_thresholds,
                resume=not args.no_resume,
            )
        
        log_experiment_complete(
            experiment_name=f"Phase 3: {model_name}",
            results_file=results_file,
        )
        
        print(f"\n{'='*80}")
        print(f"Phase 3 Complete: {model_name}")
        print(f"{'='*80}")
    
    print(f"\n{'='*80}")
    print("Phase 3 Experiments Complete!")
    print(f"Results saved to: {results_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
