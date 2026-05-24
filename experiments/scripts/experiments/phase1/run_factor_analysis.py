#!/usr/bin/env python3
"""
Run Phase 1: Factor Analysis Experiments

One-factor-at-a-time (OFAT) analysis across 7 axes for YOLO World and SAM3.

Usage:
    python experiments/scripts/experiments/phase1/run_factor_analysis.py --model yolo_world
    python experiments/scripts/experiments/phase1/run_factor_analysis.py --model sam3
    python experiments/scripts/experiments/phase1/run_factor_analysis.py --model all
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Force unbuffered output for real-time logging
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
sys.stderr.reconfigure(line_buffering=True) if hasattr(sys.stderr, 'reconfigure') else None

# Add project root to path
# __file__ is experiments/scripts/experiments/phase1/run_factor_analysis.py
# So parent.parent.parent.parent.parent is the actual project root (AgVFM2)
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from agvfm.config.experiments import (
    FACTOR_AXES,
    build_prompt_from_components,
    generate_factor_combinations,
)
from agvfm.data import get_holdout_test_paths, get_full_test_paths
from agvfm.experiments import Evaluator, analyze_factor_contributions, run_factor_analysis
from agvfm.models import SAM3Model, YOLOWorldModel
from agvfm.utils import setup_logging, log_experiment_start, log_experiment_complete


def get_data_paths() -> Dict[str, Path]:
    """Get dataset paths. Uses holdout test set (138 images) to match previous experiments."""
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
        f"  - {agvfm_data}\n"
        f"Please set DATA_ROOT environment variable or update this script."
    )


def load_existing_results(results_file: Path) -> Dict:
    """Load existing results if file exists."""
    if results_file.exists():
        with open(results_file, 'r') as f:
            return json.load(f)
    return {}


def convert_to_native_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
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
    else:
        return obj


def save_results(results: Dict, results_file: Path):
    """Save results to JSON file."""
    results_file.parent.mkdir(parents=True, exist_ok=True)
    # Convert numpy types to native Python types
    results_serializable = convert_to_native_types(results)
    with open(results_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)


def run_model_factor_analysis(
    model_name: str,
    model,
    images_dir: Path,
    labels_dir: Path,
    results_dir: Path,
    iou_thresholds: List[float] = None,
    resume: bool = True,
    image_paths: List[Path] = None,
    no_emoji: bool = False,
) -> Dict:
    """
    Run factor analysis for a single model.
    
    Args:
        model_name: Name of model ("yolo_world" or "sam3")
        model: Model instance
        images_dir: Directory containing test images
        labels_dir: Directory containing YOLO format labels
        results_dir: Directory to save results
        iou_thresholds: List of IoU thresholds for evaluation
        resume: If True, skip already-completed configs
        
    Returns:
        Complete results dictionary
    """
    if iou_thresholds is None:
        iou_thresholds = [0.3, 0.5]
    
    # Setup evaluator with filtered image paths (for holdout test set)
    # Use batch_size=6 to better utilize GPU memory
    evaluator = Evaluator(
        model=model,
        images_dir=images_dir,
        labels_dir=labels_dir,
        conf_threshold=0.1,
        image_paths=image_paths,
        batch_size=6,  # Conservative batch size for SAM3 (memory ~3GB, can handle ~6)
    )
    
    # Results file
    results_file = results_dir / f"{model_name}_factor_analysis.json"
    
    # Load existing results if resuming
    existing_results = {}
    if resume and results_file.exists():
        existing_results = load_existing_results(results_file)
        print(f"📂 Loaded {len(existing_results.get('results', {}))} existing results")
    
    # Generate all combinations
    combinations = generate_factor_combinations()

    # Optionally strip emoji-axis variants before reporting totals
    if no_emoji:
        combinations = [c for c in combinations if not c.get("emoji", "")]
        print("⚙️  --no-emoji: emoji-axis variants excluded from this run.")

    print("=" * 80)
    print(f"Phase 1: Factor Analysis - {model.model_name}")
    print("=" * 80)
    print(f"Total combinations: {len(combinations)}")
    print(f"IoU thresholds: {iou_thresholds}")
    print(f"Images: {len(evaluator.image_paths)}")
    print(f"Results will be saved to: {results_file}")
    print()
    
    # Run experiments
    results = existing_results.get("results", {})
    
    for idx, combo in enumerate(combinations, 1):
        prompt = build_prompt_from_components(combo)
        
        # Check if this is the baseline (all axes at baseline values)
        # Note: Must check exact match to baseline value, not just empty string
        # The bare noun (grammar='') is NOT baseline (baseline is grammar='a')
        is_baseline = all(
            combo.get(axis.name, "") == (axis.baseline if axis.baseline is not None else "")
            for axis in FACTOR_AXES
        )
        
        if is_baseline:
            combo_name = "baseline"
        else:
            combo_name = "_".join([f"{k}:{v}" for k, v in combo.items() if v and v != ""])
        
        # Skip if already completed
        if resume and combo_name in results:
            print(f"[{idx}/{len(combinations)}] {combo_name} - SKIPPED (already completed)")
            continue
        
        print(f"[{idx}/{len(combinations)}] {combo_name}")
        print(f"  Prompt: {prompt}")
        
        # Log to file as well
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"[{idx}/{len(combinations)}] Starting: {combo_name} - '{prompt}'")
        
        try:
            eval_result = evaluator.evaluate_prompt(
                prompt=prompt,
                iou_thresholds=iou_thresholds,
            )
            
            results[combo_name] = {
                "components": combo,
                "prompt": prompt,
                **eval_result,
            }
            
            # Print summary
            metrics_05 = eval_result["metrics_by_iou"]["iou_0.5"]
            print(f"  ✓ mAP@0.5: {metrics_05['map']:.4f}, F1: {metrics_05['f1']:.4f}")
            logger.info(f"[{idx}/{len(combinations)}] Completed: {combo_name} - mAP@0.5: {metrics_05['map']:.4f}, F1: {metrics_05['f1']:.4f}")
            
            # Save incrementally
            full_results = {
                "experiment_type": "factor_analysis",
                "model": model.model_name,
                "n_combinations": len(combinations),
                "iou_thresholds": iou_thresholds,
                "test_set": "full" if len(evaluator.image_paths) == 158 else "holdout",
                "n_images": len(evaluator.image_paths),
                "results": results,
            }
            save_results(full_results, results_file)
            print(f"  💾 Saved to {results_file}")
            logger.info(f"  Saved progress: {len(results)}/{len(combinations)} configurations complete")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            # Continue with next config
        
        print()
    
    # Final save
    full_results = {
        "experiment_type": "factor_analysis",
        "model": model.model_name,
        "n_combinations": len(combinations),
        "iou_thresholds": iou_thresholds,
        "results": results,
    }
    save_results(full_results, results_file)
    
    print("=" * 80)
    print(f"✅ Factor analysis complete for {model.model_name}")
    print(f"   Results saved to: {results_file}")
    print("=" * 80)
    
    return full_results


def main():
    parser = argparse.ArgumentParser(description="Run Phase 1: Factor Analysis Experiments")
    parser.add_argument(
        "--model",
        type=str,
        choices=["yolo_world", "sam3", "all"],
        default="all",
        help="Model to run (default: all)",
    )
    parser.add_argument(
        "--yolo-weights",
        type=str,
        default="model_weights/yolo_world/yolov8x-worldv2.pt",
        help="Path to YOLO World weights (default: model_weights/yolo_world/yolov8x-worldv2.pt)",
    )
    parser.add_argument(
        "--sam3-model-id",
        type=str,
        default="facebook/sam3",
        help="SAM3 Hugging Face model ID (default: facebook/sam3)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (default: cuda)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Results directory (default: experiments/results/phase1_factor_analysis)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume from existing results (start fresh)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Root directory of dataset (overrides auto-detection)",
    )
    parser.add_argument(
        "--full-test-set",
        action="store_true",
        help="Use full test set (158 images) instead of holdout test set (138 images)",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Directory containing test images (overrides auto-detection)",
    )
    parser.add_argument(
        "--labels-dir",
        type=str,
        default=None,
        help="Directory containing test labels (overrides auto-detection)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        metavar="N",
        help="Randomly sample N images from the test set (reproducible; seed=42). "
             "Default: use all images.",
    )
    parser.add_argument(
        "--no-emoji",
        action="store_true",
        help="Skip all emoji-axis variants (combinations where the 'emoji' factor "
             "is set to a non-baseline value). All other factor axes are still tested.",
    )

    args = parser.parse_args()
    
    # Immediate output to verify script is running
    print("=" * 80, flush=True)
    print("Phase 1: Factor Analysis Experiments", flush=True)
    print("=" * 80, flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"Device: {args.device}", flush=True)
    print(f"Full test set: {args.full_test_set}", flush=True)
    print("", flush=True)
    
    # Setup paths
    if args.data_root:
        data_root = Path(args.data_root)
        images_dir = data_root / "test" / "images"
        labels_dir = data_root / "test" / "labels"
        dev_manifest = data_root / "dev" / "manifest.txt"
    elif args.image_dir and args.labels_dir:
        images_dir = Path(args.image_dir)
        labels_dir = Path(args.labels_dir)
        dev_manifest = None  # Can't use holdout test set if custom paths are provided
    else:
        paths = get_data_paths()
        images_dir = paths["images_dir"]
        labels_dir = paths["labels_dir"]
        dev_manifest = paths["dev_manifest"]
    
    if not images_dir.exists() or not labels_dir.exists():
        print(f"❌ Dataset not found:")
        print(f"   Images: {images_dir}")
        print(f"   Labels: {labels_dir}")
        sys.exit(1)
    
    # Get test set (full or holdout)
    if args.full_test_set or dev_manifest is None:
        image_paths = get_full_test_paths(images_dir)
        print(f"📊 Using FULL test set: {len(image_paths)} images (includes dev images)")
    else:
        image_paths = get_holdout_test_paths(images_dir, dev_manifest)
        print(f"📊 Using holdout test set: {len(image_paths)} images (excluded dev images)")

    # Optional random sub-sample
    if args.sample_size is not None:
        if args.sample_size >= len(image_paths):
            print(f"⚠️  --sample-size {args.sample_size} ≥ dataset size {len(image_paths)}; "
                  f"using all images.")
        else:
            rng = random.Random(42)
            image_paths = rng.sample(image_paths, args.sample_size)
            print(f"🎲 Sampled {len(image_paths)} images (seed=42) from dataset")
    
    # Results directory
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        # project_root is already AgVFM2, so just add experiments/results/phase1_factor_analysis
        results_dir = project_root / "experiments" / "results" / "phase1_factor_analysis"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging with model-specific prefix (use dashes instead of underscores for easier parsing)
    log_dir = results_dir / "logs"
    if args.model in ["yolo_world", "sam3"]:
        # Replace underscores with dashes in model name for cleaner parsing
        model_name_dashed = args.model.replace("_", "-")
        log_prefix = f"{model_name_dashed}_run"
    else:
        log_prefix = "phase1_factor_analysis"
    log_file = setup_logging(log_dir, log_prefix)
    
    import logging
    logger = logging.getLogger(__name__)
    
    # Log experiment start
    config = {
        "model": args.model,
        "device": args.device,
        "resume": not args.no_resume,
        "test_set": "full" if args.full_test_set else "holdout",
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "n_images": len(image_paths),
        "results_dir": str(results_dir),
    }
    log_experiment_start("Phase 1: Factor Analysis", config)
    
    print("=" * 80)
    print("Phase 1: Factor Analysis Experiments")
    print("=" * 80)
    print(f"Dataset:")
    print(f"  Images: {images_dir}")
    print(f"  Labels: {labels_dir}")
    print(f"Results: {results_dir}")
    print()
    
    # Run experiments
    if args.model in ["yolo_world", "all"]:
        print("Loading YOLO World model...")
        yolo_model = YOLOWorldModel(weights_path=args.yolo_weights)
        run_model_factor_analysis(
            model_name="yolo_world",
            model=yolo_model,
            images_dir=images_dir,
            labels_dir=labels_dir,
            results_dir=results_dir,
            resume=not args.no_resume,
            image_paths=image_paths,
            no_emoji=args.no_emoji,
        )
        print()
    
    if args.model in ["sam3", "all"]:
        print("Loading SAM3 model...")
        sam3_model = SAM3Model(model_id=args.sam3_model_id, device=args.device)
        run_model_factor_analysis(
            model_name="sam3",
            model=sam3_model,
            images_dir=images_dir,
            labels_dir=labels_dir,
            results_dir=results_dir,
            resume=not args.no_resume,
            image_paths=image_paths,
            no_emoji=args.no_emoji,
        )
        print()
    
    log_experiment_complete("Phase 1: Factor Analysis", results_dir)
    print("=" * 80)
    print("✅ All experiments complete!")
    print(f"📊 Results: {results_dir}")
    print(f"📝 Logs: {log_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
