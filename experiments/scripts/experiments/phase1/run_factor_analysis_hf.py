#!/usr/bin/env python3
"""
Run Phase 1: Factor Analysis — HuggingFace OVD Models

One-factor-at-a-time (OFAT) analysis across 7 axes for
GroundingDINO and OWLv2.  Produces the same result JSON schema and uses the
same Evaluator / metrics pipeline as the YOLO World / SAM3 runner
(run_factor_analysis.py), so outputs are directly comparable.

Usage:
    python experiments/scripts/experiments/phase1/run_factor_analysis_hf.py \\
        --model grounding_dino

    python experiments/scripts/experiments/phase1/run_factor_analysis_hf.py \\
        --model owlv2

    python experiments/scripts/experiments/phase1/run_factor_analysis_hf.py \\
        --model all --no-emoji --sample-size 50

CLI flags
---------
--model                 grounding_dino | owlv2 | all  (default: all)
--gdino-model-id        HF model ID for GroundingDINO
                        (default: IDEA-Research/grounding-dino-base)
--owlv2-model-id        HF model ID for OWLv2
                        (default: google/owlv2-base-patch16-ensemble)
--gdino-box-threshold   box_threshold for GDino post-processor (default: 0.3)
--gdino-text-threshold  text_threshold for GDino post-processor (default: 0.25)
--device                cuda | cpu  (default: cuda)
--results-dir           output directory
                        (default: experiments/results/phase1_factor_analysis_hf)
--no-resume             start fresh, ignoring existing results
--data-root             explicit root of the _data/ tree
--image-dir             explicit test images directory
--labels-dir            explicit test labels directory
--full-test-set         use all 158 images (default: 138-image holdout set)
--sample-size N         randomly sample N images (seed=42, reproducible)
--no-emoji              skip emoji-axis variants; test all other factor axes
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Force unbuffered output for real-time cluster logging
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None
sys.stderr.reconfigure(line_buffering=True) if hasattr(sys.stderr, "reconfigure") else None

# ---------------------------------------------------------------------------
# Path setup — __file__ is
#   experiments/scripts/experiments/phase1/run_factor_analysis_hf.py
# parent×5 = project root (AgVFM)
# ---------------------------------------------------------------------------
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from agvfm.config.experiments import (
    FACTOR_AXES,
    build_prompt_from_components,
    generate_factor_combinations,
)
from agvfm.data import get_holdout_test_paths, get_full_test_paths
from agvfm.experiments import Evaluator
from agvfm.models import GroundingDINOModel, OWLv2Model
from agvfm.utils import setup_logging, log_experiment_start, log_experiment_complete


# ---------------------------------------------------------------------------
# Data-path resolution (mirrors run_factor_analysis.py exactly)
# ---------------------------------------------------------------------------

_DATASET_SUBPATH = (
    "_data/T4_REAL_FLOWER_SOBJ.v18-train_2023_test_location_genotype"
)


def get_data_paths() -> Dict[str, Path]:
    """Resolve dataset paths, trying several standard locations."""
    data_root_env = os.environ.get("DATA_ROOT")

    candidates: List[Path] = []
    if data_root_env:
        candidates.append(Path(data_root_env) / _DATASET_SUBPATH)
    candidates += [
        project_root / _DATASET_SUBPATH,
        Path("/data2/jmearles/AgVFM2") / _DATASET_SUBPATH,
        project_root.parent / "AgVFM" / _DATASET_SUBPATH,
    ]

    for base in candidates:
        if base.exists():
            return {
                "images_dir": base / "test" / "images",
                "labels_dir": base / "test" / "labels",
                "dev_manifest": base / "dev" / "manifest.txt",
            }

    raise FileNotFoundError(
        "Dataset not found. Tried:\n"
        + "\n".join(f"  - {p}" for p in candidates)
        + "\nSet DATA_ROOT or pass --data-root / --image-dir + --labels-dir."
    )


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def convert_to_native_types(obj):
    """Recursively convert numpy scalars/arrays to plain Python types."""
    import numpy as np

    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_to_native_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_native_types(v) for v in obj]
    return obj


def load_existing_results(results_file: Path) -> Dict:
    if results_file.exists():
        with open(results_file, "r") as fh:
            return json.load(fh)
    return {}


def save_results(results: Dict, results_file: Path) -> None:
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "w") as fh:
        json.dump(convert_to_native_types(results), fh, indent=2)


# ---------------------------------------------------------------------------
# Core experiment loop
# ---------------------------------------------------------------------------

def run_model_factor_analysis(
    model_name: str,
    model,
    images_dir: Path,
    labels_dir: Path,
    results_dir: Path,
    iou_thresholds: Optional[List[float]] = None,
    resume: bool = True,
    image_paths: Optional[List[Path]] = None,
    no_emoji: bool = False,
) -> Dict:
    """
    Run OFAT factor analysis for a single model.

    Args:
        model_name:      Short identifier used in the results filename
                         (e.g. ``"grounding_dino"``).
        model:           Model instance (must implement ``BaseModel``).
        images_dir:      Directory containing test images.
        labels_dir:      Directory containing YOLO-format labels.
        results_dir:     Directory where the JSON results file is written.
        iou_thresholds:  IoU thresholds for evaluation (default: [0.3, 0.5]).
        resume:          If True, skip already-completed configurations.
        image_paths:     Pre-filtered list of image paths (overrides auto-load).
        no_emoji:        If True, strip emoji-axis variants from the sweep.

    Returns:
        Complete results dictionary.
    """
    if iou_thresholds is None:
        iou_thresholds = [0.3, 0.5]

    # Evaluator — batch_size=1 is safe for both GDino and OWLv2 (sequential
    # predict_batch); increase if memory / throughput profiling justifies it.
    evaluator = Evaluator(
        model=model,
        images_dir=images_dir,
        labels_dir=labels_dir,
        conf_threshold=0.1,
        image_paths=image_paths,
        batch_size=1,
    )

    results_file = results_dir / f"{model_name}_factor_analysis.json"

    existing_results: Dict = {}
    if resume and results_file.exists():
        existing_results = load_existing_results(results_file)
        print(f"📂 Loaded {len(existing_results.get('results', {}))} existing results")

    # Build combination list
    combinations = generate_factor_combinations()

    if no_emoji:
        combinations = [c for c in combinations if not c.get("emoji", "")]
        print("⚙️  --no-emoji: emoji-axis variants excluded from this run.")

    print("=" * 80)
    print(f"Phase 1 (HF): Factor Analysis — {model.model_name}")
    print("=" * 80)
    print(f"Total combinations : {len(combinations)}")
    print(f"IoU thresholds     : {iou_thresholds}")
    print(f"Images             : {len(evaluator.image_paths)}")
    print(f"Results file       : {results_file}")
    print()

    results = existing_results.get("results", {})
    logger = logging.getLogger(__name__)

    for idx, combo in enumerate(combinations, 1):
        prompt = build_prompt_from_components(combo)

        # Determine combo name (mirrors run_factor_analysis.py exactly)
        is_baseline = all(
            combo.get(axis.name, "") == (axis.baseline if axis.baseline is not None else "")
            for axis in FACTOR_AXES
        )
        if is_baseline:
            combo_name = "baseline"
        else:
            combo_name = "_".join([f"{k}:{v}" for k, v in combo.items() if v and v != ""])

        if resume and combo_name in results:
            print(f"[{idx}/{len(combinations)}] {combo_name} — SKIPPED (already completed)")
            continue

        print(f"[{idx}/{len(combinations)}] {combo_name}")
        print(f"  Prompt: {prompt}")
        logger.info(f"[{idx}/{len(combinations)}] Starting: {combo_name} — '{prompt}'")

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

            m05 = eval_result["metrics_by_iou"]["iou_0.5"]
            print(f"  ✓ mAP@0.5: {m05['map']:.4f}  F1: {m05['f1']:.4f}")
            logger.info(
                f"[{idx}/{len(combinations)}] Completed: {combo_name} — "
                f"mAP@0.5: {m05['map']:.4f}  F1: {m05['f1']:.4f}"
            )

            # Incremental save
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
            print(f"  💾 Saved ({len(results)}/{len(combinations)} done)")
            logger.info(f"  Saved: {len(results)}/{len(combinations)} complete")

        except Exception as exc:
            print(f"  ❌ Error: {exc}")
            import traceback
            traceback.print_exc()
            # Continue — do not abort the whole sweep on a single failure

        print()

    # Final save (captures last entry + clean metadata)
    full_results = {
        "experiment_type": "factor_analysis",
        "model": model.model_name,
        "n_combinations": len(combinations),
        "iou_thresholds": iou_thresholds,
        "results": results,
    }
    save_results(full_results, results_file)

    print("=" * 80)
    print(f"✅ Factor analysis complete — {model.model_name}")
    print(f"   Results saved to: {results_file}")
    print("=" * 80)

    return full_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 1 Factor Analysis — HuggingFace OVD models "
                    "(GroundingDINO, OWLv2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model selection
    p.add_argument(
        "--model",
        choices=["grounding_dino", "owlv2", "all"],
        default="all",
        help="Which model(s) to run.",
    )

    # Model-specific IDs
    p.add_argument(
        "--gdino-model-id",
        default="IDEA-Research/grounding-dino-base",
        help="HuggingFace model ID for GroundingDINO.",
    )
    p.add_argument(
        "--owlv2-model-id",
        default="google/owlv2-base-patch16-ensemble",
        help="HuggingFace model ID for OWLv2.",
    )

    # GroundingDINO post-processor thresholds
    p.add_argument(
        "--gdino-box-threshold",
        type=float,
        default=0.3,
        metavar="T",
        help="box_threshold for GDino post-processor.",
    )
    p.add_argument(
        "--gdino-text-threshold",
        type=float,
        default=0.25,
        metavar="T",
        help="text_threshold for GDino post-processor.",
    )

    # Runtime
    p.add_argument("--device", default="cuda", help="Torch device.")

    # Output
    p.add_argument(
        "--results-dir",
        default=None,
        help="Directory for results JSON files "
             "(default: experiments/results/phase1_factor_analysis_hf).",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing results and start fresh.",
    )

    # Data paths
    p.add_argument("--data-root", default=None, help="Root of the _data/ tree.")
    p.add_argument("--image-dir", default=None, help="Test images directory.")
    p.add_argument("--labels-dir", default=None, help="Test labels directory.")
    p.add_argument(
        "--full-test-set",
        action="store_true",
        help="Use the full 158-image test set instead of the 138-image holdout set.",
    )

    # Sampling / filtering
    p.add_argument(
        "--sample-size",
        type=int,
        default=None,
        metavar="N",
        help="Randomly sample N images (seed=42, reproducible). Default: all images.",
    )
    p.add_argument(
        "--no-emoji",
        action="store_true",
        help="Skip emoji-axis variants; all other factor axes are still swept.",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 80, flush=True)
    print("Phase 1: Factor Analysis — HuggingFace OVD Models", flush=True)
    print("=" * 80, flush=True)
    print(f"Model   : {args.model}", flush=True)
    print(f"Device  : {args.device}", flush=True)
    print(f"No-emoji: {args.no_emoji}", flush=True)
    print(flush=True)

    # ------------------------------------------------------------------
    # Resolve data paths
    # ------------------------------------------------------------------
    if args.data_root:
        base = Path(args.data_root)
        images_dir  = base / "test" / "images"
        labels_dir  = base / "test" / "labels"
        dev_manifest = base / "dev" / "manifest.txt"
    elif args.image_dir and args.labels_dir:
        images_dir   = Path(args.image_dir)
        labels_dir   = Path(args.labels_dir)
        dev_manifest = None
    else:
        paths = get_data_paths()
        images_dir   = paths["images_dir"]
        labels_dir   = paths["labels_dir"]
        dev_manifest = paths["dev_manifest"]

    if not images_dir.exists() or not labels_dir.exists():
        print(f"❌ Dataset not found:\n   Images: {images_dir}\n   Labels: {labels_dir}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Build image list
    # ------------------------------------------------------------------
    if args.full_test_set or dev_manifest is None:
        image_paths = get_full_test_paths(images_dir)
        print(f"📊 Full test set : {len(image_paths)} images")
    else:
        image_paths = get_holdout_test_paths(images_dir, dev_manifest)
        print(f"📊 Holdout test set : {len(image_paths)} images")

    if args.sample_size is not None:
        if args.sample_size >= len(image_paths):
            print(
                f"⚠️  --sample-size {args.sample_size} ≥ dataset size "
                f"{len(image_paths)}; using all images."
            )
        else:
            rng = random.Random(42)
            image_paths = rng.sample(image_paths, args.sample_size)
            print(f"🎲 Sampled {len(image_paths)} images (seed=42)")

    # ------------------------------------------------------------------
    # Results & logging directories
    # ------------------------------------------------------------------
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        results_dir = project_root / "experiments" / "results" / "phase1_factor_analysis_hf"
    results_dir.mkdir(parents=True, exist_ok=True)

    log_dir = results_dir / "logs"
    if args.model in ("grounding_dino", "owlv2"):
        log_prefix = f"{args.model.replace('_', '-')}_run"
    else:
        log_prefix = "phase1_hf_factor_analysis"
    log_file = setup_logging(log_dir, log_prefix)

    config = {
        "model": args.model,
        "device": args.device,
        "resume": not args.no_resume,
        "no_emoji": args.no_emoji,
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "n_images": len(image_paths),
        "results_dir": str(results_dir),
    }
    log_experiment_start("Phase 1 (HF): Factor Analysis", config)

    print(f"Dataset  images : {images_dir}")
    print(f"Dataset  labels : {labels_dir}")
    print(f"Results dir     : {results_dir}")
    print()

    # ------------------------------------------------------------------
    # Run selected model(s)
    # ------------------------------------------------------------------
    run_gdino = args.model in ("grounding_dino", "all")
    run_owlv2 = args.model in ("owlv2", "all")

    if run_gdino:
        print(f"Loading GroundingDINO: {args.gdino_model_id} …")
        gdino_model = GroundingDINOModel(
            model_id=args.gdino_model_id,
            device=args.device,
            box_threshold=args.gdino_box_threshold,
            text_threshold=args.gdino_text_threshold,
        )
        run_model_factor_analysis(
            model_name="grounding_dino",
            model=gdino_model,
            images_dir=images_dir,
            labels_dir=labels_dir,
            results_dir=results_dir,
            resume=not args.no_resume,
            image_paths=image_paths,
            no_emoji=args.no_emoji,
        )
        # Explicitly free GPU memory before loading the next model
        del gdino_model
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print()

    if run_owlv2:
        print(f"Loading OWLv2: {args.owlv2_model_id} …")
        owlv2_model = OWLv2Model(
            model_id=args.owlv2_model_id,
            device=args.device,
        )
        run_model_factor_analysis(
            model_name="owlv2",
            model=owlv2_model,
            images_dir=images_dir,
            labels_dir=labels_dir,
            results_dir=results_dir,
            resume=not args.no_resume,
            image_paths=image_paths,
            no_emoji=args.no_emoji,
        )
        print()

    log_experiment_complete("Phase 1 (HF): Factor Analysis", results_dir)
    print("=" * 80)
    print("✅ All HF experiments complete!")
    print(f"📊 Results : {results_dir}")
    print(f"📝 Logs    : {log_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
