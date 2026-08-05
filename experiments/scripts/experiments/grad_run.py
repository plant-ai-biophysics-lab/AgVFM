#!/usr/bin/env python3
"""PEZ gradient prompt optimisation entry point.

⚠️  Under development — gradient-based prompt search is experimental.

Runs PEZ-style continuous-embedding optimisation for all requested models and
classes.  Supports two data sources:

  • YOLO disk data  (--img-dir / --lbl-dir / --classes)
  • AgML datasets   (--agml-dataset / --agml-classes)

Usage
-----
    # YOLO disk data, composite spatial loss
    python grad_run.py \\
        --img-dir /data/cowpea/images \\
        --lbl-dir /data/cowpea/labels \\
        --classes flower \\
        --crop "cowpea flower" \\
        --model owlv2 \\
        --loss-types composite

    # AgML dataset
    python grad_run.py \\
        --agml-dataset grape_detection_californiaday \\
        --agml-classes grape \\
        --crop grape \\
        --model owlv2 \\
        --loss-types iou_cls composite

    # Quick cosine-only run
    python grad_run.py \\
        --img-dir /data/tomato/images --lbl-dir /data/tomato/labels \\
        --classes ripe unripe --crop tomato \\
        --model yolo_world --loss-types cosine --n-steps 200
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace") if hasattr(sys.stdout, "reconfigure") else None
sys.stderr.reconfigure(line_buffering=True, encoding="utf-8", errors="replace") if hasattr(sys.stderr, "reconfigure") else None

project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from agvfm.optimizer.grad_adapters import load_adapter
from agvfm.optimizer.grad_prompt import GradOptConfig, run_grad_optimization
from agvfm.optimizer.types import AgVFMAdapter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

_ALL_MODELS = ["yolo_world", "grounding_dino", "owlv2"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PEZ gradient prompt optimisation (experimental)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data source (mutually exclusive groups) ──────────────────────────
    data_grp = p.add_argument_group("Data source (choose one)")
    data_grp.add_argument("--img-dir", default=None, metavar="PATH",
                          help="Directory of images (YOLO disk mode)")
    data_grp.add_argument("--lbl-dir", default=None, metavar="PATH",
                          help="Directory of YOLO .txt labels (YOLO disk mode)")
    data_grp.add_argument("--classes", nargs="+", default=None, metavar="CLASS",
                          help="Detection class names (YOLO disk mode)")
    data_grp.add_argument("--agml-dataset", default=None, metavar="NAME",
                          help="AgML dataset name (AgML mode)")
    data_grp.add_argument("--agml-classes", nargs="+", default=None, metavar="CLASS",
                          help="Class names for AgML dataset")
    data_grp.add_argument("--agml-data-root", default=None, metavar="PATH",
                          help="Override HuggingFace dataset cache directory (default: ~/.cache/huggingface/)")

    # ── Model ────────────────────────────────────────────────────────────
    p.add_argument("--model", choices=["yolo_world", "grounding_dino", "owlv2", "all"],
                   default="owlv2")
    p.add_argument("--yolo-weights", default=None, metavar="PATH")
    p.add_argument("--gdino-checkpoint", default="IDEA-Research/grounding-dino-base")
    p.add_argument("--owlv2-checkpoint", default="google/owlv2-base-patch16-ensemble")
    p.add_argument("--device", default="cuda")

    # ── Crop / experiment ────────────────────────────────────────────────
    p.add_argument("--crop", required=True, metavar="NAME",
                   help="Human-readable crop name (used in result file names)")

    # ── PEZ hyperparameters ──────────────────────────────────────────────
    p.add_argument("--loss-types", nargs="+",
                   choices=["cosine", "iou_cls", "composite"],
                   default=["composite"],
                   help="Loss variants to run")
    p.add_argument("--n-steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--token-budget", type=int, default=8)
    p.add_argument("--fluency-lambda", type=float, default=0.1,
                   help="Fluency regularisation weight for GDINO (0 = disabled)")
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--proxy-images", type=int, default=30)
    p.add_argument("--iou-batch", type=int, default=4,
                   help="Proxy images per gradient step for iou_cls/composite loss")
    p.add_argument("--random-init", action="store_true",
                   help="Initialise prompt from random noise instead of class embeddings")
    p.add_argument("--train-split", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)

    # ── Output ───────────────────────────────────────────────────────────
    p.add_argument("--output-dir", default="experiments/results/grad", metavar="PATH")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_dataset(args: argparse.Namespace):
    if args.agml_dataset:
        from agvfm.data.agml_loader import load_dataset
        classes = args.agml_classes or [args.crop]
        logger.info(f"Loading AgML dataset: {args.agml_dataset}")
        return load_dataset(
            args.agml_dataset,
            classes=classes,
            train_split=args.train_split,
            seed=args.seed,
            data_root=args.agml_data_root,
        )
    if args.img_dir and args.lbl_dir and args.classes:
        from agvfm.data.disk_loader import load_disk_dataset
        logger.info(f"Loading disk dataset from {args.img_dir}")
        return load_disk_dataset(
            name=f"{args.crop}_disk",
            classes=args.classes,
            images_dir=Path(args.img_dir),
            labels_dir=Path(args.lbl_dir),
            train_split=args.train_split,
            seed=args.seed,
        )
    print("ERROR: Provide either --img-dir/--lbl-dir/--classes or --agml-dataset/--agml-classes")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Model + adapter factory
# ---------------------------------------------------------------------------

def _build_model_and_adapter(model_key: str, args: argparse.Namespace):
    from agvfm.models import YOLOWorldModel, GroundingDINOModel, OWLv2Model

    device = args.device

    if model_key == "yolo_world":
        weights = args.yolo_weights
        if weights is None:
            candidates = [
                project_root / "model_weights" / "yolov8x-worldv2.pt",
                Path("/group/jmearlesgrp/GEMINI/lars/grounding/model_weights/yolov8x-worldv2.pt"),
            ]
            for p in candidates:
                if p.exists():
                    weights = str(p)
                    break
            if weights is None:
                raise FileNotFoundError("YOLO World weights not found. Pass --yolo-weights PATH.")
        logger.info(f"Loading YOLO World: {weights}")
        agvfm_model = YOLOWorldModel(weights_path=weights, device=device)

    elif model_key == "grounding_dino":
        logger.info(f"Loading GroundingDINO: {args.gdino_checkpoint}")
        agvfm_model = GroundingDINOModel(model_id=args.gdino_checkpoint, device=device)

    elif model_key == "owlv2":
        logger.info(f"Loading OWLv2: {args.owlv2_checkpoint}")
        agvfm_model = OWLv2Model(model_id=args.owlv2_checkpoint, device=device)
    else:
        raise ValueError(f"Unknown model: {model_key!r}")

    detection_model = AgVFMAdapter(agvfm_model, name=model_key)
    adapter = load_adapter(model_key, agvfm_model, device=device)
    return detection_model, adapter


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = _load_dataset(args)
    logger.info(
        f"Dataset '{dataset.name}': {len(dataset.train)} train, {len(dataset.test)} test images"
    )

    grad_config = GradOptConfig(
        n_steps=args.n_steps,
        lr=args.lr,
        token_budget=args.token_budget,
        loss_types=tuple(args.loss_types),
        fluency_lambda=args.fluency_lambda,
        eval_every=args.eval_every,
        patience=args.patience,
        proxy_images=args.proxy_images,
        seed=args.seed,
        random_init=args.random_init,
        iou_loss_images_per_step=args.iou_batch,
    )

    model_keys = _ALL_MODELS if args.model == "all" else [args.model]

    for model_key in model_keys:
        logger.info(f"\n{'='*60}\nModel: {model_key}\n{'='*60}")
        try:
            detection_model, adapter = _build_model_and_adapter(model_key, args)
        except Exception as exc:
            logger.error(f"Failed to load {model_key}: {exc}", exc_info=True)
            continue

        model_output_dir = output_dir / model_key
        try:
            results = run_grad_optimization(
                dataset=dataset,
                detection_model=detection_model,
                adapter=adapter,
                crop=args.crop,
                config=grad_config,
                output_dir=model_output_dir,
            )
        except Exception as exc:
            logger.error(f"Gradient optimization failed for {model_key}: {exc}", exc_info=True)
        else:
            logger.info(f"\n{'─'*60}\nSummary — {model_key}")
            for r in results:
                gain = r.best_map - r.baseline_map
                logger.info(
                    f"  [{r.class_name}][{r.loss_type}]  "
                    f"baseline={r.baseline_map:.4f}  best={r.best_map:.4f}  "
                    f"gain={gain:+.4f}  prompt={r.best_prompt!r}"
                )

        try:
            import torch
            del detection_model, adapter
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    logger.info(f"\nDone. Results written to {output_dir}/")


if __name__ == "__main__":
    main()
