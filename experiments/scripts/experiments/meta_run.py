#!/usr/bin/env python3
"""Meta-prompt optimisation entry point.

⚠️  Under development — LLM-guided iterative prompt search is experimental.

Uses an OpenAI-compatible LLM server to iteratively propose and evaluate
detection prompts, keeping the best across iterations (patience-based stopping).

Supports two data sources:

  • YOLO disk data  (--img-dir / --lbl-dir / --classes)
  • AgML datasets   (--agml-dataset / --agml-classes)

Usage
-----
    # YOLO disk data
    python meta_run.py \\
        --img-dir /data/cowpea/images \\
        --lbl-dir /data/cowpea/labels \\
        --classes flower \\
        --crop "cowpea flower" \\
        --model yolo_world \\
        --llm-url http://localhost:8000/v1 \\
        --llm-model meta-llama/Llama-3.2-1B-Instruct

    # AgML dataset, stricter patience
    python meta_run.py \\
        --agml-dataset grape_detection_californiaday \\
        --agml-classes grape \\
        --crop grape \\
        --model owlv2 \\
        --llm-url http://localhost:8000/v1 \\
        --llm-model meta-llama/Llama-3.2-1B-Instruct \\
        --patience 5 --candidates 3
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None
sys.stderr.reconfigure(line_buffering=True) if hasattr(sys.stderr, "reconfigure") else None

project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from agvfm.llm.meta_client import VLMClient
from agvfm.optimizer.meta_prompt import MetaPromptConfig, run_meta_prompt_optimization
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
        description="Meta-prompt optimisation (experimental)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data source ──────────────────────────────────────────────────────
    data_grp = p.add_argument_group("Data source (choose one)")
    data_grp.add_argument("--img-dir", default=None, metavar="PATH")
    data_grp.add_argument("--lbl-dir", default=None, metavar="PATH")
    data_grp.add_argument("--classes", nargs="+", default=None, metavar="CLASS")
    data_grp.add_argument("--agml-dataset", default=None, metavar="NAME")
    data_grp.add_argument("--agml-classes", nargs="+", default=None, metavar="CLASS")
    data_grp.add_argument("--agml-data-root", default=None, metavar="PATH")

    # ── Model ────────────────────────────────────────────────────────────
    p.add_argument("--model", choices=["yolo_world", "grounding_dino", "owlv2", "all"],
                   default="yolo_world")
    p.add_argument("--yolo-weights", default=None, metavar="PATH")
    p.add_argument("--gdino-checkpoint", default="IDEA-Research/grounding-dino-base")
    p.add_argument("--owlv2-checkpoint", default="google/owlv2-base-patch16-ensemble")
    p.add_argument("--device", default="cuda")

    # ── Crop ─────────────────────────────────────────────────────────────
    p.add_argument("--crop", required=True, metavar="NAME")

    # ── LLM ──────────────────────────────────────────────────────────────
    p.add_argument("--llm-url", required=True, metavar="URL",
                   help="Base URL of an OpenAI-compatible LLM server")
    p.add_argument("--llm-model", required=True, metavar="NAME",
                   help="Model name understood by the LLM server")
    p.add_argument("--llm-temperature", type=float, default=0.8)
    p.add_argument("--llm-max-tokens", type=int, default=512)

    # ── Meta-prompt hyperparameters ──────────────────────────────────────
    p.add_argument("--proxy-images", type=int, default=30)
    p.add_argument("--patience", type=int, default=10,
                   help="Iterations without improvement before stopping")
    p.add_argument("--candidates", type=int, default=5,
                   help="New prompt candidates requested from LLM per iteration")
    p.add_argument("--max-iterations", type=int, default=100)
    p.add_argument("--train-split", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)

    # ── Output ───────────────────────────────────────────────────────────
    p.add_argument("--output-dir", default="experiments/results/meta", metavar="PATH")

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
# Model factory
# ---------------------------------------------------------------------------

def _build_model(model_key: str, args: argparse.Namespace) -> AgVFMAdapter:
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
        return AgVFMAdapter(YOLOWorldModel(weights_path=weights, device=device), name=model_key)

    if model_key == "grounding_dino":
        return AgVFMAdapter(
            GroundingDINOModel(model_id=args.gdino_checkpoint, device=device),
            name=model_key,
        )

    if model_key == "owlv2":
        return AgVFMAdapter(
            OWLv2Model(model_id=args.owlv2_checkpoint, device=device),
            name=model_key,
        )

    raise ValueError(f"Unknown model: {model_key!r}")


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

    vlm = VLMClient(
        base_url=args.llm_url,
        model=args.llm_model,
        temperature=args.llm_temperature,
        max_tokens=args.llm_max_tokens,
    )

    meta_config = MetaPromptConfig(
        proxy_images=args.proxy_images,
        patience=args.patience,
        candidates_per_iter=args.candidates,
        max_iterations=args.max_iterations,
        seed=args.seed,
    )

    model_keys = _ALL_MODELS if args.model == "all" else [args.model]

    for model_key in model_keys:
        logger.info(f"\n{'='*60}\nModel: {model_key}\n{'='*60}")
        try:
            model = _build_model(model_key, args)
        except Exception as exc:
            logger.error(f"Failed to load {model_key}: {exc}", exc_info=True)
            continue

        model_output_dir = output_dir / model_key
        try:
            results = run_meta_prompt_optimization(
                dataset=dataset,
                model=model,
                vlm=vlm,
                crop=args.crop,
                config=meta_config,
                output_dir=model_output_dir,
            )
        except Exception as exc:
            logger.error(f"Meta-prompt optimization failed for {model_key}: {exc}", exc_info=True)
        else:
            logger.info(f"\n{'─'*60}\nSummary — {model_key}")
            for r in results:
                gain = r.best_map - r.baseline_map
                logger.info(
                    f"  [{r.class_name}]  baseline={r.baseline_map:.4f}  "
                    f"best={r.best_map:.4f}  gain={gain:+.4f}  "
                    f"iters={r.total_iterations}  prompt={r.best_prompt!r}"
                )

        try:
            import torch
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    logger.info(f"\nDone. Results written to {output_dir}/")


if __name__ == "__main__":
    main()
