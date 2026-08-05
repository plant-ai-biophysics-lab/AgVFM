#!/usr/bin/env python3
"""Axis-based OFAT + combinatorial prompt optimisation entry point (single dataset).

Runs :func:`agvfm.optimizer.loop.run_optimization` — the same OFAT (Phase 1)
+ combinatorial-sweep + negation + emoji (Phase 2) algorithm as
``load_and_run.py``, generalised beyond the hardcoded cowpea-flower
``FACTOR_AXES`` via :class:`~agvfm.optimizer.axes.PromptAxes` so it can run
against any AgML or on-disk dataset. Supports two data sources:

  • YOLO disk data  (--img-dir / --lbl-dir / --classes)
  • AgML datasets   (--agml-dataset / --agml-classes)

For pooling this same search across *many* training datasets at once and
transferring the result zero-shot to held-out crops, see ``run_template.py``
(PAPER.md's shift from dataset-by-dataset comparison to cross-dataset
template discovery + transfer).

Usage
-----
    # YOLO disk data
    python run.py \\
        --img-dir /data/cowpea/images \\
        --lbl-dir /data/cowpea/labels \\
        --classes flower \\
        --crop "cowpea flower" \\
        --model yolo_world \\
        --llm-url http://localhost:8000/v1 \\
        --llm-model meta-llama/Llama-3.2-1B-Instruct

    # AgML dataset
    python run.py \\
        --agml-dataset grape_detection_californiaday \\
        --agml-classes grape \\
        --crop grape \\
        --model owlv2 \\
        --llm-url http://localhost:8000/v1 \\
        --llm-model meta-llama/Llama-3.2-1B-Instruct

    # No LLM server: pull a model from HuggingFace and run it locally instead
    # (omit --llm-url)
    python run.py \\
        --agml-dataset grape_detection_californiaday \\
        --agml-classes grape \\
        --crop grape \\
        --model owlv2 \\
        --llm-model Qwen/Qwen3-4B
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

from agvfm.llm.client import LLMClient
from agvfm.optimizer.loop import OptimizationConfig, run_optimization
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
        description="Axis-based OFAT + combinatorial prompt optimisation (single dataset)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data source ──────────────────────────────────────────────────────
    data_grp = p.add_argument_group("Data source (choose one)")
    data_grp.add_argument("--img-dir", default=None, metavar="PATH")
    data_grp.add_argument("--lbl-dir", default=None, metavar="PATH")
    data_grp.add_argument("--classes", nargs="+", default=None, metavar="CLASS")
    data_grp.add_argument("--agml-dataset", default=None, metavar="NAME")
    data_grp.add_argument("--agml-classes", nargs="+", default=None, metavar="CLASS")
    data_grp.add_argument("--agml-data-root", default=None, metavar="PATH",
                          help="Override HuggingFace dataset cache directory (default: ~/.cache/huggingface/)")

    # ── Model ────────────────────────────────────────────────────────────
    p.add_argument("--model", choices=["yolo_world", "grounding_dino", "owlv2", "all"],
                   default="yolo_world")
    p.add_argument("--yolo-weights", default=None, metavar="PATH")
    p.add_argument("--gdino-checkpoint", default="IDEA-Research/grounding-dino-base")
    p.add_argument("--owlv2-checkpoint", default="google/owlv2-base-patch16-ensemble")
    p.add_argument("--device", default="cuda")

    # ── Crop ─────────────────────────────────────────────────────────────
    p.add_argument("--crop", required=True, metavar="NAME")

    # ── LLM (axis value generation) ─────────────────────────────────────
    p.add_argument("--llm-url", default=None, metavar="URL",
                   help="Base URL of an OpenAI-compatible LLM server. Omit to pull "
                        "--llm-model from HuggingFace and run it locally instead.")
    p.add_argument("--llm-model", default="Qwen/Qwen3-4B", metavar="NAME",
                   help="Model name understood by the LLM server (served backend) or "
                        "a HuggingFace Hub model id to load locally (local backend).")
    p.add_argument("--llm-device", default="cuda",
                   help="Device for the local HF LLM backend (ignored when --llm-url is set)")
    p.add_argument("--llm-temperature", type=float, default=0.7)
    p.add_argument("--llm-max-tokens", type=int, default=512)

    # ── Optimisation hyperparameters ─────────────────────────────────────
    p.add_argument("--proxy-images", type=int, nargs="+", default=[30],
                   help="Proxy images per evaluation; pass multiple values to sweep e.g. 1 5 10 30")
    p.add_argument("--top-n-negation", type=int, default=5,
                   help="Top base prompts to append negation variants to")
    p.add_argument("--train-split", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)

    # ── Output ───────────────────────────────────────────────────────────
    p.add_argument("--output-dir", default="experiments/results/axis", metavar="PATH")

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
            for cand in candidates:
                if cand.exists():
                    weights = str(cand)
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

    llm = LLMClient(
        base_url=args.llm_url,
        model=args.llm_model,
        temperature=args.llm_temperature,
        max_tokens=args.llm_max_tokens,
        device=args.llm_device,
    )

    logger.info("Generating axis values...")
    axis_values_per_class = {
        class_name: llm.generate_axis_values(crop=args.crop, taxonomy=class_name)
        for class_name in dataset.classes
    }
    for class_name, av in axis_values_per_class.items():
        logger.info(f"  [{class_name}] axis_values: {av}")

    opt_config = OptimizationConfig(
        proxy_images=args.proxy_images if len(args.proxy_images) > 1 else args.proxy_images[0],
        top_n_negation=args.top_n_negation,
        seed=args.seed,
    )

    model_keys = _ALL_MODELS if args.model == "all" else [args.model]
    all_dataset_results: dict[str, list] = {}

    for model_key in model_keys:
        logger.info(f"\n{'='*60}\nModel: {model_key}\n{'='*60}")
        try:
            model = _build_model(model_key, args)
        except Exception as exc:
            logger.error(f"Failed to load {model_key}: {exc}", exc_info=True)
            continue

        try:
            results = run_optimization(
                dataset=dataset,
                model=model,
                axis_values_per_class=axis_values_per_class,
                crop=args.crop,
                config=opt_config,
                output_dir=output_dir / dataset.name / model_key,
            )
        except Exception as exc:
            logger.error(f"Axis-based optimization failed for {model_key}: {exc}", exc_info=True)
        else:
            all_dataset_results[model_key] = results
            logger.info(f"\n{'─'*60}\nSummary — {model_key}")
            for r in results:
                gain = r.best_map - r.baseline_map
                logger.info(
                    f"  [{r.class_name}]  baseline={r.baseline_map:.4f}  "
                    f"best={r.best_map:.4f}  gain={gain:+.4f}  prompt={r.best_prompt!r}"
                )

        try:
            import torch
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    _save_dataset_summary(dataset.name, all_dataset_results, output_dir / dataset.name)
    logger.info(f"\nDone. Results written to {output_dir}/")


def _save_dataset_summary(dataset_name: str, all_results: dict[str, list], output_dir: Path) -> None:
    if not all_results:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.txt"
    lines: list[str] = []

    def w(s: str = "") -> None:
        lines.append(s)

    sep = "=" * 80
    w(sep)
    w(f"  DATASET SUMMARY: {dataset_name}")
    w(sep)

    classes: list[str] = []
    for results in all_results.values():
        for r in results:
            if r.class_name not in classes:
                classes.append(r.class_name)

    models = list(all_results.keys())

    for class_name in classes:
        w()
        w(f"  Class: {class_name!r}")
        w()

        w(f"  {'Model':<20} {'Baseline':>10} {'Best':>10} {'Gain':>10}  Best Prompt")
        w(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}  {'-'*40}")
        for model_name in models:
            r = next((x for x in all_results[model_name] if x.class_name == class_name), None)
            if r is None:
                continue
            gain = r.best_map - r.baseline_map
            w(f"  {model_name:<20} {r.baseline_map:>10.4f} {r.best_map:>10.4f} {gain:>+10.4f}  {r.best_prompt!r}")
            axes_str = ", ".join(
                f"{k}={v!r}" for k, v in r.best_axes.items() if v and k != "taxonomy"
            )
            if axes_str:
                w(f"  {'':<20} {'':<10} {'':<10} {'':<10}  [{axes_str}]")

        for model_name in models:
            r = next((x for x in all_results[model_name] if x.class_name == class_name), None)
            if r is None or not r.ofat_summary:
                continue
            w()
            w(f"  OFAT [{model_name}] — axis informativeness (delta vs ofat-baseline):")
            w(f"  {'Axis':<12} {'Best Value':<24} {'mAP':>8} {'Delta':>8}")
            w(f"  {'-'*12} {'-'*24} {'-'*8} {'-'*8}")
            for o in sorted(r.ofat_summary, key=lambda x: x.delta, reverse=True):
                bar = "█" * max(0, round(o.delta * 100))
                w(f"  {o.axis:<12} {o.best_value!r:<24} {o.best_map:>8.4f} {o.delta:>+8.4f}  {bar}")

        for model_name in models:
            r = next((x for x in all_results[model_name] if x.class_name == class_name), None)
            if r is None:
                continue
            w()
            w(f"  Combinatorial [{model_name}] — best per phase (delta vs baseline):")
            w(f"  {'Phase':<12} {'mAP':>8} {'Delta':>8}  Prompt")
            w(f"  {'-'*12} {'-'*8} {'-'*8}  {'-'*40}")
            for pb in r.phase_best:
                marker = " ←" if pb.prompt == r.best_prompt else ""
                w(f"  {pb.phase:<12} {pb.map_score:>8.4f} {pb.delta:>+8.4f}  {pb.prompt!r}{marker}")

    w()
    w(sep)

    summary_path.write_text("\n".join(lines))
    logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
