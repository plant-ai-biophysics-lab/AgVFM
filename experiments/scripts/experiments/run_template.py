#!/usr/bin/env python3
"""Cross-dataset prompt-*template* OFAT optimisation with zero-shot transfer.

Unlike ``run.py`` (which tunes :class:`~agvfm.optimizer.axes.PromptAxes` per
dataset/class), this script pools ~N images per class across every *train*
dataset listed in ``--datasets-file`` (every entry not flagged
``held_out: true``), sweeps structural axis values (grammar, color, size,
anatomy, phenology) against that pooled mix via
:func:`agvfm.optimizer.prompt_template.run_template_ofat` (OFAT, aggregated
across datasets), and keeps the combination that generalises best. The
resulting template is then evaluated **zero-shot** — no further tuning — on
every held-out dataset in ``--datasets-file``, both filled from bare
class/crop metadata and (optionally) from a VLM's visual read of the
held-out crop.

Only AgML-backed datasets are supported here (pooling many on-disk
directories at once adds path-management complexity with no immediate
payoff — add disk support if/when a training pool needs it). This script is
intentionally separate from ``run.py`` so the original per-dataset OFAT
pipeline is untouched — see PAPER.md sections 3.1/3.2 for how discovery vs.
constrained zero-shot transfer map onto these two scripts.

Usage
-----
    python run_template.py --datasets-file datasets_pool.yaml \\
        --llm-url http://localhost:8000/v1 --llm-model meta-llama/Llama-3.2-1B-Instruct \\
        --model yolo_world

    # Also fill held-out templates from a VLM's visual read of a sample image
    python run_template.py --datasets-file datasets_pool.yaml \\
        --llm-url http://localhost:8000/v1 --llm-model meta-llama/Llama-3.2-1B-Instruct \\
        --vlm-insight-url http://localhost:8001/v1 --vlm-insight-model google/gemma-3-4b-it \\
        --model yolo_world owlv2

``--datasets-file`` schema (YAML) — see datasets_pool.example.yaml for a fuller example
------------------------------------------------------------------------------------------
    datasets:
      - name: apple_detection_usa   # bare name -> Project-AgML/apple_detection_usa on HF
        crop: apple
        classes: [apple]
        held_out: false             # omit/false = training pool, true = zero-shot test set
      - name: grape_detection_californianight
        crop: grape
        classes: [grape]
        held_out: true
        rarity: common              # "common" | "rare" (held-out entries only; default "common")
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace") if hasattr(sys.stdout, "reconfigure") else None
sys.stderr.reconfigure(line_buffering=True, encoding="utf-8", errors="replace") if hasattr(sys.stderr, "reconfigure") else None

project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from agvfm.llm.client import LLMClient
from agvfm.llm.vlm_insight_client import VLMInsightClient
from agvfm.optimizer.prompt_template import (
    TemplateResult,
    build_train_groups,
    evaluate_zero_shot,
    generate_axis_values_per_crop,
    run_template_ofat,
    save_template_result,
)
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
        description="Cross-dataset prompt-template OFAT optimisation with zero-shot transfer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--datasets-file", required=True, metavar="PATH",
                   help="YAML file listing training-pool + held-out AgML datasets (see module docstring)")
    p.add_argument("--model", nargs="+", choices=_ALL_MODELS + ["all"], default=["yolo_world"])
    p.add_argument("--yolo-weights", default=None, metavar="PATH")
    p.add_argument("--gdino-checkpoint", default="IDEA-Research/grounding-dino-base")
    p.add_argument("--owlv2-checkpoint", default="google/owlv2-base-patch16-ensemble")
    p.add_argument("--device", default="cuda")

    p.add_argument("--llm-url", default=None, metavar="URL",
                   help="Base URL of an OpenAI-compatible LLM server (axis value generation). "
                        "Omit to pull --llm-model from HuggingFace and run it locally instead.")
    p.add_argument("--llm-model", default="Qwen/Qwen3-4B", metavar="NAME",
                   help="Model name understood by the LLM server (served backend) or a "
                        "HuggingFace Hub model id to load locally (local backend).")
    p.add_argument("--llm-device", default="cuda",
                   help="Device for the local HF LLM backend (ignored when --llm-url is set)")
    p.add_argument("--llm-temperature", type=float, default=0.7)
    p.add_argument("--llm-max-tokens", type=int, default=512)

    p.add_argument("--vlm-insight-url", default=None, metavar="URL",
                   help="Vision-capable OpenAI-compatible endpoint for zero-shot VLM template fill "
                        "(optional). Omit with --vlm-insight-model set to run that model locally instead.")
    p.add_argument("--vlm-insight-model", default=None, metavar="NAME",
                   help="Server model name (served backend) or HuggingFace Hub model id (local "
                        "backend). Setting this enables VLM-insight template fill.")
    p.add_argument("--vlm-insight-device", default="cuda",
                   help="Device for the local HF VLM-insight backend (ignored when --vlm-insight-url is set)")
    p.add_argument("--vlm-insight-temperature", type=float, default=0.2)
    p.add_argument("--vlm-insight-max-tokens", type=int, default=300)

    p.add_argument("--train-samples-per-class", type=int, default=20)
    p.add_argument("--train-split", type=float, default=0.8)
    p.add_argument("--agml-data-root", default=None, metavar="PATH",
                   help="Override HuggingFace dataset cache directory (default: ~/.cache/huggingface/)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="experiments/results/template", metavar="PATH")

    return p.parse_args()


def _load_datasets_file(path: str) -> tuple[list[dict], list[dict]]:
    with open(path) as f:
        data = yaml.safe_load(f)
    entries = data.get("datasets", [])
    train = [d for d in entries if not d.get("held_out", False)]
    test = [d for d in entries if d.get("held_out", False)]
    if not train:
        logger.error("No training-pool datasets found (all entries flagged held_out: true?)")
        sys.exit(1)
    if not test:
        logger.warning("No held-out datasets found — zero-shot section will be empty")
    return train, test


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
        return AgVFMAdapter(GroundingDINOModel(model_id=args.gdino_checkpoint, device=device), name=model_key)
    if model_key == "owlv2":
        return AgVFMAdapter(OWLv2Model(model_id=args.owlv2_checkpoint, device=device), name=model_key)
    raise ValueError(f"Unknown model: {model_key!r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    from agvfm.data.agml_loader import load_dataset

    train_ds_cfgs_list, test_ds_cfgs = _load_datasets_file(args.datasets_file)
    train_ds_cfgs = {d["name"]: d for d in train_ds_cfgs_list}

    train_splits = {}
    for name, ds_cfg in train_ds_cfgs.items():
        try:
            train_splits[name] = load_dataset(
                name, ds_cfg["classes"], train_split=args.train_split, seed=args.seed,
                data_root=args.agml_data_root,
            )
        except BaseException as exc:
            logger.warning(f"Skipping train dataset {name}: {type(exc).__name__}: {exc}", exc_info=True)

    groups = build_train_groups(train_splits, train_ds_cfgs, n_per_class=args.train_samples_per_class, seed=args.seed)
    if not groups:
        logger.error("No training groups built — aborting")
        sys.exit(1)
    logger.info(
        f"Pooled training mix: {len(groups)} (dataset, class) groups from "
        f"{len({g.dataset_name for g in groups})} datasets, "
        f"{sum(len(g.samples) for g in groups)} total images"
    )

    llm = LLMClient(
        base_url=args.llm_url, model=args.llm_model,
        temperature=args.llm_temperature, max_tokens=args.llm_max_tokens,
        device=args.llm_device,
    )
    crops = sorted({ds_cfg["crop"] for ds_cfg in train_ds_cfgs.values()})
    axes_file = Path(args.output_dir) / "factor_axes_by_crop.json"
    axis_values = generate_axis_values_per_crop(llm, crops, axes_file)
    logger.info(f"Pooled per-crop axis values for template search ({crops}): {axis_values}")

    vlm_insight = None
    if args.vlm_insight_model:
        vlm_insight = VLMInsightClient(
            base_url=args.vlm_insight_url or args.llm_url,
            model=args.vlm_insight_model,
            temperature=args.vlm_insight_temperature,
            max_tokens=args.vlm_insight_max_tokens,
            device=args.vlm_insight_device,
        )

    test_splits = []
    for ds_cfg in test_ds_cfgs:
        try:
            split = load_dataset(
                ds_cfg["name"], ds_cfg["classes"], train_split=args.train_split, seed=args.seed,
                data_root=args.agml_data_root,
            )
        except BaseException as exc:
            logger.warning(f"Skipping held-out dataset {ds_cfg['name']}: {type(exc).__name__}: {exc}", exc_info=True)
            continue
        test_splits.append((ds_cfg, split))

    model_keys = _ALL_MODELS if "all" in args.model else args.model
    output_dir = Path(args.output_dir)
    all_results: list[TemplateResult] = []

    for model_key in model_keys:
        logger.info(f"=== Model: {model_key} ===")
        try:
            model = _build_model(model_key, args)
        except Exception as exc:
            logger.error(f"Failed to load {model_key}: {exc}", exc_info=True)
            continue

        result = run_template_ofat(
            model=model,
            groups=groups,
            axis_values=axis_values,
            train_datasets=sorted(train_splits.keys()),
            test_datasets=[ds_cfg["name"] for ds_cfg, _ in test_splits],
        )

        for ds_cfg, test_split in test_splits:
            result.zero_shot.extend(
                evaluate_zero_shot(
                    model=model,
                    test_split=test_split,
                    crop=ds_cfg["crop"],
                    best_axes=result.best_axes,
                    n_eval_images=args.train_samples_per_class,
                    seed=args.seed,
                    vlm_insight=vlm_insight,
                    llm=llm,
                )
            )

        all_results.append(result)
        save_template_result(result, output_dir / model_key)

        try:
            import torch
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    _print_summary(all_results)
    logger.info(f"Done. Results written to {output_dir}/")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_summary(results: list[TemplateResult]) -> None:
    if not results:
        logger.info("No template results to summarise.")
        return

    sep = "=" * 90
    logger.info(sep)
    logger.info("  PROMPT TEMPLATE — ZERO-SHOT TRANSFER SUMMARY")
    logger.info(sep)

    for r in results:
        gain = r.best_aggregate_map - r.baseline_aggregate_map
        logger.info(
            f"  [{r.model_name}] train aggregate mAP: baseline={r.baseline_aggregate_map:.4f} "
            f"best={r.best_aggregate_map:.4f} gain={gain:+.4f}"
        )
        axes_str = ", ".join(f"{k}={v!r}" for k, v in r.best_axes.items() if v)
        logger.info(f"    best template axes: [{axes_str}]")

        if not r.zero_shot:
            continue
        logger.info(f"    {'Dataset':<28} {'Class':<20} {'Naive':>8} {'Meta':>8} {'VLM':>8}")
        for z in r.zero_shot:
            vlm_str = f"{z.template_vlm_map:>8.4f}" if z.template_vlm_map is not None else f"{'—':>8}"
            logger.info(
                f"    {z.dataset_name:<28} {z.class_name:<20} "
                f"{z.naive_map:>8.4f} {z.template_metadata_map:>8.4f} {vlm_str}"
            )

    logger.info(sep)


if __name__ == "__main__":
    main()
