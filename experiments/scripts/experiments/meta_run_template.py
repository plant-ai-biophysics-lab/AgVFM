#!/usr/bin/env python3
"""Cross-dataset LLM prompt-*template* search with zero-shot transfer.

Unlike ``meta_run.py`` (which asks an LLM to propose full prompt strings per
dataset/class), this script pools ~N images per class across every *train*
dataset listed in ``--datasets-file`` (every entry not flagged
``held_out: true``) and asks the LLM to propose reusable **templates** —
strings containing a literal ``"{class}"`` placeholder — scored by
substituting the real class name of every pooled (dataset, class) group and
aggregating detections across all of them
(:func:`agvfm.optimizer.meta_prompt_template.run_meta_template_search`). The
winning template is then evaluated **zero-shot** on every held-out dataset in
``--datasets-file``, filling ``{class}`` from bare metadata and (optionally)
from a VLM's visual read of the held-out crop.

This is the unconstrained counterpart to ``run_template.py`` — see PAPER.md
section 3.3 for how "summary-informed zero-shot" (this script, carrying a
:class:`~agvfm.optimizer.meta_prompt.CrossRunSummary` across models) compares
against ``run_template.py``'s constrained transfer.

Usage
-----
    python meta_run_template.py --datasets-file datasets_pool.yaml \\
        --llm-url http://localhost:8000/v1 --llm-model meta-llama/Llama-3.2-11B-Vision-Instruct \\
        --model yolo_world --patience 5 --candidates 3

``--datasets-file`` schema: identical to ``run_template.py`` — see its module docstring.
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

from agvfm.llm.meta_client import VLMClient
from agvfm.llm.vlm_insight_client import VLMInsightClient
from agvfm.optimizer.meta_prompt import CrossRunSummary
from agvfm.optimizer.meta_prompt_template import (
    MetaTemplateConfig,
    MetaTemplateResult,
    evaluate_zero_shot,
    run_meta_template_search,
    save_meta_template_result,
)
from agvfm.optimizer.prompt_template import build_train_groups
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
        description="Cross-dataset LLM prompt-template search with zero-shot transfer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--datasets-file", required=True, metavar="PATH",
                   help="YAML file listing training-pool + held-out AgML datasets (see run_template.py docstring)")
    p.add_argument("--model", nargs="+", choices=_ALL_MODELS + ["all"], default=["yolo_world"])
    p.add_argument("--yolo-weights", default=None, metavar="PATH")
    p.add_argument("--gdino-checkpoint", default="IDEA-Research/grounding-dino-base")
    p.add_argument("--owlv2-checkpoint", default="google/owlv2-base-patch16-ensemble")
    p.add_argument("--device", default="cuda")

    p.add_argument("--llm-url", default=None, metavar="URL",
                   help="Base URL of an OpenAI-compatible LLM server (template proposal). "
                        "Omit to pull --llm-model from HuggingFace and run it locally instead.")
    p.add_argument("--llm-model", default="Qwen/Qwen3-4B", metavar="NAME",
                   help="Model name understood by the LLM server (served backend) or a "
                        "HuggingFace Hub model id to load locally (local backend).")
    p.add_argument("--llm-device", default="cuda",
                   help="Device for the local HF LLM backend (ignored when --llm-url is set)")
    p.add_argument("--llm-temperature", type=float, default=0.8)
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

    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--candidates", type=int, default=5, dest="candidates_per_iter")
    p.add_argument("--max-iterations", type=int, default=100)

    p.add_argument("--train-samples-per-class", type=int, default=20)
    p.add_argument("--train-split", type=float, default=0.8)
    p.add_argument("--agml-data-root", default=None, metavar="PATH",
                   help="Override HuggingFace dataset cache directory (default: ~/.cache/huggingface/)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="experiments/results/meta_template", metavar="PATH")

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

    vlm = VLMClient(
        base_url=args.llm_url, model=args.llm_model,
        temperature=args.llm_temperature, max_tokens=args.llm_max_tokens,
        device=args.llm_device,
    )
    logger.info(f"LLM: {args.llm_model} @ {args.llm_url or f'local ({args.llm_device})'}")

    meta_config = MetaTemplateConfig(
        patience=args.patience,
        candidates_per_iter=args.candidates_per_iter,
        max_iterations=args.max_iterations,
    )
    logger.info(
        f"MetaTemplateConfig: patience={meta_config.patience}, "
        f"candidates_per_iter={meta_config.candidates_per_iter}, "
        f"max_iterations={meta_config.max_iterations}"
    )

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
    summary_path = output_dir / "cross_run_summary.json"
    summary = CrossRunSummary.load(summary_path) if summary_path.exists() else CrossRunSummary()

    all_results: list[MetaTemplateResult] = []

    for model_key in model_keys:
        logger.info(f"=== Model: {model_key} ===")
        try:
            model = _build_model(model_key, args)
        except Exception as exc:
            logger.error(f"Failed to load {model_key}: {exc}", exc_info=True)
            continue

        result = run_meta_template_search(
            model=model,
            vlm=vlm,
            groups=groups,
            config=meta_config,
            train_datasets=sorted(train_splits.keys()),
            test_datasets=[ds_cfg["name"] for ds_cfg, _ in test_splits],
            cross_run_summary=summary.text,
        )

        for ds_cfg, test_split in test_splits:
            result.zero_shot.extend(
                evaluate_zero_shot(
                    model=model,
                    test_split=test_split,
                    crop=ds_cfg["crop"],
                    best_template=result.best_template,
                    llm=vlm,
                    n_eval_images=args.train_samples_per_class,
                    seed=args.seed,
                    vlm_insight=vlm_insight,
                )
            )

        all_results.append(result)
        save_meta_template_result(result, output_dir / model_key)

        summary.runs.append({
            "model": model_key,
            "best_template": result.best_template,
            "baseline_aggregate_map": round(result.baseline_aggregate_map, 4),
            "best_aggregate_map": round(result.best_aggregate_map, 4),
        })

        try:
            import torch
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    summary.save(summary_path)
    _print_summary(all_results)
    logger.info(f"Done. Results written to {output_dir}/")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_summary(results: list[MetaTemplateResult]) -> None:
    if not results:
        logger.info("No meta-template results to summarise.")
        return

    sep = "=" * 90
    logger.info(sep)
    logger.info("  META PROMPT TEMPLATE — ZERO-SHOT TRANSFER SUMMARY")
    logger.info(sep)

    for r in results:
        gain = r.best_aggregate_map - r.baseline_aggregate_map
        logger.info(
            f"  [{r.model_name}] train aggregate mAP: baseline={r.baseline_aggregate_map:.4f} "
            f"best={r.best_aggregate_map:.4f} gain={gain:+.4f}"
        )
        logger.info(f"    best template: {r.best_template!r}")

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
