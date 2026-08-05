#!/usr/bin/env python3
"""Full pipeline runner — the paper's central cross-dataset comparison.

For every requested detector model, runs the three template-based,
cross-dataset conditions the central research question in PAPER.md depends
on, all against the SAME pooled training mix and held-out split from
``--datasets-file``:

  1. **Discovery** — axis-based OFAT over the pooled training mix
     (:func:`agvfm.optimizer.prompt_template.run_template_ofat`).
  2. **Constrained transfer** — the discovered template evaluated zero-shot
     on every held-out dataset, both metadata-filled and (optionally)
     VLM-insight-filled (:func:`agvfm.optimizer.prompt_template.evaluate_zero_shot`).
  3. **Metaprompting, summary-informed zero-shot** — an LLM proposes a
     free-text template against the same pooled mix
     (:func:`agvfm.optimizer.meta_prompt_template.run_meta_template_search`),
     evaluated zero-shot the same way, with a
     :class:`~agvfm.optimizer.meta_prompt.CrossRunSummary` carried across
     models so later models benefit from earlier ones' results without
     touching the held-out data.

If ``gemma4`` is included in ``--model``, also runs the VLM-as-unified-
pipeline condition (PAPER.md 3.6) directly against the held-out sets, in
whichever ``--gemma4-mode`` is selected (raw/constrained/unconstrained) — see
``_evaluate_gemma4_zero_shot`` below for why this needs its own evaluation
path rather than reusing conditions 1-3's ``VFMBase``/``AgVFMAdapter`` plumbing.

Deliberately excludes (see AGENT.md's "explicitly out of scope", and
explicit user direction to focus on the template/cross-dataset benchmarking
track for now): metaprompt cold-start (``meta_prompt.py``/``meta_run.py``),
single-dataset axis search (``run.py``/``optimizer/loop.py``), and the
PEZ/LoRA gradient baselines — those target one dataset at a time rather than
benchmarking across many, which is this script's whole point.

Every condition is logged via :class:`agvfm.instrumentation.tracking.RunTracker`
to ``<output-dir>/instrumentation.jsonl``; at the end,
:mod:`agvfm.reporting.aggregate` turns that into the paper's cost/performance
comparison table (written to ``<output-dir>/comparison_table.csv`` and
printed to the console).

Usage
-----
    python run_full_pipeline.py --datasets-file datasets_pool.example.yaml \\
        --llm-url http://localhost:8000/v1 --llm-model meta-llama/Llama-3.2-1B-Instruct \\
        --model yolo_world owlv2

    # No LLM server — local HF backend for axis/template generation
    python run_full_pipeline.py --datasets-file datasets_pool.example.yaml \\
        --llm-model Qwen/Qwen3-4B --model yolo_world

    # Include Gemma 4 (constrained mode) alongside two open-vocab detectors
    python run_full_pipeline.py --datasets-file datasets_pool.example.yaml \\
        --llm-url http://localhost:8000/v1 --llm-model meta-llama/Llama-3.2-1B-Instruct \\
        --model yolo_world owlv2 gemma4 --gemma4-mode constrained \\
        --gemma4-url http://localhost:8002/v1 --gemma4-model google/gemma-4-E4B-it

``--datasets-file`` schema: identical to ``run_template.py`` — see its module
docstring (includes the optional ``rarity: common|rare`` flag on held-out
entries, logged as ``held_out_rarity`` on every CallRecord this script emits).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import yaml

sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace") if hasattr(sys.stdout, "reconfigure") else None
sys.stderr.reconfigure(line_buffering=True, encoding="utf-8", errors="replace") if hasattr(sys.stderr, "reconfigure") else None

project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from agvfm.instrumentation.tracking import CallRecord, RunTracker
from agvfm.llm.client import LLMClient
from agvfm.llm.meta_client import VLMClient
from agvfm.llm.vlm_insight_client import VLMInsightClient
from agvfm.optimizer.meta_prompt import CrossRunSummary
from agvfm.optimizer.meta_prompt_template import (
    MetaTemplateConfig,
    evaluate_zero_shot as meta_evaluate_zero_shot,
    run_meta_template_search,
    save_meta_template_result,
)
from agvfm.optimizer.prompt_template import (
    build_train_groups,
    evaluate_zero_shot as axis_evaluate_zero_shot,
    generate_axis_values_per_crop,
    run_template_ofat,
    save_template_result,
)
from agvfm.optimizer.types import AgVFMAdapter, Detection, _match_detections, sample_proxy_images
from agvfm.data.agml_loader import sample_proxy_images_for_class
from agvfm.reporting.aggregate import aggregate_records, format_table, load_records, write_csv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

_ALL_DETECTORS = ["yolo_world", "grounding_dino", "owlv2"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Full cross-dataset comparison pipeline (discovery + constrained transfer + "
                    "summary-informed metaprompting, optionally + Gemma 4 unified pipeline)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--datasets-file", required=True, metavar="PATH")
    p.add_argument("--model", nargs="+",
                   choices=_ALL_DETECTORS + ["all", "gemma4"], default=["yolo_world"])
    p.add_argument("--yolo-weights", default=None, metavar="PATH")
    p.add_argument("--gdino-checkpoint", default="IDEA-Research/grounding-dino-base")
    p.add_argument("--owlv2-checkpoint", default="google/owlv2-base-patch16-ensemble")
    p.add_argument("--device", default="cuda")

    # Axis-value generation (discovery)
    p.add_argument("--llm-url", default=None, metavar="URL",
                   help="Base URL of an OpenAI-compatible LLM server. Omit to pull "
                        "--llm-model from HuggingFace and run it locally instead.")
    p.add_argument("--llm-model", default="Qwen/Qwen3-4B", metavar="NAME")
    p.add_argument("--llm-device", default="cuda")
    p.add_argument("--llm-temperature", type=float, default=0.7)
    p.add_argument("--llm-max-tokens", type=int, default=512)

    # Metaprompt template-proposal LLM (defaults to the axis-generation LLM's endpoint/model)
    p.add_argument("--meta-llm-url", default=None, metavar="URL")
    p.add_argument("--meta-llm-model", default=None, metavar="NAME")
    p.add_argument("--meta-llm-device", default=None)
    p.add_argument("--meta-llm-temperature", type=float, default=0.8)
    p.add_argument("--meta-llm-max-tokens", type=int, default=512)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--candidates", type=int, default=5, dest="candidates_per_iter")
    p.add_argument("--max-iterations", type=int, default=100)

    # Optional VLM-insight zero-shot fill (shared by transfer + metaprompt-summary)
    p.add_argument("--vlm-insight-url", default=None, metavar="URL")
    p.add_argument("--vlm-insight-model", default=None, metavar="NAME")
    p.add_argument("--vlm-insight-device", default="cuda")
    p.add_argument("--vlm-insight-temperature", type=float, default=0.2)
    p.add_argument("--vlm-insight-max-tokens", type=int, default=300)

    # Gemma 4 unified pipeline (only used if "gemma4" in --model)
    p.add_argument("--gemma4-mode", choices=["raw", "constrained", "unconstrained"], default="constrained")
    p.add_argument("--gemma4-url", default=None, metavar="URL")
    p.add_argument("--gemma4-model", default="google/gemma-4-E4B-it", metavar="NAME")
    p.add_argument("--gemma4-device", default="cuda")

    p.add_argument("--train-samples-per-class", type=int, default=20)
    p.add_argument("--train-split", type=float, default=0.8)
    p.add_argument("--agml-data-root", default=None, metavar="PATH")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="experiments/results/full_pipeline", metavar="PATH")

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

def _build_detector_adapter(model_key: str, args: argparse.Namespace) -> AgVFMAdapter:
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
    raise ValueError(f"Unknown detector model: {model_key!r}")


def _build_gemma4(args: argparse.Namespace):
    from agvfm.models import Gemma4VLMPipeline
    return Gemma4VLMPipeline(
        base_url=args.gemma4_url,
        model=args.gemma4_model,
        mode=args.gemma4_mode,
        device=args.gemma4_device,
    )


# ---------------------------------------------------------------------------
# Per-model condition runners (detectors 1-3: discovery, transfer, metaprompt-summary)
# ---------------------------------------------------------------------------

def _run_discovery_and_transfer(
    model_key: str,
    model: AgVFMAdapter,
    groups,
    axis_values: dict,
    train_dataset_names: list[str],
    test_dataset_names: list[str],
    test_splits: list[tuple[dict, object]],
    vlm_insight,
    llm,
    args: argparse.Namespace,
    tracker: RunTracker,
    output_dir: Path,
) -> dict:
    """Conditions 1 (discovery) + 2 (constrained transfer). Returns the discovered ``best_axes``."""
    logger.info(f"[{model_key}] === Discovery (axis-based OFAT) ===")
    template_result = run_template_ofat(
        model=model, groups=groups, axis_values=axis_values,
        train_datasets=train_dataset_names, test_datasets=test_dataset_names,
    )
    tracker.log(CallRecord(
        dataset_id="training_pool", model_id=model_key, method="discovery",
        prompt_space="n/a", split_id="train",
        wall_clock_seconds=template_result.wall_clock_seconds,
        map_score=template_result.best_aggregate_map,
        extra_metrics={"baseline_aggregate_map": template_result.baseline_aggregate_map, "best_axes": template_result.best_axes},
    ))

    logger.info(f"[{model_key}] === Constrained transfer (zero-shot) ===")
    for ds_cfg, test_split in test_splits:
        rarity = ds_cfg.get("rarity", "common")
        results = axis_evaluate_zero_shot(
            model=model, test_split=test_split, crop=ds_cfg["crop"],
            best_axes=template_result.best_axes,
            n_eval_images=args.train_samples_per_class, seed=args.seed,
            vlm_insight=vlm_insight, llm=llm,
        )
        template_result.zero_shot.extend(results)
        for z in results:
            tracker.log(CallRecord(
                dataset_id=z.dataset_name, model_id=model_key, method="transfer",
                prompt_space="constrained", held_out_rarity=rarity, split_id="test",
                n_api_calls=(1 if z.class_translate_tokens_in or z.class_translate_tokens_out else 0),
                tokens_in=z.class_translate_tokens_in, tokens_out=z.class_translate_tokens_out,
                wall_clock_seconds=z.wall_clock_seconds, map_score=z.template_metadata_map,
                extra_metrics={
                    "variant": "metadata", "class": z.class_name, "naive_map": z.naive_map,
                    "prompt": z.template_prompt_metadata, "translated_class_name": z.translated_class_name,
                },
            ))
            if z.template_vlm_map is not None:
                tracker.log(CallRecord(
                    dataset_id=z.dataset_name, model_id=model_key, method="transfer",
                    prompt_space="constrained", held_out_rarity=rarity, split_id="test",
                    n_api_calls=1, tokens_in=z.vlm_insight_tokens_in, tokens_out=z.vlm_insight_tokens_out,
                    n_unlabeled_images_used=z.n_unlabeled_images_used, map_score=z.template_vlm_map,
                    extra_metrics={
                        "variant": "vlm_insight", "class": z.class_name, "naive_map": z.naive_map,
                        "prompt": z.template_prompt_vlm,
                    },
                ))

    save_template_result(template_result, output_dir / model_key)
    return template_result.best_axes


def _run_metaprompt_summary(
    model_key: str,
    model: AgVFMAdapter,
    vlm: VLMClient,
    groups,
    meta_config: MetaTemplateConfig,
    train_dataset_names: list[str],
    test_dataset_names: list[str],
    test_splits: list[tuple[dict, object]],
    vlm_insight,
    llm,
    cross_run_summary: CrossRunSummary,
    args: argparse.Namespace,
    tracker: RunTracker,
    output_dir: Path,
) -> None:
    """Condition 3b: metaprompting, summary-informed zero-shot."""
    logger.info(f"[{model_key}] === Metaprompting (summary-informed, zero-shot) ===")
    result = run_meta_template_search(
        model=model, vlm=vlm, groups=groups, config=meta_config,
        train_datasets=train_dataset_names, test_datasets=test_dataset_names,
        cross_run_summary=cross_run_summary.text,
    )
    tracker.log(CallRecord(
        dataset_id="training_pool", model_id=model_key, method="metaprompt-summary",
        prompt_space="unconstrained", split_id="train",
        wall_clock_seconds=result.wall_clock_seconds, n_api_calls=result.total_api_calls,
        tokens_in=result.total_tokens_in, tokens_out=result.total_tokens_out,
        map_score=result.best_aggregate_map,
        extra_metrics={
            "baseline_aggregate_map": result.baseline_aggregate_map,
            "best_template": result.best_template,
            "total_iterations": result.total_iterations,
            "patience_exhausted": result.patience_exhausted,
        },
    ))

    for ds_cfg, test_split in test_splits:
        rarity = ds_cfg.get("rarity", "common")
        results = meta_evaluate_zero_shot(
            model=model, test_split=test_split, crop=ds_cfg["crop"],
            best_template=result.best_template,
            n_eval_images=args.train_samples_per_class, seed=args.seed,
            vlm_insight=vlm_insight, llm=llm,
        )
        result.zero_shot.extend(results)
        for z in results:
            tracker.log(CallRecord(
                dataset_id=z.dataset_name, model_id=model_key, method="metaprompt-summary",
                prompt_space="unconstrained", held_out_rarity=rarity, split_id="test",
                n_api_calls=(1 if z.class_translate_tokens_in or z.class_translate_tokens_out else 0),
                tokens_in=z.class_translate_tokens_in, tokens_out=z.class_translate_tokens_out,
                wall_clock_seconds=z.wall_clock_seconds, map_score=z.template_metadata_map,
                extra_metrics={
                    "variant": "metadata", "class": z.class_name, "naive_map": z.naive_map,
                    "prompt": z.template_prompt_metadata, "translated_class_name": z.translated_class_name,
                },
            ))
            if z.template_vlm_map is not None:
                tracker.log(CallRecord(
                    dataset_id=z.dataset_name, model_id=model_key, method="metaprompt-summary",
                    prompt_space="unconstrained", held_out_rarity=rarity, split_id="test",
                    n_api_calls=1, tokens_in=z.vlm_insight_tokens_in, tokens_out=z.vlm_insight_tokens_out,
                    n_unlabeled_images_used=z.n_unlabeled_images_used, map_score=z.template_vlm_map,
                    extra_metrics={
                        "variant": "vlm_insight", "class": z.class_name, "naive_map": z.naive_map,
                        "prompt": z.template_prompt_vlm,
                    },
                ))

    save_meta_template_result(result, output_dir / model_key)
    cross_run_summary.runs.append({
        "model": model_key,
        "best_template": result.best_template,
        "baseline_aggregate_map": round(result.baseline_aggregate_map, 4),
        "best_aggregate_map": round(result.best_aggregate_map, 4),
    })


# ---------------------------------------------------------------------------
# Gemma 4 condition (5): needs its own evaluation loop
# ---------------------------------------------------------------------------
#
# Gemma4VLMPipeline.predict() accepts crop=/axis_template= kwargs (see its
# module docstring) beyond the plain prompt string every other BaseModel
# subclass takes — richer context it needs to generate its own prompt rather
# than just consume one. VFMBase.predict()/compute_map() (used by
# run_template_ofat/evaluate_zero_shot/run_meta_template_search for every
# other model) only threads a single prompt string through, by design, so
# Gemma 4 can't reuse that path without changing VFMBase's signature and
# every one of its other callers (grad_prompt.py, meta_prompt.py, ...) —
# out of scope for this pipeline. This function computes the same
# naive-vs-filled F1 proxy directly against Gemma4VLMPipeline instead.

def _evaluate_gemma4_zero_shot(
    gemma4,
    test_split,
    ds_cfg: dict,
    axis_template: dict,
    n_eval_images: int,
    seed: int,
    tracker: RunTracker,
) -> None:
    crop = ds_cfg["crop"]
    rarity = ds_cfg.get("rarity", "common")
    mode = gemma4.mode

    def _score(samples, class_name: str, **predict_kwargs) -> float:
        tp = fp = fn = 0
        for sample in samples:
            fd, name = tempfile.mkstemp(suffix=".jpg")
            os.close(fd)
            tmp_path = Path(name)
            try:
                sample.image.save(tmp_path, format="JPEG", quality=95)
                w, h = sample.image.size
                boxes, confs = gemma4.predict(tmp_path, class_name, conf_threshold=0.1, **predict_kwargs)
                dets = [
                    Detection(bbox=[b[0] / w, b[1] / h, b[2] / w, b[3] / h], score=float(c), class_id=0, class_name=class_name)
                    for b, c in zip(boxes, confs)
                ]
                dtp, dfp, dfn = _match_detections(dets, sample.annotations, 0.5)
                tp += dtp
                fp += dfp
                fn += dfn
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    for class_name in test_split.classes:
        if len(test_split.classes) == 1:
            samples = sample_proxy_images(test_split, n_eval_images, seed=seed)
        else:
            samples = sample_proxy_images_for_class(test_split, class_name, n_eval_images, seed=seed)
        if not samples:
            logger.warning(f"No samples for held-out {test_split.name}/{class_name!r} — skipping (gemma4)")
            continue

        # Naive baseline: force_raw=True skips self-generation regardless of
        # this instance's configured mode, so the same weights/backend serve
        # as their own detection-only baseline (see Gemma4VLMPipeline.predict).
        # Measured and excluded from the pipeline's own cost/timing below —
        # it's a reference point for the paper's comparison, not part of what
        # a real deployment of this pipeline would pay.
        naive_map = _score(samples, class_name, force_raw=True)

        usage_before = gemma4.usage_snapshot()
        t0 = time.time()
        filled_kwargs = {"crop": crop}
        if mode == "constrained":
            filled_kwargs["axis_template"] = axis_template
        filled_map = _score(samples, class_name, **filled_kwargs)
        elapsed = time.time() - t0
        usage_after = gemma4.usage_snapshot()

        prompt_space = "n/a" if mode == "raw" else ("constrained" if mode == "constrained" else "unconstrained")
        pipeline_role = "detection" if mode == "raw" else "combined"
        delta = {k: usage_after[k] - usage_before[k] for k in usage_after}
        gen_usage = gemma4.usage_snapshot("prompt_generation")

        tracker.log(CallRecord(
            dataset_id=test_split.name, model_id=f"gemma4-{mode}", method=f"vlm_pipeline-{mode}",
            prompt_space=prompt_space, pipeline_role=pipeline_role,
            held_out_rarity=rarity, split_id="test",
            wall_clock_seconds=elapsed, n_api_calls=delta["n_api_calls"],
            tokens_in=delta["tokens_in"], tokens_out=delta["tokens_out"],
            map_score=filled_map,
            extra_metrics={"naive_map": naive_map, "class": class_name, "prompt_generation_calls_total": gen_usage["n_api_calls"]},
        ))
        logger.info(
            f"  [gemma4-{mode}][{test_split.name}][{class_name}] naive={naive_map:.4f} filled={filled_map:.4f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    from agvfm.data.agml_loader import load_dataset

    train_ds_cfgs_list, test_ds_cfgs = _load_datasets_file(args.datasets_file)
    train_ds_cfgs = {d["name"]: d for d in train_ds_cfgs_list}

    output_dir = Path(args.output_dir)
    tracker = RunTracker(output_dir / "instrumentation.jsonl")

    # --- Load train datasets, pool ~N images per class -----------------------
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
    train_dataset_names = sorted(train_splits.keys())
    logger.info(
        f"Pooled training mix: {len(groups)} (dataset, class) groups from "
        f"{len({g.dataset_name for g in groups})} datasets, {sum(len(g.samples) for g in groups)} total images"
    )

    # --- Load held-out test datasets -----------------------------------------
    test_splits: list[tuple[dict, object]] = []
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
    test_dataset_names = [ds_cfg["name"] for ds_cfg, _ in test_splits]

    # --- Shared clients -------------------------------------------------------
    llm = LLMClient(
        base_url=args.llm_url, model=args.llm_model,
        temperature=args.llm_temperature, max_tokens=args.llm_max_tokens, device=args.llm_device,
    )
    crops = sorted({ds_cfg["crop"] for ds_cfg in train_ds_cfgs.values()})
    axes_file = output_dir / "factor_axes_by_crop.json"
    axis_values = generate_axis_values_per_crop(llm, crops, axes_file)
    logger.info(f"Pooled per-crop axis values for template search ({crops}): {axis_values}")

    vlm = VLMClient(
        base_url=args.meta_llm_url or args.llm_url,
        model=args.meta_llm_model or args.llm_model,
        temperature=args.meta_llm_temperature, max_tokens=args.meta_llm_max_tokens,
        device=args.meta_llm_device or args.llm_device,
    )
    meta_config = MetaTemplateConfig(
        patience=args.patience, candidates_per_iter=args.candidates_per_iter, max_iterations=args.max_iterations,
    )

    vlm_insight = None
    if args.vlm_insight_model:
        vlm_insight = VLMInsightClient(
            base_url=args.vlm_insight_url, model=args.vlm_insight_model,
            temperature=args.vlm_insight_temperature, max_tokens=args.vlm_insight_max_tokens,
            device=args.vlm_insight_device,
        )

    summary_path = output_dir / "cross_run_summary.json"
    cross_run_summary = CrossRunSummary.load(summary_path) if summary_path.exists() else CrossRunSummary()

    model_keys = _ALL_DETECTORS if "all" in args.model else args.model
    detector_keys = [k for k in model_keys if k != "gemma4"]
    run_gemma4 = "gemma4" in model_keys

    shared_best_axes: Optional[dict] = None

    for model_key in detector_keys:
        logger.info(f"\n{'='*70}\nModel: {model_key}\n{'='*70}")
        try:
            model = _build_detector_adapter(model_key, args)
        except Exception as exc:
            logger.error(f"Failed to load {model_key}: {exc}", exc_info=True)
            continue

        try:
            best_axes = _run_discovery_and_transfer(
                model_key, model, groups, axis_values, train_dataset_names, test_dataset_names,
                test_splits, vlm_insight, llm, args, tracker, output_dir,
            )
            if shared_best_axes is None:
                shared_best_axes = best_axes
        except Exception as exc:
            logger.error(f"Discovery/transfer failed for {model_key}: {exc}", exc_info=True)

        try:
            _run_metaprompt_summary(
                model_key, model, vlm, groups, meta_config, train_dataset_names, test_dataset_names,
                test_splits, vlm_insight, llm, cross_run_summary, args, tracker, output_dir,
            )
        except Exception as exc:
            logger.error(f"Metaprompt-summary failed for {model_key}: {exc}", exc_info=True)

        try:
            import torch
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    cross_run_summary.save(summary_path)

    if run_gemma4:
        logger.info(f"\n{'='*70}\nModel: gemma4 (mode={args.gemma4_mode})\n{'='*70}")
        if args.gemma4_mode == "constrained" and shared_best_axes is None:
            logger.warning(
                "gemma4 mode='constrained' but no other model ran discovery this session — "
                "falling back to a bare structural baseline (grammar='a', rest empty) for axis_template."
            )
            shared_best_axes = {"grammar": "a", "color": "", "size": "", "anatomy": "", "phenology": ""}
        try:
            gemma4 = _build_gemma4(args)
            for ds_cfg, test_split in test_splits:
                _evaluate_gemma4_zero_shot(
                    gemma4, test_split, ds_cfg, shared_best_axes or {},
                    args.train_samples_per_class, args.seed, tracker,
                )
        except Exception as exc:
            logger.error(f"Gemma4 pipeline failed: {exc}", exc_info=True)

    # --- Aggregate everything into the paper's comparison table --------------
    records = load_records(output_dir)
    rows = aggregate_records(records)
    write_csv(rows, output_dir / "comparison_table.csv")
    print("\n" + format_table(rows))
    logger.info(f"\nDone. Results + comparison table written to {output_dir}/")


if __name__ == "__main__":
    main()
