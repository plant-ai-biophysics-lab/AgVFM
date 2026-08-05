"""Cross-dataset free-text prompt-*template* search via LLM, for zero-shot transfer.

Where :mod:`agvfm.optimizer.meta_prompt` asks an LLM to propose full prompt
strings per (dataset, class), this module asks the LLM to propose reusable
**templates** — strings containing a literal ``"{class}"`` placeholder — and
scores each template by substituting the real class name of every (dataset,
class) group in the pooled cross-dataset training mix, then aggregating
tp/fp/fn across all of them (see
:func:`agvfm.optimizer.prompt_template.aggregate_map`).

The winning template is then evaluated zero-shot on held-out test datasets,
filling ``{class}`` either with the bare class name (metadata) or with a
short VLM-derived descriptive phrase for the specific held-out crop (see
:class:`~agvfm.llm.vlm_insight_client.VLMInsightClient`).

This is the unconstrained/open-ended counterpart to
:mod:`agvfm.optimizer.prompt_template` — same pooled-training-mix +
zero-shot-transfer structure, but the search itself is over free text rather
than axis slots (PAPER.md section 3.3's "summary-informed zero-shot" mode,
via :class:`~agvfm.optimizer.meta_prompt.CrossRunSummary`).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from agvfm.data.agml_loader import sample_proxy_images_for_class
from agvfm.evaluation.metrics import compute_metrics_at_iou
from agvfm.llm.meta_client import VLMClient
from agvfm.llm.vlm_insight_client import VLMInsightClient
from agvfm.optimizer.prompt_template import TrainGroup
from agvfm.optimizer.types import DatasetSplit, VFMBase, _xywh_to_xyxy, sample_proxy_images

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MetaTemplateEvalResult:
    iteration: int          # 0 = baseline
    template: str
    aggregate_map: float
    is_best: bool
    # Populated on iteration>0 rows only — the baseline (iteration 0) makes
    # no LLM call. Per PAPER.md 5.2: per-iteration granularity, not just a
    # final summary, since convergence speed (iterations/calls/tokens) is
    # exactly what's being compared against condition 2 (transfer).
    wall_clock_seconds: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    n_api_calls: int = 0


@dataclass
class MetaZeroShotEvalResult:
    dataset_name: str
    class_name: str
    crop: str
    naive_map: float
    template_metadata_map: float
    template_prompt_metadata: str
    translated_class_name: str = ""        # class_name after LLMClient.translate_class_name, if used
    template_vlm_map: Optional[float] = None
    template_prompt_vlm: Optional[str] = None
    wall_clock_seconds: float = 0.0
    class_translate_tokens_in: int = 0
    class_translate_tokens_out: int = 0
    vlm_insight_tokens_in: int = 0
    vlm_insight_tokens_out: int = 0
    n_unlabeled_images_used: int = 0


@dataclass
class MetaTemplateConfig:
    patience: int = 10
    candidates_per_iter: int = 5
    max_iterations: int = 100


@dataclass
class MetaTemplateResult:
    model_name: str
    train_datasets: list[str]
    test_datasets: list[str]
    baseline_template: str
    baseline_aggregate_map: float
    best_template: str
    best_aggregate_map: float
    total_iterations: int
    patience_exhausted: bool
    total_evaluations: int
    history: list[MetaTemplateEvalResult] = field(default_factory=list)
    zero_shot: list[MetaZeroShotEvalResult] = field(default_factory=list)
    wall_clock_seconds: float = 0.0  # total search time (all iterations)
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_api_calls: int = 0


# ---------------------------------------------------------------------------
# Template search over the pooled training mix
# ---------------------------------------------------------------------------

def run_meta_template_search(
    model: VFMBase,
    vlm: VLMClient,
    groups: list[TrainGroup],
    config: MetaTemplateConfig,
    train_datasets: list[str],
    test_datasets: list[str],
    cross_run_summary: str = "",
) -> MetaTemplateResult:
    """Search for one free-text template that generalises across ``groups``."""
    if not groups:
        raise ValueError("No training groups to search a template against.")

    crops = sorted({g.crop for g in groups})

    def _score(template: str) -> float:
        return _aggregate_template_map(model, groups, template)

    history: list[MetaTemplateEvalResult] = []

    baseline_template = "{class}"
    baseline_map = _score(baseline_template)
    logger.info(f"[meta-template] baseline aggregate mAP={baseline_map:.4f}  template={baseline_template!r}")

    best_map = baseline_map
    best_template = baseline_template
    history.append(MetaTemplateEvalResult(iteration=0, template=baseline_template, aggregate_map=baseline_map, is_best=True))

    no_improve_count = 0
    patience_exhausted = False
    iteration = 0
    _search_start = time.time()

    for iteration in range(1, config.max_iterations + 1):
        if no_improve_count >= config.patience:
            patience_exhausted = True
            logger.info(
                f"[meta-template] Stopping: {no_improve_count} iterations without improvement "
                f"(patience={config.patience})"
            )
            break

        trimmed_history = _trim_history(history, iteration)
        _iter_start = time.time()
        _usage_before = vlm.usage_snapshot()
        candidates = vlm.suggest_template_prompts(
            crops=crops,
            history=[{"prompt": r.template, "map_score": r.aggregate_map} for r in trimmed_history],
            n_candidates=config.candidates_per_iter,
            cross_run_summary=cross_run_summary,
            all_seen_templates={r.template for r in history},
        )
        _usage_after = vlm.usage_snapshot()
        # Per-iteration LLM-call cost (PAPER.md 5.2: needed at iteration
        # granularity, not just a final summary, since convergence speed is
        # exactly what's compared against condition 2/transfer). One
        # suggest_template_prompts() call produces every candidate for this
        # iteration, so its cost is attributed to the first history row
        # appended below rather than split across candidates.
        _iter_wall = time.time() - _iter_start
        _iter_tokens_in = _usage_after["tokens_in"] - _usage_before["tokens_in"]
        _iter_tokens_out = _usage_after["tokens_out"] - _usage_before["tokens_out"]
        _iter_calls = _usage_after["n_api_calls"] - _usage_before["n_api_calls"]

        improved_this_iter = False
        first_row_this_iter = True
        for template in candidates:
            if not template or "{class}" not in template:
                continue
            score = _score(template)
            is_new_best = score > best_map
            if is_new_best:
                best_map = score
                best_template = template
                improved_this_iter = True
            history.append(MetaTemplateEvalResult(
                iteration=iteration, template=template, aggregate_map=score, is_best=is_new_best,
                wall_clock_seconds=_iter_wall if first_row_this_iter else 0.0,
                tokens_in=_iter_tokens_in if first_row_this_iter else 0,
                tokens_out=_iter_tokens_out if first_row_this_iter else 0,
                n_api_calls=_iter_calls if first_row_this_iter else 0,
            ))
            first_row_this_iter = False
            logger.debug(
                f"[meta-template] iter={iteration}  aggregate mAP={score:.4f}"
                f"{'  ← new best' if is_new_best else ''}  {template!r}"
            )

        if improved_this_iter:
            no_improve_count = 0
            logger.info(f"[meta-template] iter={iteration}  NEW BEST aggregate mAP={best_map:.4f}  {best_template!r}")
        else:
            no_improve_count += 1
            logger.info(
                f"[meta-template] iter={iteration}  no improvement "
                f"(patience {no_improve_count}/{config.patience})"
            )

    return MetaTemplateResult(
        model_name=model.name,
        train_datasets=train_datasets,
        test_datasets=test_datasets,
        baseline_template=baseline_template,
        wall_clock_seconds=time.time() - _search_start,
        total_tokens_in=sum(r.tokens_in for r in history),
        total_tokens_out=sum(r.tokens_out for r in history),
        total_api_calls=sum(r.n_api_calls for r in history),
        baseline_aggregate_map=baseline_map,
        best_template=best_template,
        best_aggregate_map=best_map,
        total_iterations=iteration,
        patience_exhausted=patience_exhausted,
        total_evaluations=len(history),
        history=history,
    )


def _aggregate_template_map(
    model: VFMBase,
    groups: list[TrainGroup],
    template: str,
    iou_threshold: float = 0.5,
) -> float:
    """Score one free-text template across every group in ``groups``.

    Analogous to :func:`agvfm.optimizer.prompt_template.aggregate_map`, but
    fills the template's ``"{class}"`` placeholder directly instead of going
    through :class:`~agvfm.optimizer.axes.PromptAxes`, since free-text
    templates don't decompose into axes. Scored as mAP@*iou_threshold*
    (101-point interpolated AP over ground-truth/detections pooled across all
    groups — same metric ``load_and_run.py``'s ``Evaluator`` reports), not
    the F1-at-implicit-threshold proxy this used to compute.
    """
    list_gt_xyxy: list[np.ndarray] = []
    list_pred_xyxy: list[np.ndarray] = []
    list_pred_conf: list[np.ndarray] = []

    for group in groups:
        prompt = template.format(**{"class": group.class_name})
        for sample in group.samples:
            preds = model.predict(sample.image, [prompt])
            # Only this group's class counts as ground truth — see the same
            # fix/comment in agvfm.optimizer.prompt_template.aggregate_map.
            class_annotations = [a for a in sample.annotations if a["category_name"] == group.class_name]
            gt_xyxy = (
                np.array([_xywh_to_xyxy(a["bbox"]) for a in class_annotations], dtype=np.float64)
                if class_annotations else np.zeros((0, 4), dtype=np.float64)
            )
            pred_xyxy = (
                np.array([d.bbox for d in preds], dtype=np.float64)
                if preds else np.zeros((0, 4), dtype=np.float64)
            )
            pred_conf = (
                np.array([d.score for d in preds], dtype=np.float64)
                if preds else np.zeros(0, dtype=np.float64)
            )
            list_gt_xyxy.append(gt_xyxy)
            list_pred_xyxy.append(pred_xyxy)
            list_pred_conf.append(pred_conf)

    metrics = compute_metrics_at_iou(list_gt_xyxy, list_pred_xyxy, list_pred_conf, iou_threshold=iou_threshold)
    return metrics["map"]


def _trim_history(history: list[MetaTemplateEvalResult], current_iteration: int) -> list[MetaTemplateEvalResult]:
    if not history:
        return []
    best = max(history, key=lambda r: r.aggregate_map)
    prev_iter = current_iteration - 1
    recent = [r for r in history if r.iteration == prev_iter] or history[-1:]

    seen: set[str] = set()
    trimmed: list[MetaTemplateEvalResult] = []
    trimmed.append(best)
    seen.add(best.template)
    for r in sorted(recent, key=lambda x: x.aggregate_map, reverse=True):
        if r.template not in seen:
            trimmed.append(r)
            seen.add(r.template)
    return trimmed


# ---------------------------------------------------------------------------
# Zero-shot evaluation on held-out test datasets
# ---------------------------------------------------------------------------

def evaluate_zero_shot(
    model: VFMBase,
    test_split: DatasetSplit,
    crop: str,
    best_template: str,
    n_eval_images: Optional[int] = 20,
    seed: int = 42,
    vlm_insight: Optional[VLMInsightClient] = None,
    llm=None,
) -> list[MetaZeroShotEvalResult]:
    """Evaluate the discovered free-text template zero-shot on every class of ``test_split``.

    ``best_template`` (discovered against the *training* pool) is used
    completely unchanged — that's the actual transfer test. The only thing
    that varies per held-out crop is what fills ``{class}``: when ``llm`` is
    given, the raw dataset ``class_name`` is translated to a natural phrase
    via ``llm.translate_class_name(crop, class_name)`` (a single realistic
    zero-shot input, one output — not the iterative candidate search used
    during training) before formatting the template; without ``llm``, the
    raw ``class_name`` is used as before. The naive baseline always uses the
    raw ``class_name`` regardless.
    """
    results: list[MetaZeroShotEvalResult] = []

    for class_name in test_split.classes:
        _class_start = time.time()
        if len(test_split.classes) == 1:
            samples = sample_proxy_images(test_split, n_eval_images, seed=seed)
        else:
            samples = sample_proxy_images_for_class(test_split, class_name, n_eval_images, seed=seed)
        if not samples:
            logger.warning(f"No samples for held-out {test_split.name}/{class_name!r} — skipping")
            continue

        naive_map = model.compute_map(samples, [class_name])

        translated_class = class_name
        translate_tokens_in = translate_tokens_out = 0
        if llm is not None:
            _usage_before = llm.usage_snapshot()
            translated_class = llm.translate_class_name(crop=crop, class_name=class_name)
            _usage_after = llm.usage_snapshot()
            translate_tokens_in = _usage_after["tokens_in"] - _usage_before["tokens_in"]
            translate_tokens_out = _usage_after["tokens_out"] - _usage_before["tokens_out"]

        meta_prompt = best_template.format(**{"class": translated_class})
        meta_map = model.compute_map(samples, [meta_prompt])

        result = MetaZeroShotEvalResult(
            dataset_name=test_split.name,
            class_name=class_name,
            crop=crop,
            naive_map=naive_map,
            template_metadata_map=meta_map,
            template_prompt_metadata=meta_prompt,
            translated_class_name=translated_class,
            class_translate_tokens_in=translate_tokens_in,
            class_translate_tokens_out=translate_tokens_out,
        )

        if vlm_insight is not None:
            try:
                _usage_before = vlm_insight.usage_snapshot()
                insight = vlm_insight.describe(image=samples[0].image, crop=crop, class_name=class_name)
                _usage_after = vlm_insight.usage_snapshot()
                result.vlm_insight_tokens_in = _usage_after["tokens_in"] - _usage_before["tokens_in"]
                result.vlm_insight_tokens_out = _usage_after["tokens_out"] - _usage_before["tokens_out"]
                result.n_unlabeled_images_used = 1
                phrase = insight.get("phrase") or translated_class
                vlm_prompt = best_template.format(**{"class": phrase})
                result.template_vlm_map = model.compute_map(samples, [vlm_prompt])
                result.template_prompt_vlm = vlm_prompt
            except Exception as exc:
                logger.warning(f"VLM insight fill failed for {class_name!r}: {exc}")

        result.wall_clock_seconds = time.time() - _class_start
        logger.info(
            f"  [zero-shot][{test_split.name}][{class_name}] "
            f"naive={naive_map:.4f} template(meta)={meta_map:.4f}"
            + (f" template(vlm)={result.template_vlm_map:.4f}" if result.template_vlm_map is not None else "")
        )
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_meta_template_result(result: MetaTemplateResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fname = output_dir / f"{result.model_name}_meta_template.json"
    data = {
        "model": result.model_name,
        "train_datasets": result.train_datasets,
        "test_datasets": result.test_datasets,
        "baseline_template": result.baseline_template,
        "baseline_aggregate_map": result.baseline_aggregate_map,
        "best_template": result.best_template,
        "best_aggregate_map": result.best_aggregate_map,
        "gain": round(result.best_aggregate_map - result.baseline_aggregate_map, 6),
        "total_iterations": result.total_iterations,
        "patience_exhausted": result.patience_exhausted,
        "total_evaluations": result.total_evaluations,
        "wall_clock_seconds": round(result.wall_clock_seconds, 3),
        "total_tokens_in": result.total_tokens_in,
        "total_tokens_out": result.total_tokens_out,
        "total_api_calls": result.total_api_calls,
        "history": [
            {
                "iteration": r.iteration, "template": r.template, "aggregate_map": round(r.aggregate_map, 6),
                "is_best": r.is_best, "wall_clock_seconds": round(r.wall_clock_seconds, 3),
                "tokens_in": r.tokens_in, "tokens_out": r.tokens_out, "n_api_calls": r.n_api_calls,
            }
            for r in result.history
        ],
        "zero_shot": [
            {
                "dataset": z.dataset_name,
                "class": z.class_name,
                "crop": z.crop,
                "naive_map": z.naive_map,
                "template_metadata_map": z.template_metadata_map,
                "template_prompt_metadata": z.template_prompt_metadata,
                "translated_class_name": z.translated_class_name,
                "template_vlm_map": z.template_vlm_map,
                "template_prompt_vlm": z.template_prompt_vlm,
                "gain_vs_naive_metadata": round(z.template_metadata_map - z.naive_map, 6),
                "gain_vs_naive_vlm": (
                    round(z.template_vlm_map - z.naive_map, 6) if z.template_vlm_map is not None else None
                ),
                "wall_clock_seconds": round(z.wall_clock_seconds, 3),
                "class_translate_tokens_in": z.class_translate_tokens_in,
                "class_translate_tokens_out": z.class_translate_tokens_out,
                "vlm_insight_tokens_in": z.vlm_insight_tokens_in,
                "vlm_insight_tokens_out": z.vlm_insight_tokens_out,
                "n_unlabeled_images_used": z.n_unlabeled_images_used,
            }
            for z in result.zero_shot
        ],
    }
    with open(fname, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved meta-template result → {fname}")
