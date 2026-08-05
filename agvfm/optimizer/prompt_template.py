"""Cross-dataset prompt-*template* optimisation for zero-shot transfer.

:mod:`agvfm.optimizer.loop` tunes a prompt per (dataset, class) by sweeping
:class:`~agvfm.optimizer.axes.PromptAxes`. This module instead tunes ONE set
of structural axis values — a *template* such as ``"a ripe {color}
{taxonomy} {anatomy}"`` — against a pool of samples mixed across many
**train** datasets/crops at once, so the result is a template whose
non-taxonomy slots are expected to generalise rather than overfit to one
crop.

The winning template is then evaluated zero-shot on **held-out test**
datasets the optimiser never saw, filling the ``taxonomy`` slot from plain
metadata (the class name) and, optionally, filling the descriptive slots
(color/size/anatomy/phenology) from a VLM's visual read of a held-out sample
image instead of the value tuned on the train mix — see
:class:`~agvfm.llm.vlm_insight_client.VLMInsightClient`.

This is the "discovery" (PAPER.md 3.1) + "constrained zero-shot transfer"
(PAPER.md 3.2) pairing built directly on the axis/template representation,
run in parallel across the training pool rather than one dataset at a time.
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
from agvfm.llm.vlm_insight_client import VLMInsightClient
from agvfm.optimizer.axes import PromptAxes
from agvfm.optimizer.types import DatasetSplit, Sample, VFMBase, _xywh_to_xyxy, sample_proxy_images

logger = logging.getLogger(__name__)

# Structural axes searched by the template optimiser. "taxonomy" is always
# filled per-group/per-class from metadata, never searched.
_STRUCTURAL_AXES = ["grammar", "color", "size", "anatomy", "phenology"]

# Fixed reference entry always present in the per-crop axes file (see
# generate_axis_values_per_crop) — the original single-dataset baseline this
# whole cross-dataset template pipeline generalizes beyond.
_COWPEA_BASELINE_KEY = "cowpea flower"


# ---------------------------------------------------------------------------
# Per-crop axis-value generation (combinatorial/axis-based discovery only —
# NOT used by meta_prompt_template.py's free-text template search)
# ---------------------------------------------------------------------------

def generate_axis_values_per_crop(
    llm,
    crops: list[str],
    axes_file: Path,
    max_values_per_axis: int = 6,
) -> dict[str, list[str]]:
    """Generate (or reuse) per-crop axis candidates, pooled into one union ``axis_values`` dict.

    For every crop in *crops* not already present in *axes_file*, calls
    ``llm.translate_factor_axes(crop)`` — which anchors the request on the
    original cowpea-flower ``FACTOR_AXES`` baseline the same way
    ``load_and_run.py``'s Qwen-based translation always has, rather than
    generating values for an unanchored "generic crop" cold. Every crop's
    result (plus the literal cowpea-flower baseline itself, kept as a fixed
    reference/comparison entry) is persisted to *axes_file* as JSON, so a
    repeat run against the same training pool reuses it instead of
    re-querying the LLM — this is the cross-dataset-pool analogue of
    ``load_and_run.py``'s single-crop ``factor_axes.json``.

    :func:`run_template_ofat`'s OFAT sweep consumes one ``dict[str,
    list[str]]`` of candidate values shared across the whole pooled training
    mix (a template's structural axes are meant to generalize across crops,
    not vary per crop), so this returns the *union* of every requested
    crop's candidate values per axis, deduplicated — the per-crop breakdown
    lives in *axes_file* itself for inspection/reuse, not in the return value.

    The union is capped at *max_values_per_axis* per axis (round-robin
    across crops, so every crop gets a fair shot at contributing a value
    rather than the first crop's list silently crowding out the rest) —
    without a cap, OFAT sweep cost scales linearly with pool size (N crops
    × ~5-8 raw values each), which stops being a "quick discovery pass"
    somewhere around 3-4 pooled crops.
    """
    from agvfm.config.experiments import FACTOR_AXES

    axes_file = Path(axes_file)
    axes_by_crop: dict[str, list[dict]] = {}
    if axes_file.exists():
        with open(axes_file) as f:
            axes_by_crop = json.load(f)
        logger.info(f"Loaded existing per-crop axes from {axes_file} ({len(axes_by_crop)} crop(s))")

    changed = _COWPEA_BASELINE_KEY not in axes_by_crop
    if changed:
        axes_by_crop[_COWPEA_BASELINE_KEY] = [
            {"name": ax.name, "values": list(ax.values), "baseline": ax.baseline}
            for ax in FACTOR_AXES
        ]

    for crop in crops:
        if crop in axes_by_crop:
            logger.info(f"Reusing cached axes for crop={crop!r} from {axes_file}")
            continue
        logger.info(f"Generating axis values for crop={crop!r} (translated from the cowpea-flower baseline)")
        axes_by_crop[crop] = llm.translate_factor_axes(crop=crop)
        changed = True

    if changed or not axes_file.exists():
        axes_file.parent.mkdir(parents=True, exist_ok=True)
        with open(axes_file, "w") as f:
            json.dump(axes_by_crop, f, indent=2)
        logger.info(f"Saved per-crop axes ({len(axes_by_crop)} crop(s), incl. cowpea-flower baseline) → {axes_file}")

    # Pool every requested crop's candidate values into one union per axis,
    # round-robin across crops and capped at max_values_per_axis so sweep
    # cost stays bounded regardless of how many crops are in the pool.
    axis_names = (*_STRUCTURAL_AXES, "negation", "emoji")
    per_crop_values: dict[str, dict[str, list[str]]] = {
        crop: {e["name"]: e.get("values", []) for e in axes_by_crop.get(crop, []) if e.get("name") in axis_names}
        for crop in crops
    }

    pooled: dict[str, list[str]] = {axis: [] for axis in axis_names}
    for axis in axis_names:
        seen: set[str] = set()
        cursors = {crop: 0 for crop in crops}
        while len(pooled[axis]) < max_values_per_axis and any(
            cursors[crop] < len(per_crop_values[crop].get(axis, [])) for crop in crops
        ):
            for crop in crops:
                if len(pooled[axis]) >= max_values_per_axis:
                    break
                values = per_crop_values[crop].get(axis, [])
                idx = cursors[crop]
                if idx >= len(values):
                    continue
                cursors[crop] += 1
                v = values[idx]
                if v not in seen:
                    seen.add(v)
                    pooled[axis].append(v)
        if "" not in pooled[axis] and any("" in per_crop_values[crop].get(axis, []) for crop in crops):
            pooled[axis].append("")  # keep the no-value baseline candidate even if it got crowded out

    return pooled


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TrainGroup:
    """One (dataset, class) slice of the pooled cross-dataset training mix."""
    dataset_name: str
    crop: str
    class_name: str
    samples: list[Sample]


@dataclass
class TemplateAxisResult:
    axis: str
    best_value: str
    aggregate_map: float
    delta: float  # vs template ofat-baseline
    wall_clock_seconds: float = 0.0  # time spent sweeping this axis's candidate values


@dataclass
class ZeroShotEvalResult:
    dataset_name: str
    class_name: str
    crop: str
    naive_map: float                       # bare class-name baseline
    template_metadata_map: float            # template filled from crop/class metadata only
    template_prompt_metadata: str
    translated_class_name: str = ""        # class_name after LLMClient.translate_class_name, if used
    template_vlm_map: Optional[float] = None       # template filled using VLM visual insight
    template_prompt_vlm: Optional[str] = None
    wall_clock_seconds: float = 0.0
    # Populated only when llm (class-name translation) is used — diffed from
    # LLMClient.usage_snapshot() around the translate_class_name() call.
    class_translate_tokens_in: int = 0
    class_translate_tokens_out: int = 0
    # Populated only when vlm_insight is used — diffed from VLMInsightClient.usage_snapshot()
    # around this class's describe() call, so cost is attributable per (dataset, class).
    vlm_insight_tokens_in: int = 0
    vlm_insight_tokens_out: int = 0
    n_unlabeled_images_used: int = 0


@dataclass
class TemplateResult:
    model_name: str
    train_datasets: list[str]
    test_datasets: list[str]
    baseline_aggregate_map: float
    best_aggregate_map: float
    best_axes: dict                        # structural slots only (no taxonomy)
    axis_values: dict
    ofat_summary: list[TemplateAxisResult]
    zero_shot: list[ZeroShotEvalResult] = field(default_factory=list)
    wall_clock_seconds: float = 0.0  # total discovery time (OFAT sweep + best-combo eval)


# ---------------------------------------------------------------------------
# Building the pooled cross-dataset training mix
# ---------------------------------------------------------------------------

def build_train_groups(
    train_splits: dict[str, DatasetSplit],
    train_ds_cfgs: dict[str, dict],
    n_per_class: int,
    seed: int = 42,
) -> list[TrainGroup]:
    """Sample ~``n_per_class`` images per class from every train dataset and pool them.

    Single-class datasets sample ``n_per_class`` images total (there being
    only one class to fill); multi-class datasets sample ``n_per_class``
    images per class via :func:`~agvfm.data.agml_loader.sample_proxy_images_for_class`.
    """
    groups: list[TrainGroup] = []
    for name, split in train_splits.items():
        crop = train_ds_cfgs[name]["crop"]
        for class_name in split.classes:
            if len(split.classes) == 1:
                samples = sample_proxy_images(split, n_per_class, seed=seed)
            else:
                samples = sample_proxy_images_for_class(split, class_name, n_per_class, seed=seed)
            if samples:
                groups.append(TrainGroup(dataset_name=name, crop=crop, class_name=class_name, samples=samples))
            else:
                logger.warning(f"No samples found for {name}/{class_name!r} — dropping from training mix")
    return groups


# ---------------------------------------------------------------------------
# Aggregate scoring across the pooled mix
# ---------------------------------------------------------------------------

def aggregate_map(
    model: VFMBase,
    groups: list[TrainGroup],
    axes_dict: dict,
    iou_threshold: float = 0.5,
) -> float:
    """Score one structural axis combination across every group in ``groups``.

    ``axes_dict`` holds the structural slots only (no "taxonomy"); the class
    name of each group is substituted in before evaluating. Every sample's
    ground-truth boxes and every detection's box+confidence are pooled across
    ALL groups into one precision-recall curve, then scored as mAP@*iou_threshold*
    (101-point interpolated AP, :func:`agvfm.evaluation.metrics.compute_metrics_at_iou`
    — the same metric ``load_and_run.py``'s ``Evaluator`` reports) — so larger
    datasets/classes naturally weigh more, matching how
    :meth:`~agvfm.optimizer.types.VFMBase.compute_map` aggregates within one
    dataset. Replaces the F1-at-implicit-threshold proxy this used to compute.
    """
    list_gt_xyxy: list[np.ndarray] = []
    list_pred_xyxy: list[np.ndarray] = []
    list_pred_conf: list[np.ndarray] = []

    for group in groups:
        prompt = PromptAxes.from_dict({**axes_dict, "taxonomy": group.class_name}).to_prompt()
        for sample in group.samples:
            preds = model.predict(sample.image, [prompt])
            # Only this group's class counts as ground truth — a multi-class
            # dataset's sample (e.g. ghai_broccoli_detection's Canopy/Crown/crop)
            # otherwise contaminates every other class's score with boxes the
            # single-class prompt was never asked to find.
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


# ---------------------------------------------------------------------------
# Template OFAT search
# ---------------------------------------------------------------------------

def run_template_ofat(
    model: VFMBase,
    groups: list[TrainGroup],
    axis_values: dict[str, list[str]],
    train_datasets: list[str],
    test_datasets: list[str],
) -> TemplateResult:
    """Sweep one structural axis at a time (OFAT) against the pooled train mix.

    Mirrors the OFAT phase in :func:`agvfm.optimizer.loop.run_optimization`,
    except every evaluation is an :func:`aggregate_map` over ALL pooled groups
    rather than one dataset's proxy set, so the winning per-axis value is one
    that generalises across the mixed crops rather than one that overfits a
    single class.
    """
    if not groups:
        raise ValueError("No training groups to optimise a template against.")

    _run_start = time.time()
    naive_map = aggregate_map(model, groups, {})
    logger.info(f"[template] naive baseline aggregate mAP={naive_map:.4f}")

    ofat_base = {"grammar": "a", "color": "", "size": "", "anatomy": "", "phenology": ""}
    ofat_baseline_map = aggregate_map(model, groups, ofat_base)

    best_per_axis = dict(ofat_base)
    ofat_summary: list[TemplateAxisResult] = []

    for axis_name in _STRUCTURAL_AXES:
        _axis_start = time.time()
        best_val = ofat_base[axis_name]
        best_axis_map = ofat_baseline_map

        for val in axis_values.get(axis_name, []):
            candidate = {**ofat_base, axis_name: val}
            score = aggregate_map(model, groups, candidate)
            logger.debug(f"  [template ofat][{axis_name}={val!r}] aggregate mAP={score:.4f}")
            if score > best_axis_map:
                best_axis_map = score
                best_val = val

        best_per_axis[axis_name] = best_val
        ofat_summary.append(TemplateAxisResult(
            axis=axis_name,
            best_value=best_val,
            aggregate_map=best_axis_map,
            delta=best_axis_map - ofat_baseline_map,
            wall_clock_seconds=time.time() - _axis_start,
        ))
        logger.info(
            f"  [template ofat] axis={axis_name:<10} best={best_val!r:<20} "
            f"aggregate mAP={best_axis_map:.4f}  delta={best_axis_map - ofat_baseline_map:+.4f}"
        )

    best_map = aggregate_map(model, groups, best_per_axis)
    logger.info(f"[template] combined best-per-axis aggregate mAP={best_map:.4f}")

    return TemplateResult(
        model_name=model.name,
        train_datasets=train_datasets,
        test_datasets=test_datasets,
        baseline_aggregate_map=naive_map,
        best_aggregate_map=best_map,
        best_axes=best_per_axis,
        axis_values=axis_values,
        ofat_summary=ofat_summary,
        wall_clock_seconds=time.time() - _run_start,
    )


# ---------------------------------------------------------------------------
# Zero-shot evaluation on held-out test datasets
# ---------------------------------------------------------------------------

def evaluate_zero_shot(
    model: VFMBase,
    test_split: DatasetSplit,
    crop: str,
    best_axes: dict,
    n_eval_images: Optional[int] = 20,
    seed: int = 42,
    vlm_insight: Optional[VLMInsightClient] = None,
    llm=None,
) -> list[ZeroShotEvalResult]:
    """Evaluate the discovered template zero-shot on every class of ``test_split``.

    For each class: a naive (bare class-name) baseline, the template filled
    from crop/class metadata only, and — if ``vlm_insight`` is provided — the
    template filled using the VLM's visual read of one sample image instead
    of the tuned generic values.

    ``best_axes`` (the structural grammar/color/size/anatomy/phenology values
    discovered against the *training* pool) is used completely unchanged —
    that's the actual transfer test; re-deriving them per held-out crop would
    just be discovery again, not zero-shot transfer. The only thing that
    varies per held-out crop is the taxonomy slot: when ``llm`` (an
    :class:`~agvfm.llm.client.LLMClient`) is given, the raw dataset
    ``class_name`` is translated to a natural phrase via
    ``llm.translate_class_name(crop, class_name)`` — a single realistic
    zero-shot input (crop + class name, nothing else) producing a single
    output, not the multi-candidate list used during training's axis search
    — before filling the template's taxonomy slot; without ``llm``, the raw
    ``class_name`` is used as before. The naive baseline always uses the raw
    ``class_name`` regardless (it's meant to be un-engineered).
    """
    results: list[ZeroShotEvalResult] = []

    for class_name in test_split.classes:
        _class_start = time.time()
        if len(test_split.classes) == 1:
            samples = sample_proxy_images(test_split, n_eval_images, seed=seed)
        else:
            samples = sample_proxy_images_for_class(test_split, class_name, n_eval_images, seed=seed)
        if not samples:
            logger.warning(f"No samples for held-out {test_split.name}/{class_name!r} — skipping")
            continue

        naive_prompt = class_name
        naive_map = model.compute_map(samples, [naive_prompt])

        translated_class = class_name
        translate_tokens_in = translate_tokens_out = 0
        if llm is not None:
            _usage_before = llm.usage_snapshot()
            translated_class = llm.translate_class_name(crop=crop, class_name=class_name)
            _usage_after = llm.usage_snapshot()
            translate_tokens_in = _usage_after["tokens_in"] - _usage_before["tokens_in"]
            translate_tokens_out = _usage_after["tokens_out"] - _usage_before["tokens_out"]

        meta_axes = PromptAxes.from_dict({**best_axes, "taxonomy": translated_class})
        meta_prompt = meta_axes.to_prompt()
        meta_map = model.compute_map(samples, [meta_prompt])

        result = ZeroShotEvalResult(
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
                vlm_axes_dict = {**best_axes, "taxonomy": translated_class}
                for k in ("color", "size", "anatomy", "phenology"):
                    if insight.get(k):
                        vlm_axes_dict[k] = insight[k]
                vlm_prompt = PromptAxes.from_dict(vlm_axes_dict).to_prompt()
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

def save_template_result(result: TemplateResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fname = output_dir / f"{result.model_name}_template.json"
    data = {
        "model": result.model_name,
        "train_datasets": result.train_datasets,
        "test_datasets": result.test_datasets,
        "baseline_aggregate_map": result.baseline_aggregate_map,
        "best_aggregate_map": result.best_aggregate_map,
        "gain": round(result.best_aggregate_map - result.baseline_aggregate_map, 6),
        "best_axes": result.best_axes,
        "axis_values": result.axis_values,
        "wall_clock_seconds": round(result.wall_clock_seconds, 3),
        "ofat_summary": [
            {
                "axis": r.axis, "best_value": r.best_value, "aggregate_map": r.aggregate_map,
                "delta": round(r.delta, 6), "wall_clock_seconds": round(r.wall_clock_seconds, 3),
            }
            for r in result.ofat_summary
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
    logger.info(f"Saved template result → {fname}")
