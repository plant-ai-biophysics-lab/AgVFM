"""Axis-based OFAT + combinatorial prompt optimisation for a single dataset/class.

Generalises :mod:`agvfm.config.experiments`' cowpea-flower-specific
``FACTOR_AXES``/``build_prompt_from_components`` pipeline (the one
``experiments/scripts/experiments/load_and_run.py`` runs) to arbitrary
AgML/disk datasets via :class:`~agvfm.optimizer.axes.PromptAxes`. The
algorithm shape is unchanged from ``load_and_run.py``'s Phase 1 + Phase 2:

1. **Baseline** — bare class-name prompt (``PromptAxes.naive``).
2. **Phase 1 (OFAT)** — vary one structural axis at a time (grammar, color,
   size, anatomy, phenology) from a ``grammar="a"`` baseline; ``taxonomy`` is
   always fixed to the dataset's class name rather than swept, since AgML
   datasets carry one canonical class name rather than cowpea-flower's set of
   alternate species phrasings.
3. **Phase 2 (combinatorial sweeps)** — mirrors ``load_and_run.py``'s
   ``get_configs()`` Table 2 sweeps: Color × Size (best grammar⋆), Grammar ×
   Color, and Anatomy (best grammar⋆ × color⋆) — each derived from Phase 1's
   OFAT winners exactly as ``_best_axis_value_from_ph1`` does there.
4. **Negation** — appended to the top-N base/sweep combinations.
5. **Emoji** — appended to the single best base+negation combination.

Used by ``experiments/scripts/experiments/run.py`` for the original
per-dataset axis-based strategy; superseded for cross-dataset generalisation
by :mod:`agvfm.optimizer.prompt_template`, which pools this same OFAT +
sweep + negation + emoji structure across many training datasets at once and
evaluates the result zero-shot on held-out crops (see PAPER.md section 3.1/3.2).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from agvfm.optimizer.axes import PromptAxes
from agvfm.optimizer.types import DatasetSplit, VFMBase, sample_proxy_images

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    prompt: str
    map_score: float
    phase: str  # baseline | ofat | sweep1 | sweep2 | sweep3 | negation | emoji
    axes: dict


@dataclass
class OfatAxisResult:
    axis: str
    best_value: str
    best_map: float
    delta: float  # vs ofat baseline (grammar="a", rest "")


@dataclass
class PhaseBest:
    phase: str
    prompt: str
    map_score: float
    delta: float  # vs naive baseline


@dataclass
class OptimizationResult:
    dataset_name: str
    model_name: str
    class_name: str
    proxy_images: int
    baseline_map: float
    best_map: float
    best_prompt: str
    best_axes: dict
    axis_values: dict
    ofat_summary: list[OfatAxisResult]
    phase_best: list[PhaseBest]
    total_evaluations: int
    history: list[EvalResult] = field(default_factory=list)


@dataclass
class OptimizationConfig:
    proxy_images: int | list[int] = 30
    top_n_negation: int = 5
    seed: int = 42


def run_optimization(
    dataset: DatasetSplit,
    model: VFMBase,
    axis_values_per_class: dict[str, dict[str, list[str]]],
    crop: str,
    config: OptimizationConfig,
    output_dir: Path | None = None,
) -> list[OptimizationResult]:
    proxy_counts = config.proxy_images if isinstance(config.proxy_images, list) else [config.proxy_images]
    all_results = []

    for n_proxy in proxy_counts:
        proxy_samples = sample_proxy_images(dataset, n_proxy, seed=config.seed)
        logger.info(f"[{dataset.name}][{model.name}] proxy_images={n_proxy}")

        for class_name in dataset.classes:
            axis_values = axis_values_per_class[class_name]

            result = _optimize_class(
                dataset=dataset,
                model=model,
                class_name=class_name,
                proxy_samples=proxy_samples,
                axis_values=axis_values,
                config=config,
            )
            all_results.append(result)
            _log_summary(result)

            if output_dir:
                _save_result(result, output_dir)

    return all_results


def _optimize_class(
    dataset: DatasetSplit,
    model: VFMBase,
    class_name: str,
    proxy_samples: list,
    axis_values: dict[str, list[str]],
    config: OptimizationConfig,
) -> OptimizationResult:
    history: list[EvalResult] = []
    best_map: float
    best_prompt: str
    best_axes: PromptAxes

    def _eval(axes: PromptAxes, phase: str) -> float:
        nonlocal best_map, best_prompt, best_axes
        prompt = axes.to_prompt()
        map_score = model.compute_map(proxy_samples, [prompt])
        history.append(EvalResult(prompt=prompt, map_score=map_score, phase=phase, axes=axes.to_dict()))
        logger.info(f"    [{phase}] mAP={map_score:.4f} prompt={prompt!r}")
        if map_score > best_map:
            best_map = map_score
            best_prompt = prompt
            best_axes = axes
        return map_score

    # Baseline: taxonomy only
    naive = PromptAxes.naive(class_name)
    baseline_map = model.compute_map(proxy_samples, [naive.to_prompt()])
    logger.info(f"    [baseline] mAP={baseline_map:.4f} prompt={naive.to_prompt()!r}")
    history.append(EvalResult(prompt=naive.to_prompt(), map_score=baseline_map, phase="baseline", axes=naive.to_dict()))
    best_map = baseline_map
    best_prompt = naive.to_prompt()
    best_axes = naive

    # Phase 1: OFAT — vary one axis at a time from grammar="a" baseline
    ofat_base = {"grammar": "a", "color": "", "size": "", "anatomy": "",
                 "phenology": "", "negation": "", "emoji": "", "taxonomy": class_name}
    best_per_axis = dict(ofat_base)
    ofat_baseline_map = model.compute_map(proxy_samples, [PromptAxes.from_dict(ofat_base).to_prompt()])
    ofat_summary: list[OfatAxisResult] = []

    for axis_name in ["grammar", "color", "size", "anatomy", "phenology"]:
        best_val = ofat_base[axis_name]
        best_axis_map = ofat_baseline_map

        for val in axis_values.get(axis_name, []):
            candidate = {**ofat_base, axis_name: val}
            map_score = _eval(PromptAxes.from_dict(candidate), "ofat")
            if map_score > best_axis_map:
                best_axis_map = map_score
                best_val = val

        best_per_axis[axis_name] = best_val
        ofat_summary.append(OfatAxisResult(
            axis=axis_name,
            best_value=best_val,
            best_map=best_axis_map,
            delta=best_axis_map - ofat_baseline_map,
        ))

    best_grammar = best_per_axis["grammar"]
    best_color = best_per_axis["color"]

    # Phase 2: Combinatorial sweeps (Table 2)
    base_combos: list[tuple[PromptAxes, float]] = []
    phase_best: dict[str, tuple[str, float]] = {}  # phase → (prompt, map)

    def _track_phase_best(phase: str, prompt: str, map_score: float) -> None:
        if phase not in phase_best or map_score > phase_best[phase][1]:
            phase_best[phase] = (prompt, map_score)

    # Sweep 1: Color × Size, Grammar⋆, Taxonomy
    for color in axis_values.get("color", []):
        for size in axis_values.get("size", []):
            axes = PromptAxes.from_dict({
                "grammar": best_grammar, "color": color, "size": size,
                "taxonomy": class_name, "anatomy": "", "phenology": "", "negation": "", "emoji": "",
            })
            s = _eval(axes, "sweep1")
            base_combos.append((axes, s))
            _track_phase_best("sweep1", axes.to_prompt(), s)

    # Sweep 2: Grammar × Color, Taxonomy
    for grammar in axis_values.get("grammar", []):
        for color in axis_values.get("color", []):
            axes = PromptAxes.from_dict({
                "grammar": grammar, "color": color, "size": "",
                "taxonomy": class_name, "anatomy": "", "phenology": "", "negation": "", "emoji": "",
            })
            s = _eval(axes, "sweep2")
            base_combos.append((axes, s))
            _track_phase_best("sweep2", axes.to_prompt(), s)

    # Sweep 3: Anatomy, Grammar⋆, Color⋆, Taxonomy
    for anatomy in axis_values.get("anatomy", []):
        axes = PromptAxes.from_dict({
            "grammar": best_grammar, "color": best_color, "size": "",
            "taxonomy": class_name, "anatomy": anatomy, "phenology": "", "negation": "", "emoji": "",
        })
        s = _eval(axes, "sweep3")
        base_combos.append((axes, s))
        _track_phase_best("sweep3", axes.to_prompt(), s)

    # Negation: append to top-N base prompts
    top_n = sorted(base_combos, key=lambda x: x[1], reverse=True)[:config.top_n_negation]
    negation_combos: list[tuple[PromptAxes, float]] = []
    for base_axes, _ in top_n:
        for neg in axis_values.get("negation", []):
            if not neg:
                continue
            axes = PromptAxes.from_dict({**base_axes.to_dict(), "negation": neg})
            s = _eval(axes, "negation")
            negation_combos.append((axes, s))
            _track_phase_best("negation", axes.to_prompt(), s)

    # Emoji: append to top-1 overall (best base + negation combo)
    all_combos = base_combos + negation_combos
    if all_combos:
        top1_axes, _ = max(all_combos, key=lambda x: x[1])
        for emoji in axis_values.get("emoji", []):
            if not emoji:
                continue
            axes = PromptAxes.from_dict({**top1_axes.to_dict(), "emoji": emoji})
            s = _eval(axes, "emoji")
            _track_phase_best("emoji", axes.to_prompt(), s)

    phase_best_list = [
        PhaseBest(phase=phase, prompt=prompt, map_score=map_score, delta=map_score - baseline_map)
        for phase, (prompt, map_score) in phase_best.items()
    ]

    return OptimizationResult(
        dataset_name=dataset.name,
        model_name=model.name,
        class_name=class_name,
        proxy_images=len(proxy_samples),
        baseline_map=baseline_map,
        best_map=best_map,
        best_prompt=best_prompt,
        best_axes=best_axes.to_dict(),
        axis_values=axis_values,
        ofat_summary=ofat_summary,
        phase_best=phase_best_list,
        total_evaluations=len(history),
        history=history,
    )


def _log_summary(result: OptimizationResult) -> None:
    sep = "-" * 60
    logger.info(sep)
    logger.info(f"  class={result.class_name!r}  model={result.model_name}  proxy_images={result.proxy_images}")
    logger.info(f"  baseline: {result.baseline_map:.4f}  best: {result.best_map:.4f}  gain: {result.best_map - result.baseline_map:+.4f}  evals: {result.total_evaluations}")

    logger.info("  OFAT axis ranking (by delta vs ofat-baseline):")
    for r in sorted(result.ofat_summary, key=lambda x: x.delta, reverse=True):
        bar = "█" * max(0, round(r.delta * 100)) if r.delta > 0 else ""
        logger.info(f"    {r.axis:<12} best={r.best_value!r:<22} mAP={r.best_map:.4f}  delta={r.delta:+.4f}  {bar}")

    for phase in ["sweep1", "sweep2", "sweep3", "negation", "emoji"]:
        phase_entries = sorted(
            [e for e in result.history if e.phase == phase],
            key=lambda x: x.map_score, reverse=True,
        )
        if not phase_entries:
            continue
        label = {"sweep1": "Sweep 1 (Color×Size)", "sweep2": "Sweep 2 (Grammar×Color)",
                 "sweep3": "Sweep 3 (Anatomy)", "negation": "Negation", "emoji": "Emoji"}[phase]
        logger.info(f"  {label}:")
        for e in phase_entries:
            delta = e.map_score - result.baseline_map
            marker = " ← best" if e.prompt == result.best_prompt else ""
            logger.info(f"    mAP={e.map_score:.4f}  delta={delta:+.4f}  {e.prompt!r}{marker}")

    logger.info(f"  Best prompt : {result.best_prompt!r}")
    logger.info(f"  Best axes   : { {k: v for k, v in result.best_axes.items() if v} }")
    logger.info(sep)


def _save_result(result: OptimizationResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_class = result.class_name.replace("/", "_")
    fname = output_dir / f"{result.dataset_name}_{result.model_name}_{safe_class}.json"
    data = {
        "dataset": result.dataset_name,
        "model": result.model_name,
        "class": result.class_name,
        "proxy_images": result.proxy_images,
        "baseline_map": result.baseline_map,
        "best_map": result.best_map,
        "best_prompt": result.best_prompt,
        "best_axes": result.best_axes,
        "gain": round(result.best_map - result.baseline_map, 6),
        "total_evaluations": result.total_evaluations,
        "axis_values": result.axis_values,
        "ofat_summary": [
            {"axis": r.axis, "best_value": r.best_value, "best_map": r.best_map, "delta": round(r.delta, 6)}
            for r in result.ofat_summary
        ],
        "phase_best": [
            {"phase": pb.phase, "prompt": pb.prompt, "map": pb.map_score, "delta": round(pb.delta, 6)}
            for pb in result.phase_best
        ],
        "history": [
            {"prompt": r.prompt, "map": r.map_score, "phase": r.phase, "axes": r.axes}
            for r in result.history
        ],
    }
    with open(fname, "w") as f:
        json.dump(data, f, indent=2)
