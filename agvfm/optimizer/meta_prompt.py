"""Meta-prompt optimisation loop.

Unlike the axis-based optimiser in :mod:`agvfm.optimizer` (axis-based loop), which
decomposes prompts into structured axes and exhaustively sweeps them, the
meta-prompt loop treats prompt generation as an open-ended search guided by a
language model (LLM).

Algorithm
---------
For each (dataset, model, class) triple:

1.  **Baseline** – evaluate the raw class label(s) from the config and record
    the score.
2.  **Iterate** – in each iteration:
    a.  Send the LLM the crop/class info and the full history of tested prompts
        sorted by score (best first).
    b.  The LLM returns ``candidates_per_iter`` new prompt strings.
    c.  Evaluate each candidate with the detection model on the proxy set.
    d.  Update the global best if any candidate scores higher.
3.  **Patience** – if ``patience`` consecutive iterations produce no improvement
    over the current best, the search stops.  The baseline is ``patience=10``.

Results are saved as JSON files (one per class) and a dataset-level summary.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from agvfm.optimizer.types import DatasetSplit, Sample, sample_proxy_images
from AgVFM.agvfm.llm.meta_client import VLMClient
from agvfm.optimizer.types import VFMBase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MetaEvalResult:
    """Record of a single prompt evaluation."""
    iteration: int          # 0 = baseline
    prompt: str
    map_score: float
    is_best: bool           # True if this was a new global best at eval time


@dataclass
class MetaPromptConfig:
    """Configuration for the meta-prompt optimisation loop.

    Parameters
    ----------
    proxy_images:
        Number of training images to use for fast proxy evaluation.
    patience:
        Stop after this many consecutive iterations with no improvement over
        the current best.  Default is 10.
    candidates_per_iter:
        Number of new prompt candidates to request from the LLM per iteration.
    max_iterations:
        Hard cap on the number of iterations (safety valve).
    seed:
        Random seed for proxy image sampling.
    """
    proxy_images: int = 30
    patience: int = 10
    candidates_per_iter: int = 5
    max_iterations: int = 100
    seed: int = 42


@dataclass
class MetaPromptResult:
    """Outcome of the meta-prompt search for a single (dataset, model, class)."""
    dataset_name: str
    model_name: str
    class_name: str
    crop: str
    proxy_images: int
    baseline_prompt: str
    baseline_map: float
    best_prompt: str
    best_map: float
    total_iterations: int
    patience_exhausted: bool      # True if stopped by patience, False if hit max_iter
    total_evaluations: int
    history: list[MetaEvalResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def run_meta_prompt_optimization(
    dataset: DatasetSplit,
    model: VFMBase,
    vlm: VLMClient,
    crop: str,
    config: MetaPromptConfig,
    output_dir: Optional[Path] = None,
) -> list[MetaPromptResult]:
    """Run meta-prompt optimisation for every class in ``dataset``.

    Parameters
    ----------
    dataset:
        The :class:`~agvfm.optimizer.types.DatasetSplit` to optimise for.
    model:
        The detection model (VFM) used to evaluate candidate prompts.
    vlm:
        LLM client used to generate candidate prompts.
    crop:
        Human-readable crop name from the config (e.g. ``"apple"``).
    config:
        Hyperparameters for the meta-prompt loop.
    output_dir:
        Directory to write per-class JSON results.  Skipped if ``None``.

    Returns
    -------
    list[MetaPromptResult]
        One result per class in the dataset.
    """
    proxy_samples = sample_proxy_images(dataset, config.proxy_images, seed=config.seed)
    logger.info(
        f"[{dataset.name}][{model.name}] meta-prompt: {len(proxy_samples)} proxy images, "
        f"patience={config.patience}, candidates_per_iter={config.candidates_per_iter}"
    )

    results: list[MetaPromptResult] = []
    for class_name in dataset.classes:
        result = _optimize_class(
            dataset=dataset,
            model=model,
            vlm=vlm,
            class_name=class_name,
            crop=crop,
            proxy_samples=proxy_samples,
            config=config,
        )
        results.append(result)
        _log_summary(result)
        if output_dir:
            _save_result(result, output_dir)

    return results


# ---------------------------------------------------------------------------
# Per-class search loop
# ---------------------------------------------------------------------------

def _optimize_class(
    dataset: DatasetSplit,
    model: VFMBase,
    vlm: VLMClient,
    class_name: str,
    crop: str,
    proxy_samples: list[Sample],
    config: MetaPromptConfig,
) -> MetaPromptResult:
    rng = random.Random(config.seed)
    history: list[MetaEvalResult] = []

    # --- Step 1: Baseline ---------------------------------------------------
    baseline_prompt = class_name
    baseline_map = model.compute_map(proxy_samples, [baseline_prompt])
    logger.info(f"  [{dataset.name}][{class_name}] baseline mAP={baseline_map:.4f}  prompt={baseline_prompt!r}")

    best_map = baseline_map
    best_prompt = baseline_prompt
    history.append(MetaEvalResult(
        iteration=0,
        prompt=baseline_prompt,
        map_score=baseline_map,
        is_best=True,
    ))

    def _history_for_llm() -> list[dict]:
        return sorted(
            [{"prompt": r.prompt, "map_score": r.map_score} for r in history],
            key=lambda d: d["map_score"],
            reverse=True,
        )

    # --- Step 2: Meta-prompt loop -------------------------------------------
    no_improve_count = 0
    patience_exhausted = False
    iteration = 0

    for iteration in range(1, config.max_iterations + 1):
        if no_improve_count >= config.patience:
            patience_exhausted = True
            logger.info(
                f"  [{class_name}] Stopping: {no_improve_count} iterations without improvement "
                f"(patience={config.patience})"
            )
            break

        candidates = vlm.suggest_prompts(
            crop=crop,
            class_name=class_name,
            baseline_labels=dataset.classes,
            history=_history_for_llm(),
            n_candidates=config.candidates_per_iter,
        )

        improved_this_iter = False
        for prompt in candidates:
            if not prompt or not prompt.strip():
                continue
            prompt = prompt.strip()
            map_score = model.compute_map(proxy_samples, [prompt])
            is_new_best = map_score > best_map
            if is_new_best:
                best_map = map_score
                best_prompt = prompt
                improved_this_iter = True
            history.append(MetaEvalResult(
                iteration=iteration,
                prompt=prompt,
                map_score=map_score,
                is_best=is_new_best,
            ))
            logger.debug(
                f"  [{class_name}] iter={iteration}  mAP={map_score:.4f}"
                f"{'  ← new best' if is_new_best else ''}  {prompt!r}"
            )

        if improved_this_iter:
            no_improve_count = 0
            logger.info(
                f"  [{class_name}] iter={iteration}  NEW BEST mAP={best_map:.4f}  {best_prompt!r}"
            )
        else:
            no_improve_count += 1
            logger.info(
                f"  [{class_name}] iter={iteration}  no improvement "
                f"(patience {no_improve_count}/{config.patience})"
            )

    return MetaPromptResult(
        dataset_name=dataset.name,
        model_name=model.name,
        class_name=class_name,
        crop=crop,
        proxy_images=len(proxy_samples),
        baseline_prompt=baseline_prompt,
        baseline_map=baseline_map,
        best_prompt=best_prompt,
        best_map=best_map,
        total_iterations=iteration,
        patience_exhausted=patience_exhausted,
        total_evaluations=len(history),
        history=history,
    )


# ---------------------------------------------------------------------------
# Logging and persistence
# ---------------------------------------------------------------------------

def _log_summary(result: MetaPromptResult) -> None:
    sep = "-" * 60
    gain = result.best_map - result.baseline_map
    stop_reason = "patience" if result.patience_exhausted else "max_iterations"
    logger.info(sep)
    logger.info(
        f"  [{result.dataset_name}][{result.model_name}][{result.class_name}]"
    )
    logger.info(
        f"  baseline={result.baseline_map:.4f}  best={result.best_map:.4f}"
        f"  gain={gain:+.4f}  evals={result.total_evaluations}"
        f"  iters={result.total_iterations}  stopped_by={stop_reason}"
    )
    logger.info(f"  baseline prompt : {result.baseline_prompt!r}")
    logger.info(f"  best prompt     : {result.best_prompt!r}")

    top5 = sorted(result.history, key=lambda r: r.map_score, reverse=True)[:5]
    logger.info("  Top-5 prompts:")
    for rank, entry in enumerate(top5, 1):
        d = entry.map_score - result.baseline_map
        logger.info(f"    #{rank}  mAP={entry.map_score:.4f}  delta={d:+.4f}  {entry.prompt!r}")
    logger.info(sep)


def _save_result(result: MetaPromptResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_class = result.class_name.replace("/", "_").replace(" ", "_")
    fname = output_dir / f"{result.dataset_name}_{result.model_name}_{safe_class}_meta.json"
    data = {
        "dataset": result.dataset_name,
        "model": result.model_name,
        "class": result.class_name,
        "crop": result.crop,
        "proxy_images": result.proxy_images,
        "baseline_prompt": result.baseline_prompt,
        "baseline_map": result.baseline_map,
        "best_prompt": result.best_prompt,
        "best_map": result.best_map,
        "gain": round(result.best_map - result.baseline_map, 6),
        "total_iterations": result.total_iterations,
        "patience_exhausted": result.patience_exhausted,
        "total_evaluations": result.total_evaluations,
        "history": [
            {
                "iteration": r.iteration,
                "prompt": r.prompt,
                "map_score": round(r.map_score, 6),
                "is_best": r.is_best,
            }
            for r in result.history
        ],
    }
    with open(fname, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved meta-prompt result → {fname}")
