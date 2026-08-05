"""PEZ-style gradient prompt optimisation loop.

Algorithm (per class)
---------------------
1. **Initialise** the continuous prompt P ∈ R^{T × d} from the baseline class
   label's token embeddings (warm start), padded or truncated to the token
   budget T.  P is the *only* parameter updated; the vision/text encoders are
   frozen throughout.

2. **For each step** (up to ``n_steps``):
   a. Project P → P_proj by finding the nearest token embedding for each
      position (discrete projection).
   b. Construct the straight-through estimate:
          P_ste = P + (P_proj - P).detach()
      which equals P_proj in the forward pass but lets gradients flow to P
      in the backward pass (Bengio et al. straight-through estimator).
   c. Compute the loss:
        • ``"cosine"`` — 1 - cosine_sim(text_feat(P_ste), mean_image_feat)
          Fast, all-at-once; image features precomputed once.
        • ``"iou_cls"`` — IoU × classification-score loss on a randomly
          selected proxy image.  Slower per step; measures true detection
          signal.
                • ``"composite"`` — classification + soft spatial GIoU + L1 loss.
                    Keeps the spatial path differentiable via soft selection.
   d. Optional (GDINO only): add fluency regularisation weighted by λ.
   e. Back-propagate and update P with Adam.

3. Every ``eval_every`` steps, decode the current P_proj → string and measure
   mAP on all proxy samples with the detection model.  Track the best.

4. Stop when mAP has not improved for ``patience`` evaluations.

Results
-------
Saved as a JSON file per (dataset, model, class, loss_type) with schema
compatible with the axis-based and meta-prompt result files.
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from agvfm.optimizer.types import DatasetSplit, Sample, sample_proxy_images
from agvfm.optimizer.types import VFMBase
from agvfm.optimizer.grad_adapters import PEZAdapter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration and result types
# ---------------------------------------------------------------------------

@dataclass
class GradOptConfig:
    """Hyperparameters for PEZ gradient prompt optimisation.

    Parameters
    ----------
    n_steps:
        Total gradient update steps.
    lr:
        Adam learning rate for the continuous prompt P.
    token_budget:
        Number of free prompt tokens T (BOS/EOS added on top internally).
    loss_types:
        Which loss variants to run.  Each entry runs a separate optimisation
        and produces its own result.  Valid: ``"cosine"``, ``"iou_cls"``,
        ``"composite"``.
    fluency_lambda:
        Weight of the adjacent-token fluency regularisation term (for GDINO).
        Set to 0.0 to disable.
    eval_every:
        Evaluate mAP with the detection model every this many steps.
    patience:
        Stop after this many evaluations with no mAP improvement.
    proxy_images:
        Number of proxy images for the mAP evaluation during optimisation.
    seed:
        Random seed for proxy image sampling and step-level sample selection.
    random_init:
        If True, initialise the continuous prompt P from a random normal
        distribution instead of the class-name token embeddings.
    iou_loss_images_per_step:
        For the ``"iou_cls"`` and ``"composite"`` losses, how many proxy
        images to use per gradient step (mini-batch to keep GPU memory
        manageable).
    """
    n_steps: int = 500
    lr: float = 5e-3
    token_budget: int = 8
    loss_types: tuple[str, ...] = ("cosine", "iou_cls")
    fluency_lambda: float = 0.1          # 0 → disabled
    eval_every: int = 50
    patience: int = 10                   # in evaluation intervals
    proxy_images: int = 30
    seed: int = 42
    random_init: bool = False
    iou_loss_images_per_step: int = 4    # mini-batch size for iou_cls loss


@dataclass
class GradStepRecord:
    """Lightweight record kept for every step (not necessarily an evaluation)."""
    step: int
    loss: float
    prompt: Optional[str] = None         # non-None only when evaluated
    map_score: Optional[float] = None    # non-None only when evaluated
    is_best: bool = False


@dataclass
class GradOptResult:
    """Outcome of PEZ optimisation for one (dataset, model, class, loss_type)."""
    dataset_name: str
    model_name: str
    class_name: str
    crop: str
    loss_type: str                       # "cosine" | "iou_cls" | "composite"
    proxy_images: int
    token_budget: int
    baseline_prompt: str
    baseline_map: float
    best_prompt: str
    best_map: float
    total_steps: int
    patience_exhausted: bool
    total_evaluations: int
    step_history: list[GradStepRecord] = field(default_factory=list)
    wall_clock_seconds: float = 0.0
    gpu_seconds: float = 0.0  # == wall_clock_seconds when a CUDA device is used, else 0.0


# ---------------------------------------------------------------------------
# Discrete projection (straight-through estimator)
# ---------------------------------------------------------------------------

def _pez_project(
    P: torch.Tensor,   # [T, d]
    E: torch.Tensor,   # [V, d]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project each row of P to the nearest token embedding in E.

    Returns
    -------
    P_ste : torch.Tensor [T, d]
        Straight-through estimate: equals P_proj in the forward pass,
        gradient flows to P in the backward pass.
    token_ids : torch.Tensor [T]
        Indices of the nearest embeddings (for decoding).
    """
    with torch.no_grad():
        # Efficient squared-distance via: ||p-e||² = ||p||² + ||e||² - 2 p·eᵀ
        p_norm = (P ** 2).sum(dim=-1, keepdim=True)           # [T, 1]
        e_norm = (E ** 2).sum(dim=-1, keepdim=True).T          # [1, V]
        dists = p_norm + e_norm - 2 * (P @ E.T)               # [T, V]
        token_ids = dists.argmin(dim=-1)                        # [T]
        P_proj = E[token_ids]                                   # [T, d]

    # Straight-through: grad flows to P
    P_ste = P + (P_proj - P).detach()
    return P_ste, token_ids


# ---------------------------------------------------------------------------
# Core optimisation loop for a single class and loss type
# ---------------------------------------------------------------------------

def pez_optimize(
    adapter: PEZAdapter,
    class_name: str,
    crop: str,
    proxy_samples: list[Sample],
    config: GradOptConfig,
    loss_type: str,
    detection_model: VFMBase,
    image_cosine_feat: Optional[torch.Tensor] = None,  # precomputed for cosine loss
) -> GradOptResult:
    """Run PEZ for one (class, loss_type) combination.

    Parameters
    ----------
    adapter:
        Model-specific PEZ adapter.
    class_name:
        The detection class label (used as initialisation and baseline prompt).
    crop:
        Human-readable crop name from the config.
    proxy_samples:
        Proxy images for mAP evaluation.
    config:
        Optimisation hyperparameters.
    loss_type:
        ``"cosine"``, ``"iou_cls"`` or ``"composite"``.
    detection_model:
        The actual VFM used to compute mAP scores.
    image_cosine_feat:
        Precomputed mean image feature (required when ``loss_type == "cosine"``).

    Returns
    -------
    GradOptResult
    """
    assert loss_type in ("cosine", "iou_cls", "composite"), f"Unknown loss_type: {loss_type!r}"
    if loss_type == "cosine" and image_cosine_feat is None:
        raise ValueError("image_cosine_feat must be provided for loss_type='cosine'")

    _start_time = time.time()
    rng = random.Random(config.seed)
    torch.manual_seed(config.seed)

    device = adapter.device
    E = adapter.embedding_matrix.to(device).detach()

    # -------------------------------------------------------------------
    # Initialise P from baseline token embeddings
    # -------------------------------------------------------------------
    baseline_prompt = class_name

    T = config.token_budget
    with torch.no_grad():
        if config.random_init:
            P_init = torch.randn(T, E.shape[1], device=device)
        else:
            baseline_ids = adapter.tokenize(class_name, max_length=config.token_budget)
            # Build initial P: slice or zero-pad to T tokens
            init_embeds = E[baseline_ids]                  # [len, d]
            if init_embeds.shape[0] >= T:
                P_init = init_embeds[:T].clone()
            else:
                pad_count = T - init_embeds.shape[0]
                pad = E[adapter.pad_token_id].unsqueeze(0).expand(pad_count, -1)
                P_init = torch.cat([init_embeds, pad], dim=0).clone()

    P = P_init.float().requires_grad_(True)
    optimiser = torch.optim.Adam([P], lr=config.lr)

    # -------------------------------------------------------------------
    # Baseline mAP
    # -------------------------------------------------------------------
    baseline_map = detection_model.compute_map(proxy_samples, [baseline_prompt])
    logger.info(
        f"  [{class_name}][{loss_type}] baseline mAP={baseline_map:.4f}  "
        f"prompt={baseline_prompt!r}"
    )

    best_map = baseline_map
    best_prompt = baseline_prompt
    no_improve_evals = 0
    step_history: list[GradStepRecord] = []
    eval_count = 0
    patience_exhausted = False

    # -------------------------------------------------------------------
    # Gradient optimisation steps
    # -------------------------------------------------------------------
    for step in range(1, config.n_steps + 1):
        P_ste, token_ids = _pez_project(P, E)

        # Compute primary loss
        if loss_type == "cosine":
            text_feat = adapter.encode_text(P_ste)
            loss = 1.0 - F.cosine_similarity(
                text_feat.unsqueeze(0),
                image_cosine_feat.unsqueeze(0),
            )
        elif loss_type in ("iou_cls", "composite"):
            # Randomly sample a mini-batch of proxy images
            batch = rng.sample(
                proxy_samples,
                min(config.iou_loss_images_per_step, len(proxy_samples)),
            )
            if loss_type == "iou_cls":
                losses = [adapter.compute_iou_cls_loss(s, P_ste) for s in batch]
            else:
                losses = []
                for s in batch:
                    try:
                        losses.append(adapter.compute_composite_loss(s, P_ste))
                    except NotImplementedError:
                        losses.append(adapter.compute_iou_cls_loss(s, P_ste))
            loss = torch.stack(losses).mean()
        else:
            raise AssertionError(f"Unhandled loss_type: {loss_type!r}")

        # Optional fluency regularisation (GDINO only)
        if config.fluency_lambda > 0 and hasattr(adapter, "compute_fluency_loss"):
            fluency = adapter.compute_fluency_loss(P_ste)
            loss = (1.0 - config.fluency_lambda) * loss + config.fluency_lambda * fluency

        loss.backward()
        optimiser.step()
        optimiser.zero_grad()

        rec = GradStepRecord(step=step, loss=float(loss.detach().cpu()))

        # -------------------------------------------------------------------
        # Periodic mAP evaluation
        # -------------------------------------------------------------------
        if step % config.eval_every == 0 or step == config.n_steps:
            with torch.no_grad():
                _, eval_ids = _pez_project(P, E)
                decoded = adapter.decode(eval_ids)

            if not decoded:
                decoded = baseline_prompt

            map_score = detection_model.compute_map(proxy_samples, [decoded])
            eval_count += 1
            is_new_best = map_score > best_map
            rec.prompt = decoded
            rec.map_score = map_score
            rec.is_best = is_new_best

            if is_new_best:
                best_map = map_score
                best_prompt = decoded
                no_improve_evals = 0
                logger.info(
                    f"  [{class_name}][{loss_type}] step={step}  "
                    f"NEW BEST mAP={best_map:.4f}  loss={float(loss):.4f}  {decoded!r}"
                )
            else:
                no_improve_evals += 1
                logger.debug(
                    f"  [{class_name}][{loss_type}] step={step}  "
                    f"mAP={map_score:.4f}  loss={float(loss):.4f}  "
                    f"no_improve={no_improve_evals}/{config.patience}"
                )

            if no_improve_evals >= config.patience:
                patience_exhausted = True
                logger.info(
                    f"  [{class_name}][{loss_type}] Early stop at step {step} "
                    f"({no_improve_evals} evaluations without improvement)"
                )
                step_history.append(rec)
                break

        step_history.append(rec)

    elapsed = time.time() - _start_time
    return GradOptResult(
        dataset_name="",        # filled by caller
        model_name="",          # filled by caller
        class_name=class_name,
        crop=crop,
        loss_type=loss_type,
        proxy_images=len(proxy_samples),
        token_budget=config.token_budget,
        baseline_prompt=baseline_prompt,
        baseline_map=baseline_map,
        best_prompt=best_prompt,
        best_map=best_map,
        total_steps=step_history[-1].step if step_history else 0,
        patience_exhausted=patience_exhausted,
        total_evaluations=eval_count,
        step_history=step_history,
        wall_clock_seconds=elapsed,
        # Wall-clock while resident on a CUDA device, as a training-time proxy —
        # not per-kernel CUDA-event timing (would need extra synchronization
        # overhead in the hot loop for marginal precision gain here).
        gpu_seconds=elapsed if str(device).startswith("cuda") else 0.0,
    )


# ---------------------------------------------------------------------------
# Dataset-level runner
# ---------------------------------------------------------------------------

def run_grad_optimization(
    dataset: DatasetSplit,
    detection_model: VFMBase,
    adapter: PEZAdapter,
    crop: str,
    config: GradOptConfig,
    output_dir: Optional[Path] = None,
) -> list[GradOptResult]:
    """Run PEZ optimisation for all classes in ``dataset`` and all configured
    loss types.

    Parameters
    ----------
    dataset:
        The dataset split to optimise for.
    detection_model:
        The VFM used for mAP evaluation (not for gradient computation).
    adapter:
        Model-specific PEZ adapter (text encoder + image feature extractor).
    crop:
        Human-readable crop name from the config.
    config:
        Optimisation hyperparameters.
    output_dir:
        Directory to write per-(class, loss_type) JSON files.

    Returns
    -------
    list[GradOptResult]
        One result per (class, loss_type) combination.
    """
    proxy_samples = sample_proxy_images(dataset, config.proxy_images, seed=config.seed)
    logger.info(
        f"[{dataset.name}][{detection_model.name}] grad-opt: "
        f"{len(proxy_samples)} proxy images, "
        f"loss_types={list(config.loss_types)}, "
        f"T={config.token_budget}, n_steps={config.n_steps}"
    )

    all_results: list[GradOptResult] = []

    for class_name in dataset.classes:
        # Precompute image features once per class (both loss types can reuse)
        logger.info(f"  Precomputing image features for class {class_name!r} …")
        if "cosine" in config.loss_types:
            image_cosine_feat = adapter.precompute_image_cosine_feat(proxy_samples)
        else:
            image_cosine_feat = None

        for loss_type in config.loss_types:
            logger.info(
                f"  [{dataset.name}][{detection_model.name}][{class_name}] "
                f"loss_type={loss_type}"
            )
            result = pez_optimize(
                adapter=adapter,
                class_name=class_name,
                crop=crop,
                proxy_samples=proxy_samples,
                config=config,
                loss_type=loss_type,
                detection_model=detection_model,
                image_cosine_feat=image_cosine_feat if loss_type == "cosine" else None,
            )
            result.dataset_name = dataset.name
            result.model_name = detection_model.name

            _log_summary(result)
            all_results.append(result)

            if output_dir:
                _save_result(result, output_dir)

    return all_results


# ---------------------------------------------------------------------------
# Logging and persistence
# ---------------------------------------------------------------------------

def _log_summary(result: GradOptResult) -> None:
    sep = "-" * 60
    gain = result.best_map - result.baseline_map
    stop = "patience" if result.patience_exhausted else "max_steps"
    logger.info(sep)
    logger.info(
        f"  [{result.dataset_name}][{result.model_name}]"
        f"[{result.class_name}][{result.loss_type}]"
    )
    logger.info(
        f"  baseline={result.baseline_map:.4f}  best={result.best_map:.4f}"
        f"  gain={gain:+.4f}  evals={result.total_evaluations}"
        f"  steps={result.total_steps}  stopped_by={stop}"
    )
    logger.info(f"  baseline prompt : {result.baseline_prompt!r}")
    logger.info(f"  best prompt     : {result.best_prompt!r}")
    # Top-3 evaluated prompts
    evals = [r for r in result.step_history if r.prompt is not None]
    top3 = sorted(evals, key=lambda r: (r.map_score or 0.0), reverse=True)[:3]
    if top3:
        logger.info("  Top-3 evaluated prompts:")
        for rank, e in enumerate(top3, 1):
            d = (e.map_score or 0.0) - result.baseline_map
            logger.info(
                f"    #{rank}  step={e.step}  mAP={e.map_score:.4f}"
                f"  delta={d:+.4f}  {e.prompt!r}"
            )
    logger.info(sep)


def _save_result(result: GradOptResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_class = result.class_name.replace("/", "_").replace(" ", "_")
    fname = (
        output_dir
        / f"{result.dataset_name}_{result.model_name}_{safe_class}_{result.loss_type}_grad.json"
    )
    data = {
        "dataset": result.dataset_name,
        "model": result.model_name,
        "class": result.class_name,
        "crop": result.crop,
        "loss_type": result.loss_type,
        "proxy_images": result.proxy_images,
        "token_budget": result.token_budget,
        "baseline_prompt": result.baseline_prompt,
        "baseline_map": result.baseline_map,
        "best_prompt": result.best_prompt,
        "best_map": result.best_map,
        "gain": round(result.best_map - result.baseline_map, 6),
        "total_steps": result.total_steps,
        "patience_exhausted": result.patience_exhausted,
        "total_evaluations": result.total_evaluations,
        "wall_clock_seconds": round(result.wall_clock_seconds, 3),
        "gpu_seconds": round(result.gpu_seconds, 3),
        "step_history": [
            {
                "step": r.step,
                "loss": round(r.loss, 6),
                "prompt": r.prompt,
                "map_score": round(r.map_score, 6) if r.map_score is not None else None,
                "is_best": r.is_best,
            }
            for r in result.step_history
            if r.prompt is not None  # only save evaluated steps to keep files small
        ],
    }
    with open(fname, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved grad result → {fname}")
