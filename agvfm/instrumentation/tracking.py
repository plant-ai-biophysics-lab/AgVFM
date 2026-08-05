"""Structured cost/performance logging shared across every optimizer/LLM module.

This is the instrumentation layer PAPER.md section 5 calls "the single
biggest gap": before this module existed, none of `llm/client.py`,
`llm/meta_client.py`, `llm/vlm_insight_client.py`, `optimizer/meta_prompt.py`,
`optimizer/meta_prompt_template.py`, or `optimizer/grad_prompt.py` captured
token counts, cost, or wall-clock time anywhere. `agvfm.utils.logging`
remains the human-readable text-log layer; this module is the structured
counterpart whose output `agvfm.reporting.aggregate` turns into the paper's
cost/performance comparison table.

Core pieces
-----------
- :class:`CallRecord` — one row per experimental run (PAPER.md section 5.1's
  schema: dataset/model/method/prompt_space/pipeline_role +
  wall_clock/tokens/cost/GPU/labeled-examples/performance fields).
- :class:`RunTracker` — appends :class:`CallRecord` rows to a JSONL file (one
  file per results directory is the expected usage — `aggregate.py` reads a
  whole tree of these).
- :class:`PricingTable` — $/1K-token rates per model, stamped with the date
  they were looked up, stored alongside results so costs stay auditable even
  after real-world pricing changes.
- Token-counting helpers for both LLM backends (:func:`extract_openai_usage`
  for served/OpenAI-compatible responses, :func:`count_tokens_local` for the
  local HF-pipeline backend, which has no `usage` field to read).

Usage
-----
    from agvfm.instrumentation.tracking import RunTracker, CallRecord

    tracker = RunTracker(results_dir / "instrumentation.jsonl")
    with tracker.timed() as timer:
        ... do the work ...
    tracker.log(CallRecord(
        dataset_id="grape_detection_californiaday", model_id="yolo_world",
        method="transfer", prompt_space="constrained",
        wall_clock_seconds=timer.elapsed, n_api_calls=1,
        tokens_in=120, tokens_out=40, usd_cost=0.0002,
        map_score=0.41,
    ))
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pricing table (PAPER.md 5.1: "store the pricing table used, with date")
# ---------------------------------------------------------------------------

PRICING_AS_OF = "2026-07-30"

# USD per 1,000 tokens. Populate/update as models are actually used —
# missing entries degrade to usd_cost=None (never a silently wrong 0.0) so
# gaps are visible in the aggregated table rather than hidden.
PRICING_USD_PER_1K_TOKENS: dict[str, dict[str, float]] = {
    # Served (billed) endpoints — fill in with whatever's actually behind
    # --llm-url/--vlm-insight-url for a given run.
    # "gpt-4o": {"input": 0.0025, "output": 0.010},
    # Local HF-pipeline backend: compute cost is GPU-time, not $/token — see
    # `gpu_hours`/`gpu_seconds` in CallRecord instead. Leave token pricing at
    # $0 for local models so usd_cost reflects "no per-token API bill", but
    # don't let that read as "free" — the GPU-hours column carries the cost.
    "Qwen/Qwen3-4B": {"input": 0.0, "output": 0.0},
    "Qwen/Qwen2-VL-7B-Instruct": {"input": 0.0, "output": 0.0},
}


def estimate_usd_cost(model_id: str, tokens_in: int, tokens_out: int) -> Optional[float]:
    """Look up `model_id` in :data:`PRICING_USD_PER_1K_TOKENS` and price the call.

    Returns ``None`` (not ``0.0``) when the model isn't in the table, so a
    missing price shows up as a gap in the aggregated report rather than a
    misleadingly free run.
    """
    rates = PRICING_USD_PER_1K_TOKENS.get(model_id)
    if rates is None:
        return None
    return (tokens_in / 1000.0) * rates["input"] + (tokens_out / 1000.0) * rates["output"]


# ---------------------------------------------------------------------------
# CallRecord — the structured row schema (PAPER.md 5.1)
# ---------------------------------------------------------------------------

@dataclass
class CallRecord:
    """One structured log row: one dataset × one model × one method × one split.

    Field groups mirror PAPER.md 5.1 exactly so `aggregate.py` doesn't have to
    guess field names:

    - identity: ``dataset_id``, ``model_id``, ``method``, ``split_id``,
      ``held_out_rarity``
    - the central axis of comparison: ``prompt_space``
      (``"constrained"`` | ``"unconstrained"`` | ``"n/a"``) — kept explicit
      rather than inferred from ``method``, per PAPER.md's instruction.
    - VLM-as-unified-pipeline attribution: ``pipeline_role``
      (``"prompt_generation"`` | ``"detection"`` | ``"combined"`` | ``"n/a"``)
    - cost: ``wall_clock_seconds``, ``n_api_calls``, ``tokens_in``,
      ``tokens_out``, ``usd_cost`` (``None`` if unpriced — see
      :func:`estimate_usd_cost`), ``gpu_seconds``, ``gpu_type``
    - data cost: ``n_labeled_examples_used``, ``n_unlabeled_images_used``
    - outcome: ``map_score`` plus a free-form ``extra_metrics`` dict for
      whatever else the calling evaluator produced (F1, best_conf, ...)
    """

    dataset_id: str
    model_id: str
    method: str  # discovery | transfer | metaprompt-cold | metaprompt-summary | pez | lora | vlm_pipeline-constrained | vlm_pipeline-unconstrained
    prompt_space: str  # "constrained" | "unconstrained" | "n/a"
    split_id: str = ""
    held_out_rarity: str = "n/a"  # "common" | "rare" | "n/a" (n/a for training-pool/discovery rows)
    pipeline_role: str = "n/a"    # "prompt_generation" | "detection" | "combined" | "n/a"

    wall_clock_seconds: float = 0.0
    n_api_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    usd_cost: Optional[float] = None
    pricing_as_of: str = PRICING_AS_OF

    gpu_seconds: float = 0.0
    gpu_type: str = ""

    n_labeled_examples_used: int = 0
    n_unlabeled_images_used: int = 0

    map_score: Optional[float] = None
    extra_metrics: dict[str, Any] = field(default_factory=dict)

    run_id: str = ""
    timestamp: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

class _Timer:
    """Returned by :meth:`RunTracker.timed`; ``.elapsed`` is valid after the block exits."""

    def __init__(self) -> None:
        self._start = time.time()
        self.elapsed = 0.0

    def _stop(self) -> None:
        self.elapsed = time.time() - self._start


# ---------------------------------------------------------------------------
# RunTracker — JSONL writer
# ---------------------------------------------------------------------------

class RunTracker:
    """Appends :class:`CallRecord` rows to a JSONL log file.

    One tracker per results directory is the expected pattern — pass the
    same ``log_path`` across every method/model/dataset combination in a run
    so :mod:`agvfm.reporting.aggregate` can read the whole set as one table.
    Safe to share across processes writing to different files; not
    thread/process-safe for concurrent appends to the *same* file.
    """

    def __init__(self, log_path: Path, run_id: str = "") -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or time.strftime("%Y%m%dT%H%M%S")

    @contextmanager
    def timed(self) -> Iterator[_Timer]:
        """Context manager measuring wall-clock time for the enclosed block.

        Usage::

            with tracker.timed() as timer:
                ... work ...
            record.wall_clock_seconds = timer.elapsed
        """
        timer = _Timer()
        try:
            yield timer
        finally:
            timer._stop()

    def log(self, record: CallRecord) -> None:
        """Append one record, filling ``run_id``/``timestamp`` if unset."""
        if not record.run_id:
            record.run_id = self.run_id
        if not record.timestamp:
            record.timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record.as_dict()) + "\n")

    @staticmethod
    def read_all(log_path: Path) -> list[dict]:
        """Read every record from a single JSONL file (skips malformed lines defensively)."""
        records: list[dict] = []
        if not Path(log_path).exists():
            return records
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"Skipping malformed instrumentation record in {log_path}: {line[:120]!r}")
        return records


# ---------------------------------------------------------------------------
# Token-counting helpers
# ---------------------------------------------------------------------------

def extract_openai_usage(response: Any) -> tuple[int, int]:
    """Pull ``(tokens_in, tokens_out)`` from an OpenAI-compatible chat completion response.

    Most OpenAI-compatible servers (vLLM included) return a `usage` field;
    degrades to ``(0, 0)`` rather than raising if it's missing, since some
    servers omit it.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0
    tokens_in = getattr(usage, "prompt_tokens", 0) or 0
    tokens_out = getattr(usage, "completion_tokens", 0) or 0
    return int(tokens_in), int(tokens_out)


def count_tokens_local(text: str, tokenizer: Any = None) -> int:
    """Count tokens in *text* for the local HF-pipeline backend, which has no `usage` field.

    Uses the pipeline's own tokenizer when available (exact); otherwise
    falls back to a whitespace-split estimate (approximate — flagged via the
    return not being used to compute `usd_cost` for local models, only
    `tokens_in`/`tokens_out` for reference).
    """
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text))
        except Exception:
            pass
    return len(text.split())


# ---------------------------------------------------------------------------
# GPU device info (best-effort — never raises)
# ---------------------------------------------------------------------------

def gpu_type_name() -> str:
    """Return the current CUDA device name, or ``""`` if unavailable/on CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return ""
