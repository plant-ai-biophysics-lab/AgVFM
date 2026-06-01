"""MetaPromptClient — text-only LLM client for meta-prompt generation.

Uses the same OpenAI-compatible endpoint as :mod:`agvfm.llm.client` but
sends full prompt strings rather than axis values.  The VLM/vision path has
been removed; image context is a possible future extension.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt for the meta-prompting LLM
# ---------------------------------------------------------------------------

META_SYSTEM_PROMPT = """\
You are an agricultural computer vision expert specialising in zero-shot object detection.
Your task is to generate improved text prompts for detecting specific crop objects in farm images.

The prompts will be passed directly as text queries to open-vocabulary vision models such as
YOLO-World, Grounding DINO, and OWL-V2.  Based on the history provided and your intuition, propose new prompts that are meaningfully different from previous ones and likely to yield better detection scores.

You will be shown:
  1. The crop type and the detection class you must improve prompts for.
  2. A history of previously tested prompts ordered by their detection score (mAP / F1).

Study the history to understand what has worked and what has not, then generate fresh candidate
prompts that are meaningfully different from any prompt already in the history.

Return a JSON array of EXACTLY {n_candidates} prompt strings and nothing else.
Example for class "apple":
["a ripe red apple on a tree branch", "green apple fruit cluster", "single apple with stem visible"]"""


class VLMClient:
    """Text-only LLM client for meta-prompt generation.

    Uses the same OpenAI-compatible endpoint as :class:`~agvfm.llm.client.LLMClient`.

    Parameters
    ----------
    base_url:
        Base URL of the OpenAI-compatible inference server.
    model:
        Model name / path understood by the server.
    temperature:
        Sampling temperature (higher → more diverse suggestions).
    max_tokens:
        Maximum tokens in the model's response.
    max_retries:
        Number of parse/request retries before falling back to heuristic prompts.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float = 0.8,
        max_tokens: int = 512,
        max_retries: int = 3,
    ) -> None:
        self._client = OpenAI(base_url=base_url, api_key="EMPTY")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def suggest_prompts(
        self,
        crop: str,
        class_name: str,
        baseline_labels: list[str],
        history: list[dict],          # list of {"prompt": str, "map_score": float}
        n_candidates: int = 5,
    ) -> list[str]:
        """Ask the LLM to propose ``n_candidates`` new detection prompts.

        Parameters
        ----------
        crop:
            Human-readable crop name from the dataset config (e.g. ``"apple"``).
        class_name:
            The specific detection class to optimise prompts for.
        baseline_labels:
            All class labels present in the dataset — provides context about
            what else the model must distinguish between.
        history:
            Previously evaluated prompts with their mAP scores, sorted by score
            descending (best first).  May be empty on the first iteration.
        n_candidates:
            Number of new candidate prompts to request.

        Returns
        -------
        list[str]
            A list of exactly ``n_candidates`` prompt strings.  Falls back to
            heuristic prompts if the LLM response cannot be parsed.
        """
        system = META_SYSTEM_PROMPT.format(n_candidates=n_candidates)
        user_content = _build_user_content(
            crop=crop,
            class_name=class_name,
            baseline_labels=baseline_labels,
            history=history,
            n_candidates=n_candidates,
        )

        for attempt in range(self.max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                raw = response.choices[0].message.content.strip()
                candidates = _parse_prompt_list(raw, n_candidates)
                # Filter out exact duplicates with history
                seen = {h["prompt"] for h in history}
                candidates = [c for c in candidates if c not in seen]
                if candidates:
                    logger.debug(f"LLM suggested {len(candidates)} new prompts (attempt {attempt + 1})")
                    return candidates[:n_candidates]
                logger.warning(
                    f"All {n_candidates} LLM suggestions were already in history "
                    f"(attempt {attempt + 1}/{self.max_retries}); retrying."
                )
            except Exception as exc:
                logger.warning(f"LLM request failed (attempt {attempt + 1}/{self.max_retries}): {exc}")

        logger.warning("Falling back to heuristic prompt suggestions after repeated failures.")
        return _heuristic_prompts(crop, class_name, n_candidates, history)


# ---------------------------------------------------------------------------
# Message building helpers
# ---------------------------------------------------------------------------

def _build_user_content(
    crop: str,
    class_name: str,
    baseline_labels: list[str],
    history: list[dict],
    n_candidates: int,
) -> str:
    text_intro = (
        f"Crop: {crop}\n"
        f"Target detection class: {class_name!r}\n"
        f"All classes in this dataset: {', '.join(baseline_labels)}\n"
    )
    history_section = _format_history(history)
    return (
        f"{text_intro}"
        f"\n--- Prompt history (best first) ---\n"
        f"{history_section}\n\n"
        f"Generate {n_candidates} new candidate prompts for detecting {class_name!r} in farm images.\n"
        f"They must be different from every prompt in the history above.\n"
        f"Return a JSON array of exactly {n_candidates} strings."
    )


def _format_history(history: list[dict]) -> str:
    if not history:
        return "  (no prompts tested yet — this is the first iteration)"
    rows = []
    for rank, entry in enumerate(history, 1):
        marker = " ← current best" if rank == 1 else ""
        rows.append(f"  #{rank:>3}  mAP={entry['map_score']:.4f}  {entry['prompt']!r}{marker}")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_prompt_list(text: str, n_candidates: int) -> list[str]:
    """Parse a JSON array of strings from the LLM response."""
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()

    try:
        result = json.loads(text)
        if isinstance(result, list) and all(isinstance(s, str) for s in result):
            return [s.strip() for s in result if s.strip()]
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[([^\[\]]+)\]", text, re.DOTALL)
    if match:
        try:
            result = json.loads(f"[{match.group(1)}]")
            if isinstance(result, list):
                return [str(s).strip() for s in result if str(s).strip()]
        except json.JSONDecodeError:
            pass

    candidates = re.findall(r'"([^"]{3,})"', text)
    if candidates:
        return [c.strip() for c in candidates if c.strip()]

    lines = [ln.strip().lstrip("-•*0123456789.) ") for ln in text.splitlines()]
    return [ln for ln in lines if 3 <= len(ln) <= 200]


# ---------------------------------------------------------------------------
# Heuristic fallback prompts
# ---------------------------------------------------------------------------

_FALLBACK_TEMPLATES = [
    "{class_name}",
    "a {crop} {class_name}",
    "close-up of {class_name}",
    "ripe {class_name} on plant",
    "single {class_name} visible in farm image",
    "overhead view of {class_name}",
    "green {class_name} in field",
]


def _heuristic_prompts(
    crop: str,
    class_name: str,
    n: int,
    history: list[dict],
) -> list[str]:
    seen = {h["prompt"] for h in history}
    candidates: list[str] = []
    for tmpl in _FALLBACK_TEMPLATES:
        p = tmpl.format(crop=crop, class_name=class_name)
        if p not in seen:
            candidates.append(p)
        if len(candidates) >= n:
            break
    while len(candidates) < n:
        candidates.append(f"{class_name} in agricultural setting #{len(candidates)}")
    return candidates[:n]
