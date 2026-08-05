"""MetaPromptClient — text-only LLM client for meta-prompt generation.

Supports the same served-vs-local backend choice as :class:`agvfm.llm.client.LLMClient`
(OpenAI-compatible endpoint, e.g. vLLM, or a model pulled from the HuggingFace
Hub and run locally via ``transformers``) but sends full prompt strings
rather than axis values. The VLM/vision path has been removed; image context
is a possible future extension.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Optional

from openai import OpenAI

from agvfm.instrumentation.tracking import count_tokens_local, extract_openai_usage
from agvfm.llm.client import _CLASS_NAME_TRANSLATE_SYSTEM_PROMPT, _clean_phrase

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
    """Text-only LLM client for meta-prompt generation (served or local HF pipeline).

    Parameters
    ----------
    base_url:
        Base URL of an OpenAI-compatible inference server. Omit (or pass
        ``None``) to use a local ``transformers`` pipeline instead — pulling
        *model* from the HuggingFace Hub and running it in-process, the same
        approach ``load_and_run.py`` used for Qwen/Qwen3-4B before
        :class:`~agvfm.llm.client.LLMClient` existed.
    model:
        Model name understood by the server (served backend) or a HF Hub
        model id to load locally (local backend).
    temperature:
        Sampling temperature (higher → more diverse suggestions).
    max_tokens:
        Maximum tokens in the model's response.
    max_retries:
        Number of parse/request retries before falling back to heuristic prompts.
    device:
        ``device_map`` passed to the local HF pipeline (ignored for the
        served backend).
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: str = "Qwen/Qwen3-4B",
        temperature: float = 0.8,
        max_tokens: int = 512,
        max_retries: int = 3,
        device: str = "cuda",
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.device = device
        self.backend = "served" if base_url else "local"
        self._client = OpenAI(base_url=base_url, api_key="EMPTY") if base_url else None
        self._hf_pipeline = None  # lazily loaded on first local-backend call

        # Cumulative usage across every _chat_complete call this instance has
        # made — see agvfm.instrumentation.tracking; read via usage_snapshot().
        self.total_calls = 0
        self.total_tokens_in = 0
        self.total_tokens_out = 0
        self.total_wall_seconds = 0.0

    def usage_snapshot(self) -> dict:
        """Return cumulative usage counters (for building a CallRecord around a phase of calls)."""
        return {
            "n_api_calls": self.total_calls,
            "tokens_in": self.total_tokens_in,
            "tokens_out": self.total_tokens_out,
            "wall_clock_seconds": self.total_wall_seconds,
        }

    def _ensure_hf_pipeline(self):
        if self._hf_pipeline is None:
            from transformers import pipeline as hf_pipeline
            logger.info(f"Loading local HF pipeline for prompt generation: {self.model}")
            self._hf_pipeline = hf_pipeline(
                "text-generation", model=self.model, device_map=self.device, torch_dtype="auto",
            )
        return self._hf_pipeline

    def _chat_complete(self, system: str, user: str) -> str:
        """Send one chat completion and return the raw response text, via whichever backend is active.

        Updates the cumulative usage counters (see :meth:`usage_snapshot`)
        regardless of backend or outcome.
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        start = time.time()
        tokens_in = tokens_out = 0
        try:
            if self.backend == "served":
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                tokens_in, tokens_out = extract_openai_usage(response)
                return response.choices[0].message.content.strip()

            pipe = self._ensure_hf_pipeline()
            output = pipe(
                messages,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=True,
                return_full_text=False,
            )
            text = output[0]["generated_text"].strip()
            tokenizer = getattr(pipe, "tokenizer", None)
            tokens_in = count_tokens_local(system + user, tokenizer)
            tokens_out = count_tokens_local(text, tokenizer)
            return text
        finally:
            self.total_calls += 1
            self.total_tokens_in += tokens_in
            self.total_tokens_out += tokens_out
            self.total_wall_seconds += time.time() - start

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
                raw = self._chat_complete(system, user_content)
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

    def suggest_template_prompts(
        self,
        crops: list[str],
        history: list[dict],          # list of {"prompt": str, "map_score": float}
        n_candidates: int = 5,
        cross_run_summary: str = "",
        all_seen_templates: Optional[set] = None,
    ) -> list[str]:
        """Ask the LLM to propose ``n_candidates`` reusable prompt *templates*.

        Unlike :meth:`suggest_prompts` (one prompt per dataset/class), a template
        contains a literal ``"{class}"`` placeholder so one string can be scored
        against every (dataset, class) group in a pooled cross-dataset training
        mix — see :mod:`agvfm.optimizer.prompt_template` and
        :mod:`agvfm.optimizer.meta_prompt_template`. ``crops`` lists every crop
        represented in the pool so the LLM can propose slots that generalise
        across all of them rather than overfitting one crop's vocabulary.
        """
        system = TEMPLATE_SYSTEM_PROMPT.format(n_candidates=n_candidates)
        user_content = _build_template_user_content(
            crops=crops,
            history=history,
            n_candidates=n_candidates,
            cross_run_summary=cross_run_summary,
        )
        seen = set(all_seen_templates or ()) | {h["prompt"] for h in history}

        for attempt in range(self.max_retries):
            try:
                raw = self._chat_complete(system, user_content)
                candidates = _parse_prompt_list(raw, n_candidates)
                candidates = [c for c in candidates if "{class}" in c and c not in seen]
                if candidates:
                    logger.debug(f"LLM suggested {len(candidates)} new templates (attempt {attempt + 1})")
                    return candidates[:n_candidates]
                logger.warning(
                    f"All LLM template suggestions were invalid or already seen "
                    f"(attempt {attempt + 1}/{self.max_retries}); retrying."
                )
            except Exception as exc:
                logger.warning(f"LLM template request failed (attempt {attempt + 1}/{self.max_retries}): {exc}")

        logger.warning("Falling back to heuristic templates after repeated failures.")
        return [t for t in _FALLBACK_TEMPLATES_CLASS if t not in seen][:n_candidates] or ["{class}"]

    def translate_class_name(self, crop: str, class_name: str) -> str:
        """Translate a raw dataset class label into one natural detection-prompt phrase.

        Same purpose as :meth:`agvfm.llm.client.LLMClient.translate_class_name`
        (single realistic zero-shot input — crop + raw class label — single
        output, not a candidate list), duplicated here so
        ``meta_run_template.py``'s standalone ``VLMClient`` instance can also
        do zero-shot class-name translation without needing a second
        ``LLMClient`` alongside it.
        """
        user_message = (
            f"Crop: {crop}\n"
            f"Detection class label: {class_name!r}\n"
            f"Produce one short phrase naming this object for a zero-shot detection prompt."
        )
        for attempt in range(self.max_retries):
            try:
                raw = self._chat_complete(_CLASS_NAME_TRANSLATE_SYSTEM_PROMPT, user_message)
                phrase = _clean_phrase(raw)
                if phrase:
                    return phrase
            except Exception as exc:
                logger.warning(
                    f"Class-name translation for {crop!r}/{class_name!r} failed "
                    f"(attempt {attempt + 1}/{self.max_retries}): {exc}"
                )

        logger.warning(f"Falling back to raw class name for {crop!r}/{class_name!r} after repeated failures")
        return class_name


# ---------------------------------------------------------------------------
# Message building helpers
# ---------------------------------------------------------------------------

TEMPLATE_SYSTEM_PROMPT = """\
You are an agricultural computer vision expert specialising in zero-shot object detection.
Your task is to generate reusable text *templates* for detecting crop objects in farm images,
to be applied across many different crops/classes at once.

Each template MUST contain the literal placeholder "{{class}}" exactly once, marking where a
specific crop/class name will be substituted (e.g. "a ripe {{class}} on the plant"). Templates
are scored by substituting the real class name of every crop in a pooled cross-dataset mix and
aggregating detections across all of them, so favour structural phrasing (grammar, framing,
generic descriptive slots) that should generalise across crops rather than wording specific to
any single crop.

Based on the history provided, propose new templates that are meaningfully different from
previous ones and likely to yield better aggregate detection scores across the crop mix.

Return a JSON array of EXACTLY {n_candidates} template strings and nothing else.
Example: ["a {{class}}", "a ripe {{class}} on the plant", "close-up of a single {{class}}"]"""


def _build_template_user_content(
    crops: list[str],
    history: list[dict],
    n_candidates: int,
    cross_run_summary: str = "",
) -> str:
    text_intro = f"Crops represented in the pooled training mix: {', '.join(crops)}\n"
    summary_section = f"\n{cross_run_summary}\n" if cross_run_summary else ""
    history_section = _format_history(history)
    return (
        f"{text_intro}"
        f"{summary_section}"
        f"\n--- Template history (best first) ---\n"
        f"{history_section}\n\n"
        f"Generate {n_candidates} new candidate templates, each containing \"{{class}}\" exactly once.\n"
        f"They must be different from every template in the history above.\n"
        f"Return a JSON array of exactly {n_candidates} strings."
    )


_FALLBACK_TEMPLATES_CLASS = [
    "{class}",
    "a {class}",
    "a single {class}",
    "close-up of a {class}",
    "a ripe {class} on the plant",
    "{class} visible in farm image",
]

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
    # Strip a reasoning model's <think>...</think> block before parsing — see
    # agvfm.llm.client._parse_json for why.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
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
