"""OpenAI-compatible LLM client for prompt-axis value generation.

Provides :class:`LLMClient` which queries any OpenAI-compatible server
(vLLM, Ollama, …) to generate candidate values for each prompt axis used
in the OFAT + combinatorial optimisation pipeline.

Also used by ``load_and_run.py`` as an alternative to the local Qwen HF
pipeline when ``--llm-url`` and ``--llm-model`` are supplied.
"""

from __future__ import annotations

import json
import logging
import re

from openai import OpenAI

logger = logging.getLogger(__name__)

# Axis names that the LLM generates values for (taxonomy is fixed to the class name).
AXIS_NAMES = ["grammar", "color", "taxonomy", "anatomy", "phenology", "negation", "size", "emoji"]
AXIS_NAMES_FREE = [n for n in AXIS_NAMES if n != "taxonomy"]

SYSTEM_PROMPT = """\
You are an agricultural computer vision expert. Generate candidate text values for prompt axes used in zero-shot object detection.

Return a JSON object with exactly these 7 keys: grammar, color, size, anatomy, phenology, negation, emoji.
Each key maps to a list of EXACTLY 4 strings. No more, no less.

Guidelines:
- grammar: articles/determiners only, e.g. ["a", "a single", "close-up of a", ""]
- color: 3 visual colors relevant to the crop + ""
- size: 3 size descriptors relevant to the crop + ""
- anatomy: 3 physical part descriptors relevant to the crop + ""
- phenology: 3 growth stage descriptors relevant to the crop + ""
- negation: 3 confuser phrases to exclude (start with "not a " or "not ") + ""
- emoji: 3 relevant emoji characters + ""

Return ONLY the JSON object. No explanation. Do not repeat values."""


class LLMClient:
    """OpenAI-compatible LLM client for axis value generation.

    Parameters
    ----------
    base_url:
        Base URL of the OpenAI-compatible inference server.
    model:
        Model name understood by the server.
    temperature:
        Sampling temperature.
    max_tokens:
        Maximum tokens in the response.
    max_retries:
        Parse/request retries before falling back to default values.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        max_retries: int = 3,
    ) -> None:
        self._client = OpenAI(base_url=base_url, api_key="EMPTY")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

    def generate_axis_values(self, crop: str, taxonomy: str) -> dict[str, list[str]]:
        """Query the LLM to generate axis values for *taxonomy* within *crop*.

        Returns a dict mapping each free axis name to a list of candidate strings.
        Falls back to :func:`_default_axis_values` after repeated failures.
        """
        user_message = (
            f"Crop: {crop}\n"
            f"Detection class: {taxonomy}\n\n"
            f"Generate candidate values for each prompt axis to improve detection of "
            f"{taxonomy} in farm images. "
            f"Return JSON with keys: {', '.join(AXIS_NAMES_FREE)}."
        )
        system = (
            SYSTEM_PROMPT
            + f"\n\nYou are generating values for {crop} detection. "
            f"All values must be relevant to {crop}."
        )

        for attempt in range(self.max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                raw = response.choices[0].message.content.strip()
                values = _parse_json(raw)
                return _validate_axis_values(values)
            except (ValueError, KeyError) as e:
                logger.warning(
                    f"LLM axis generation failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )

        logger.warning("Falling back to default axis values after repeated failures")
        return _default_axis_values(crop)


MAX_VALUES_PER_AXIS = 6


def _validate_axis_values(values: dict) -> dict[str, list[str]]:
    result = {}
    for axis in AXIS_NAMES_FREE:
        vals = values.get(axis, [""])
        if not isinstance(vals, list):
            vals = [str(vals)]
        coerced = [str(v) if v is not None else "" for v in vals]
        seen: set[str] = set()
        deduped: list[str] = []
        for v in coerced:
            if v not in seen:
                seen.add(v)
                deduped.append(v)
        deduped = deduped[:MAX_VALUES_PER_AXIS]
        if "" not in deduped:
            deduped.append("")
        result[axis] = deduped
    return result


def _default_axis_values(crop: str) -> dict[str, list[str]]:
    """Minimal fallback axis values when the LLM is unavailable."""
    return {
        "grammar": ["a", "a single", "close-up of a", ""],
        "color": ["", "red", "green", "yellow"],
        "size": ["", "large", "small"],
        "anatomy": ["", "whole", "with stem"],
        "phenology": ["", "ripe", "unripe"],
        "negation": ["", "not a leaf", "not background"],
        "emoji": [""],
    }


def _parse_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Partial recovery: extract each complete "axis": [...] pair individually
    axes_pattern = "|".join(AXIS_NAMES_FREE)
    partial: dict[str, list[str]] = {}
    for m in re.finditer(rf'"({axes_pattern})"\s*:\s*\[([^\]]+)\]', text):
        values = re.findall(r'"([^"]*)"', m.group(2))
        if values:
            partial[m.group(1)] = values
    if partial:
        return partial

    raise ValueError(f"Could not parse JSON from LLM response: {text!r}")
