"""LLM client for prompt-axis value generation.

Provides :class:`LLMClient`, which generates candidate values for each
prompt axis used in the OFAT + combinatorial optimisation pipeline via
either of two backends:

- **Served** (``base_url`` given) — any OpenAI-compatible server (vLLM,
  Ollama, …). This is the lower-latency option once a server is running,
  and the only option that supports remote/shared inference servers.
- **Local HF pipeline** (``base_url`` omitted) — pulls ``model`` from the
  HuggingFace Hub and runs it in-process via ``transformers.pipeline``, the
  same approach ``load_and_run.py`` used for its Qwen/Qwen3-4B axis
  translation before ``LLMClient`` existed. No server to stand up first, at
  the cost of loading the model fresh (and holding it in memory) for the
  lifetime of the client.

Both backends are used identically via :meth:`LLMClient.generate_axis_values`
— pick one at construction time.
"""

from __future__ import annotations

import json
import logging
import re
import time

from openai import OpenAI

from agvfm.instrumentation.tracking import count_tokens_local, extract_openai_usage

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

# ---------------------------------------------------------------------------
# Cowpea-flower-baseline translation (mirrors load_and_run.py's original
# Qwen-based _query_llm_for_axes prompt, ported onto LLMClient's dual
# served/local backend — see LLMClient.translate_factor_axes)
# ---------------------------------------------------------------------------

_TRANSLATE_SYSTEM_PROMPT = """\
You are a botanical prompt engineering assistant.
Your task is to translate a list of FACTOR_AXES originally written for detecting \
cowpea flowers into equivalent axes for a different crop and object.

RULES:
- RETURN ONLY A VALID JSON ARRAY. ABSOLUTELY NO EXPLANATION OR CHAIN OF THOUGHT.
- Each element must be an object with exactly three keys: "name", "values", "baseline".
- Preserve the same axis names (taxonomy, color, size, phenology, negation, anatomy, grammar, emoji).
- Update the values and baseline to match the new crop/object.
- Keep grammar axis unchanged. Follow current standards for baselines.
- For the negation axis, list the most common classes that a detector might confuse with the target.
- For emoji axis, try to provide a set of 3-5 relevant emojis for the target.
"""

_TRANSLATE_USER_TEMPLATE = """\
Target crop/object: {crop}

Original FACTOR_AXES (cowpea flower):
{axes_json}

Translate the values and baseline for each axis to suit the target crop/object above.
Return ONLY the JSON array.
"""

# ---------------------------------------------------------------------------
# Zero-shot class-name translation (single value, no candidate list — see
# LLMClient.translate_class_name)
# ---------------------------------------------------------------------------

_CLASS_NAME_TRANSLATE_SYSTEM_PROMPT = """\
You are an agricultural computer vision expert. Given a crop and a detection \
class label, produce ONE short natural-language noun phrase suitable as the \
subject of a zero-shot object-detection prompt (e.g. "strawberry flower", \
"ripe strawberry", "romaine lettuce head", "almond blossom").

Return ONLY the phrase — no explanation, no quotes, no surrounding \
punctuation, no chain-of-thought."""


class LLMClient:
    """LLM client for axis value generation (served or local HF pipeline).

    Parameters
    ----------
    base_url:
        Base URL of an OpenAI-compatible inference server. Omit (or pass
        ``None``) to use a local ``transformers`` pipeline instead — pulling
        *model* from the HuggingFace Hub and running it in-process.
    model:
        Model name understood by the server (served backend) or a HF Hub
        model id to load locally (local backend), e.g. ``"Qwen/Qwen3-4B"``.
    temperature:
        Sampling temperature.
    max_tokens:
        Maximum tokens in the response.
    max_retries:
        Parse/request retries before falling back to default values.
    device:
        ``device_map`` passed to the local HF pipeline (ignored for the
        served backend).
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str = "Qwen/Qwen3-4B",
        temperature: float = 0.7,
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
        # made (successful or not — a failed call still spent time/tokens).
        # See agvfm.instrumentation.tracking; read via usage_snapshot().
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
            logger.info(f"Loading local HF pipeline for axis generation: {self.model}")
            self._hf_pipeline = hf_pipeline(
                "text-generation", model=self.model, device_map=self.device, torch_dtype="auto",
            )
        return self._hf_pipeline

    def _chat_complete(self, system: str, user: str) -> str:
        """Send one chat completion and return the raw response text, via whichever backend is active.

        Updates the cumulative usage counters (see :meth:`usage_snapshot`)
        regardless of backend or outcome — a raised exception still counts
        the elapsed time, since instrumentation should reflect what was
        actually spent, not just what succeeded.
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
                raw = self._chat_complete(system, user_message)
                values = _parse_json(raw)
                return _validate_axis_values(values)
            except (ValueError, KeyError) as e:
                logger.warning(
                    f"LLM axis generation failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
            except Exception as e:
                # Broad on purpose: the local HF backend can fail in ways a served
                # OpenAI-compatible call never would (OOM, missing weights, ...).
                logger.warning(
                    f"LLM request failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )

        logger.warning("Falling back to default axis values after repeated failures")
        return _default_axis_values(crop)

    def translate_factor_axes(self, crop: str) -> list[dict]:
        """Translate the cowpea-flower FACTOR_AXES baseline into axes for *crop*.

        Mirrors ``load_and_run.py``'s original Qwen-based axis translation
        (``_query_llm_for_axes``) — gives the LLM the cowpea-flower
        ``FACTOR_AXES`` as a concrete worked example and asks it to produce
        an analogous axis set for a new crop/object, rather than generating
        values cold the way :meth:`generate_axis_values` does. Returns a
        list of ``{"name", "values", "baseline"}`` dicts — the same schema
        ``agvfm.config.experiments.FactorAxis`` uses — so results can be
        persisted/reused the same way ``load_and_run.py``'s
        ``factor_axes.json`` already does (see
        :func:`agvfm.optimizer.prompt_template.generate_axis_values_per_crop`).

        Falls back to the literal, untranslated cowpea-flower baseline after
        repeated parse/request failures, so callers always get a usable set.
        """
        from agvfm.config.experiments import FACTOR_AXES

        baseline_axes = [
            {"name": ax.name, "values": list(ax.values), "baseline": ax.baseline}
            for ax in FACTOR_AXES
        ]
        user_message = _TRANSLATE_USER_TEMPLATE.format(
            crop=crop, axes_json=json.dumps(baseline_axes, indent=2)
        )

        for attempt in range(self.max_retries):
            try:
                raw = self._chat_complete(_TRANSLATE_SYSTEM_PROMPT, user_message)
                data = _parse_json_array(raw)
                _validate_factor_axes(data)
                return _anchor_taxonomy_to_crop(data, crop)
            except (ValueError, KeyError) as e:
                logger.warning(
                    f"Axis translation for {crop!r} failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
            except Exception as e:
                logger.warning(
                    f"Axis translation request for {crop!r} failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )

        logger.warning(f"Falling back to literal cowpea-flower baseline for {crop!r} after repeated failures")
        return _anchor_taxonomy_to_crop(baseline_axes, crop)

    def translate_class_name(self, crop: str, class_name: str) -> str:
        """Translate a raw dataset class label into one natural detection-prompt phrase.

        Unlike :meth:`generate_axis_values`/:meth:`translate_factor_axes`
        (which each produce *multiple* candidate values per axis, for the
        OFAT/template *search* phase), this is the single-value, zero-shot
        counterpart used at *transfer* time: the only realistic inputs are
        the crop/class name a user would actually type (PAPER.md 3.2's
        "using only dataset metadata"), and exactly one output is needed —
        not a list to search over. Used to fill a discovered template's
        taxonomy/``{class}`` slot without touching the template's other
        (structural) values, so zero-shot evaluation genuinely tests whether
        the discovered template transfers as-is rather than silently
        re-deriving crop-specific values again for every new crop.

        Falls back to *class_name* itself (untranslated) after repeated
        failures, so callers always get a usable string.
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
            except Exception as e:
                logger.warning(
                    f"Class-name translation for {crop!r}/{class_name!r} failed "
                    f"(attempt {attempt + 1}/{self.max_retries}): {e}"
                )

        logger.warning(f"Falling back to raw class name for {crop!r}/{class_name!r} after repeated failures")
        return class_name


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
    # Reasoning models (e.g. Qwen3 without a no-think directive) may emit a
    # <think>...</think> block before the actual answer — strip it so it
    # doesn't get mistaken for (or corrupt) the JSON payload below.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
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


def _clean_phrase(text: str) -> str:
    """Extract a single clean phrase from an LLM response (used by :meth:`LLMClient.translate_class_name`)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    for line in text.splitlines():
        line = line.strip().strip('"').strip("'").rstrip(".").strip()
        if line:
            return line
    return ""


def _parse_json_array(text: str) -> list:
    """Parse a JSON array from an LLM response (used by :meth:`LLMClient.translate_factor_axes`)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()

    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Could not locate a JSON array in LLM response: {text[:200]!r}")
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON array parse error: {e}") from e


def _anchor_taxonomy_to_crop(data: list[dict], crop: str) -> list[dict]:
    """Force the taxonomy axis's baseline to *crop* itself rather than trusting the LLM's translation.

    The cowpea-flower worked example's taxonomy baseline is ``"flower"`` —
    sensible there because flower genuinely is the detection target. Left
    unguided, a small local model tends to mimic that shape literally (e.g.
    translating to ``"blossom"`` for apple) instead of anchoring on the
    actual object of interest, which is already known here (it's *crop*)
    and doesn't need to be invented. The LLM's alternate phrasings are kept
    as additional candidate ``values`` (still potentially useful context),
    just not as the baseline, and *crop* is guaranteed to be present in
    ``values`` even if the model didn't propose it verbatim.
    """
    for entry in data:
        if entry.get("name") == "taxonomy":
            entry["baseline"] = crop
            values = [v for v in entry.get("values", []) if v != crop]
            entry["values"] = [crop] + values
    return data


def _validate_factor_axes(data) -> None:
    """Raise ``ValueError`` if *data* isn't a well-formed list of FactorAxis-shaped dicts."""
    if not isinstance(data, list) or not data:
        raise ValueError("Expected a non-empty JSON array of axis objects")
    required_keys = {"name", "values", "baseline"}
    for item in data:
        if not isinstance(item, dict) or not required_keys.issubset(item.keys()):
            raise ValueError(f"Malformed axis entry: {item!r}")
        if not isinstance(item["values"], list):
            raise ValueError(f"'values' is not a list for axis {item.get('name')!r}")
