"""VLMInsightClient — visual attribute reads used to fill prompt templates.

Where :mod:`agvfm.llm.client` generates axis *candidate values* from a
crop/class name alone (text-only, no image), this client looks at an actual
sample image of a held-out (zero-shot) crop and asks a vision-capable model
what it *sees* — color, size, visible anatomy, growth stage. That reading is
used to fill a discovered prompt template's descriptive slots instead of (or
alongside) bare class/crop metadata, so a template such as
``"{grammar} {color} {taxonomy} {anatomy}"`` can be completed for a crop the
optimiser never trained on. See :mod:`agvfm.optimizer.prompt_template` and
:mod:`agvfm.optimizer.meta_prompt_template` for the zero-shot transfer callers.

Like :class:`agvfm.llm.client.LLMClient`, supports two backends: a served
OpenAI-compatible vision endpoint (``base_url`` given), or a vision-capable
model pulled from the HuggingFace Hub and run locally via
``transformers.pipeline("image-text-to-text", ...)`` (``base_url`` omitted).
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
import time

from openai import OpenAI
from PIL import Image

from agvfm.instrumentation.tracking import count_tokens_local, extract_openai_usage

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an agricultural computer-vision assistant. You will be shown a photo \
containing a specific crop object. Describe ONLY what you can see in THIS \
image (not general knowledge about the crop) using short, concrete visual \
attributes suitable for filling a detection-prompt template.

Return a JSON object with exactly these keys:
  "color": a 1-3 word color descriptor (e.g. "deep red", "yellowish green"), or "" if not visually distinctive
  "size": a 1-2 word size descriptor (e.g. "small", "large"), or ""
  "anatomy": a 1-3 word visible part/structure descriptor (e.g. "with stem", "compact head"), or ""
  "phenology": a 1-2 word growth-stage descriptor (e.g. "ripe", "unripe", "flowering"), or ""
  "phrase": a short (3-6 word) descriptive noun phrase for the object, e.g. "deep red ripe strawberry fruit"

Return ONLY the JSON object. No explanation, no markdown."""


class VLMInsightClient:
    """Queries a vision-capable model (served or local HF pipeline) for per-image visual attributes.

    Parameters
    ----------
    base_url:
        Base URL of a vision-capable OpenAI-compatible inference server.
        Omit (or pass ``None``) to use a local ``transformers``
        ``image-text-to-text`` pipeline instead, pulling *model* from the
        HuggingFace Hub and running it in-process (e.g.
        ``"Qwen/Qwen2-VL-7B-Instruct"``, ``"google/gemma-3-4b-it"``).
    model:
        Model name understood by the server (served backend) or a HF Hub
        model id to load locally (local backend).
    temperature, max_tokens:
        Sampling controls — kept low/tight since this is a short structured read.
    max_retries:
        Retries before falling back to an empty (metadata-only) reading.
    device:
        ``device_map`` passed to the local HF pipeline (ignored for the
        served backend).
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str = "Qwen/Qwen2-VL-7B-Instruct",
        temperature: float = 0.2,
        max_tokens: int = 300,
        max_retries: int = 2,
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

        # Cumulative usage across every describe() call this instance has
        # made — see agvfm.instrumentation.tracking; read via usage_snapshot().
        self.total_calls = 0
        self.total_tokens_in = 0
        self.total_tokens_out = 0
        self.total_wall_seconds = 0.0
        self.total_images = 0  # -> n_unlabeled_images_used

    def usage_snapshot(self) -> dict:
        """Return cumulative usage counters (for building a CallRecord around a phase of calls)."""
        return {
            "n_api_calls": self.total_calls,
            "tokens_in": self.total_tokens_in,
            "tokens_out": self.total_tokens_out,
            "wall_clock_seconds": self.total_wall_seconds,
            "n_unlabeled_images_used": self.total_images,
        }

    def _ensure_hf_pipeline(self):
        if self._hf_pipeline is None:
            from transformers import pipeline as hf_pipeline
            logger.info(f"Loading local HF image-text-to-text pipeline: {self.model}")
            self._hf_pipeline = hf_pipeline(
                "image-text-to-text", model=self.model, device_map=self.device, torch_dtype="auto",
            )
        return self._hf_pipeline

    def describe(self, image: Image.Image, crop: str, class_name: str) -> dict:
        """Return ``{"color", "size", "anatomy", "phenology", "phrase"}`` read from ``image``."""
        text = (
            f"Crop: {crop}\n"
            f"Target object/class label: {class_name!r}\n"
            f"Describe the {class_name} visible in this image."
        )
        self.total_images += 1

        for attempt in range(self.max_retries):
            start = time.time()
            tokens_in = tokens_out = 0
            try:
                if self.backend == "served":
                    user_content = [
                        {"type": "text", "text": text},
                        {"type": "image_url", "image_url": {"url": _encode_image(image)}},
                    ]
                    response = self._client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_content},
                        ],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
                    tokens_in, tokens_out = extract_openai_usage(response)
                    raw = response.choices[0].message.content.strip()
                else:
                    pipe = self._ensure_hf_pipeline()
                    user_content = [
                        {"type": "image", "image": image},
                        {"type": "text", "text": text},
                    ]
                    output = pipe(
                        text=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_content},
                        ],
                        max_new_tokens=self.max_tokens,
                        temperature=self.temperature,
                        do_sample=True,
                    )
                    raw = output[0]["generated_text"][-1]["content"].strip()
                    tokenizer = getattr(getattr(pipe, "processor", None), "tokenizer", None) or getattr(pipe, "tokenizer", None)
                    tokens_in = count_tokens_local(SYSTEM_PROMPT + text, tokenizer)
                    tokens_out = count_tokens_local(raw, tokenizer)
                return _parse_insight(raw, class_name)
            except Exception as exc:
                logger.warning(
                    f"VLM insight request failed (attempt {attempt + 1}/{self.max_retries}): {exc}"
                )
            finally:
                self.total_calls += 1
                self.total_tokens_in += tokens_in
                self.total_tokens_out += tokens_out
                self.total_wall_seconds += time.time() - start

        logger.warning("Falling back to metadata-only insight after repeated failures.")
        return _fallback_insight(class_name)


def _encode_image(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _parse_insight(text: str, class_name: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()

    data = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                data = None

    if not isinstance(data, dict):
        return _fallback_insight(class_name)

    return {
        "color": str(data.get("color") or ""),
        "size": str(data.get("size") or ""),
        "anatomy": str(data.get("anatomy") or ""),
        "phenology": str(data.get("phenology") or ""),
        "phrase": str(data.get("phrase") or class_name),
    }


def _fallback_insight(class_name: str) -> dict:
    return {"color": "", "size": "", "anatomy": "", "phenology": "", "phrase": class_name}
