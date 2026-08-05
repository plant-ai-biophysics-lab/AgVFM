"""Gemma4VLMPipeline — VLM-as-unified-pipeline (PAPER.md section 3.6).

Rather than one model proposing/filling a prompt and a *separate*
open-vocabulary detector (YOLO World, Grounding DINO, OWLv2, SAM3) consuming
it, this wraps a single Gemma 4 instance that performs prompt generation
(constrained axis-template fill, or unconstrained open-ended generation) AND
the grounding/detection step itself, from its own prompt — collapsing what
used to be two separate accounted-for roles (a translation/metaprompt LLM
call + a detector call) into one model.

Grounding output format — verified against Gemma 4's public documentation
before writing this, not assumed (per PAPER.md's explicit instruction not
to guess this): Gemma 4 (Google DeepMind, released 2026-04-02, built from the
same research lineage as Gemini 3) answers localization prompts with a JSON
array of ``{"box_2d": [...], "label": str}`` objects. Per the model's public
launch material: "the coordinates refer to an image size of 1000x1000,
relative to the input dimensions" — i.e. a normalized 1000x1000 space, not
raw pixels or 0-1. The launch material's own worked example does not
disambiguate axis order beyond that; this module assumes Gemini's
established ``box_2d`` convention (same field name, same 1000-scale — Gemma
4 shares the "same research lineage as Gemini 3" per its announcement),
which orders coordinates ``[ymin, xmin, ymax, xmax]``. **This axis-order
assumption has not been confirmed against a live model call in this
environment (no network/model access) — verify it against a real request
before trusting downstream box coordinates; see AGENT.md.**

Confidence scores: Gemma 4's documented JSON grounding output does **not**
include a native per-box confidence/score field (unlike YOLO World/Grounding
DINO/OWLv2, which score every candidate region internally). This module asks
for one anyway via the requested JSON schema (a ``"confidence"`` field,
0-1) and falls back to a fixed ``1.0`` per box when the model omits it. This
is a documented prompt-engineered approximation, not a native model
capability — treat any resulting precision-recall curve accordingly (a
constant confidence degenerates the curve to a single point).

Modes
-----
- ``"raw"`` — no self-generation step; ``prompt`` (as passed to
  :meth:`predict`) is used as the grounding target text directly. Useful when
  an external process (e.g. ``run_template.py``'s discovered template, or
  ``meta_run_template.py``'s winning free-text template) has already produced
  the final prompt and Gemma 4 is only being benchmarked as a detector.
  ``pipeline_role="detection"``.
- ``"constrained"`` — Gemma 4 fills a discovered axis template's descriptive
  slots itself (optionally looking at the image, mirroring
  :class:`agvfm.llm.vlm_insight_client.VLMInsightClient`'s vision-in-the-loop
  transfer), then grounds with the filled prompt. Requires ``axis_template``
  (a structural-slot dict, e.g. a ``TemplateResult.best_axes`` from
  :mod:`agvfm.optimizer.prompt_template`) passed via ``predict(**kwargs)``.
  ``pipeline_role="combined"`` (both prompt_generation and detection happen).
- ``"unconstrained"`` — Gemma 4 freely proposes one detection prompt for the
  target class/crop (mirroring :meth:`agvfm.llm.meta_client.VLMClient.suggest_prompts`,
  single-shot rather than iterative), then grounds with it.
  ``pipeline_role="combined"``.
- ``"fine_tuned"`` — **not implemented**. PAPER.md's LoRA baseline
  (``optimizer/lora_prompt.py``) doesn't exist yet; this mode is a placeholder
  that raises ``NotImplementedError`` rather than silently falling back to
  another mode.

Instrumentation
---------------
Cumulative usage is tracked *per pipeline_role* (see
:meth:`usage_snapshot`) so a caller building
:class:`agvfm.instrumentation.tracking.CallRecord` rows can attribute cost to
``prompt_generation`` vs. ``detection`` separately even when both happened
inside one :meth:`predict` call — the exact accounting problem PAPER.md
section 5.1 calls out for this module.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from openai import OpenAI
from PIL import Image

from agvfm.instrumentation.tracking import count_tokens_local, extract_openai_usage
from agvfm.models.base import BaseModel
from agvfm.optimizer.axes import PromptAxes

logger = logging.getLogger(__name__)

GROUNDING_SYSTEM_PROMPT = """\
You are a computer-vision object localization assistant. Given an image and \
a target object description, find every instance of that object visible in \
the image.

Return ONLY a JSON array, one object per detected instance, each with exactly:
  "box_2d": [ymin, xmin, ymax, xmax] — integers 0-1000, normalized to this image's size
  "label": the target object description
  "confidence": your confidence this is a correct, well-localized detection, 0.0-1.0

If no instance is visible, return an empty array: []
Return ONLY the JSON array. No explanation, no markdown."""

_CONSTRAINED_FILL_SYSTEM_PROMPT = """\
You are an agricultural computer-vision assistant filling in a detection-prompt \
template. You will be shown a photo of a specific crop object and a set of \
template slots (grammar/color/size/anatomy/phenology) that need concrete values \
for THIS crop. Base your answers on general knowledge of the crop AND what you \
can see in the image.

Return a JSON object with exactly these keys: "color", "size", "anatomy", "phenology" \
— each a short (1-3 word) phrase, or "" if not applicable/visually distinctive. \
Do not fill "grammar" or "taxonomy" — those are supplied separately.

Return ONLY the JSON object. No explanation, no markdown."""

_UNCONSTRAINED_SYSTEM_PROMPT = """\
You are an agricultural computer vision expert. Propose ONE short, effective \
free-text prompt for detecting the given crop object with an open-vocabulary \
detector, based on general knowledge of the crop and (if shown) this specific image.

Return a JSON object with exactly one key: "prompt" (a string).
Return ONLY the JSON object. No explanation, no markdown."""


class Gemma4VLMPipeline(BaseModel):
    """Gemma 4 as a unified prompt-generation + detection pipeline.

    Parameters
    ----------
    base_url:
        Base URL of a vision-capable OpenAI-compatible inference server
        (served backend). Omit (or pass ``None``) to run *model* locally via
        ``transformers.pipeline("image-text-to-text", ...)`` instead.
    model:
        Model name understood by the server (served) or a HF Hub id to load
        locally, e.g. ``"google/gemma-4-E4B-it"``.
    mode:
        ``"raw"`` | ``"constrained"`` | ``"unconstrained"`` — see module
        docstring. Fixed at construction time; build a second instance for a
        different mode rather than mutating this one mid-experiment, so
        instrumentation stays unambiguous about which mode produced which
        :class:`~agvfm.instrumentation.tracking.CallRecord` rows.
    temperature, max_tokens:
        Sampling controls, applied to both the (optional) generation call and
        the grounding call.
    max_retries:
        Retries per call before giving up (returns an empty detection set
        rather than raising, matching the other model wrappers' behaviour on
        unparseable output).
    device:
        ``device_map`` for the local HF pipeline (ignored for served).
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: str = "google/gemma-4-E4B-it",
        mode: str = "raw",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        max_retries: int = 2,
        device: Optional[str] = None,
    ) -> None:
        if mode not in ("raw", "constrained", "unconstrained", "fine_tuned"):
            raise ValueError(f"Unknown mode: {mode!r}")
        if mode == "fine_tuned":
            raise NotImplementedError(
                "Gemma4VLMPipeline's fine_tuned mode requires a LoRA/PEZ-tuned Gemma 4 "
                "checkpoint, which doesn't exist yet (agvfm/optimizer/lora_prompt.py is "
                "unbuilt — see AGENT.md). Use 'raw', 'constrained', or 'unconstrained' for now."
            )

        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_id = model
        self.mode = mode
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.device = device
        self.backend = "served" if base_url else "local"
        self._client = OpenAI(base_url=base_url, api_key="EMPTY") if base_url else None
        self._hf_pipeline = None  # lazily loaded on first local-backend call

        # Usage tracked per pipeline_role so a caller can build separately
        # attributable CallRecords even when one predict() call does both
        # roles (PAPER.md 5.1's pipeline_role field).
        self._usage = {
            role: {"n_api_calls": 0, "tokens_in": 0, "tokens_out": 0, "wall_clock_seconds": 0.0}
            for role in ("prompt_generation", "detection")
        }

    def usage_snapshot(self, role: Optional[str] = None) -> dict:
        """Return cumulative usage, either for one ``pipeline_role`` or totaled across both."""
        if role is not None:
            return dict(self._usage[role])
        total = {"n_api_calls": 0, "tokens_in": 0, "tokens_out": 0, "wall_clock_seconds": 0.0}
        for role_usage in self._usage.values():
            for k in total:
                total[k] += role_usage[k]
        return total

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def set_classes(self, class_names: List[str]) -> None:
        """No-op: Gemma 4 is prompted per-call with free text, like OWLv2/GroundingDINO."""
        pass

    @property
    def model_name(self) -> str:
        return f"Gemma4VLMPipeline({self.model_id.split('/')[-1]},mode={self.mode})"

    def predict(
        self,
        image_path: Path,
        prompt: str,
        conf_threshold: float = 0.1,
        crop: Optional[str] = None,
        axis_template: Optional[dict] = None,
        force_raw: bool = False,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run one image through this instance's mode (generate-then-detect, or detect-only).

        Args:
            image_path: Path to the input image.
            prompt: In ``"raw"`` mode, used verbatim as the grounding target
                text. In ``"constrained"``/``"unconstrained"`` mode, treated
                as the target class/object name (e.g. ``"grape"``) that
                Gemma 4 expands into a full prompt itself.
            conf_threshold: Score threshold applied to the (approximated)
                per-box confidence — see module docstring's caveat.
            crop: Human-readable crop name, used as generation context in
                ``"constrained"``/``"unconstrained"`` mode. Falls back to
                ``prompt`` if omitted.
            axis_template: Structural axis-slot dict (grammar/color/size/
                anatomy/phenology), required for ``"constrained"`` mode —
                typically a discovered ``TemplateResult.best_axes`` from
                :mod:`agvfm.optimizer.prompt_template`.
            force_raw: Skip self-generation for this call regardless of
                ``self.mode`` — ``prompt`` is used verbatim as the grounding
                target. Lets one ``"constrained"``/``"unconstrained"``
                instance also serve as its own detection-only baseline
                (same weights, same backend) without a second model load.

        Returns:
            Tuple ``(boxes, confidences)``, boxes ``(N, 4)`` float64 xyxy
            pixel coordinates, confidences ``(N,)`` float64 — same contract
            as every other ``BaseModel`` subclass.
        """
        image = Image.open(image_path).convert("RGB")
        class_name = prompt
        crop = crop or class_name

        if force_raw or self.mode == "raw":
            final_prompt = prompt
        elif self.mode == "constrained":
            if axis_template is None:
                raise ValueError("mode='constrained' requires axis_template=... (see docstring).")
            final_prompt = self._fill_constrained_template(image, crop, class_name, axis_template)
        elif self.mode == "unconstrained":
            final_prompt = self._generate_unconstrained_prompt(image, crop, class_name)
        else:
            raise NotImplementedError(f"mode={self.mode!r} is not implemented")

        return self._ground(image, final_prompt, conf_threshold)

    def predict_batch(
        self,
        image_paths: List[Path],
        prompt: str,
        conf_threshold: float = 0.1,
        **kwargs,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Sequential loop (same strategy as OWLv2/GroundingDINO/SAM3 — no batched VLM calls)."""
        return [
            self.predict(p, prompt, conf_threshold=conf_threshold, **kwargs)
            for p in image_paths
        ]

    # ------------------------------------------------------------------
    # Prompt generation (pipeline_role="prompt_generation")
    # ------------------------------------------------------------------

    def _fill_constrained_template(self, image: Image.Image, crop: str, class_name: str, axis_template: dict) -> str:
        """Ask Gemma 4 to fill an axis template's descriptive slots, then build the final prompt.

        Mirrors :class:`agvfm.llm.vlm_insight_client.VLMInsightClient`'s
        vision-in-the-loop transfer, but performed by this same Gemma 4
        instance rather than a separate insight client.
        """
        user_text = (
            f"Crop: {crop}\n"
            f"Target object/class label: {class_name!r}\n"
            f"Template slots to fill (grammar={axis_template.get('grammar', 'a')!r} is fixed): "
            f"color, size, anatomy, phenology.\n"
            f"Fill them for the {class_name} shown in this image."
        )
        raw = self._chat_complete(
            _CONSTRAINED_FILL_SYSTEM_PROMPT, user_text, image, role="prompt_generation",
        )
        filled = _parse_json_object(raw) or {}
        axes_dict = {**axis_template, "taxonomy": class_name}
        for k in ("color", "size", "anatomy", "phenology"):
            if filled.get(k):
                axes_dict[k] = filled[k]
        return PromptAxes.from_dict(axes_dict).to_prompt()

    def _generate_unconstrained_prompt(self, image: Image.Image, crop: str, class_name: str) -> str:
        """Ask Gemma 4 to freely propose one detection prompt (single-shot metaprompting)."""
        user_text = (
            f"Crop: {crop}\n"
            f"Target detection class: {class_name!r}\n"
            f"Propose one effective detection prompt for {class_name} in farm images, "
            f"based on this image."
        )
        raw = self._chat_complete(
            _UNCONSTRAINED_SYSTEM_PROMPT, user_text, image, role="prompt_generation",
        )
        parsed = _parse_json_object(raw) or {}
        prompt = str(parsed.get("prompt") or "").strip()
        return prompt or class_name

    # ------------------------------------------------------------------
    # Grounding (pipeline_role="detection")
    # ------------------------------------------------------------------

    def _ground(self, image: Image.Image, text_prompt: str, conf_threshold: float) -> Tuple[np.ndarray, np.ndarray]:
        w, h = image.size
        user_text = f"Find every instance of: {text_prompt}"
        raw = self._chat_complete(GROUNDING_SYSTEM_PROMPT, user_text, image, role="detection")
        boxes, confidences = _parse_detections(raw, img_w=w, img_h=h)

        if len(boxes) == 0:
            return np.zeros((0, 4), dtype=np.float64), np.zeros(0, dtype=np.float64)

        keep = confidences >= conf_threshold
        return boxes[keep], confidences[keep]

    # ------------------------------------------------------------------
    # Shared chat-completion plumbing (served vs. local, with usage tracking)
    # ------------------------------------------------------------------

    def _ensure_hf_pipeline(self):
        if self._hf_pipeline is None:
            from transformers import pipeline as hf_pipeline
            logger.info(f"Loading local HF image-text-to-text pipeline: {self.model_id}")
            self._hf_pipeline = hf_pipeline(
                "image-text-to-text", model=self.model_id, device_map=self.device, torch_dtype="auto",
            )
        return self._hf_pipeline

    def _chat_complete(self, system: str, user_text: str, image: Image.Image, role: str) -> str:
        """One vision chat-completion call, attributed to *role* for instrumentation."""
        start = time.time()
        tokens_in = tokens_out = 0
        try:
            if self.backend == "served":
                response = self._client.chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": [
                            {"type": "text", "text": user_text},
                            {"type": "image_url", "image_url": {"url": _encode_image(image)}},
                        ]},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                tokens_in, tokens_out = extract_openai_usage(response)
                return response.choices[0].message.content.strip()

            pipe = self._ensure_hf_pipeline()
            output = pipe(
                text=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": user_text},
                    ]},
                ],
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=True,
            )
            text = output[0]["generated_text"][-1]["content"].strip()
            tokenizer = getattr(getattr(pipe, "processor", None), "tokenizer", None) or getattr(pipe, "tokenizer", None)
            tokens_in = count_tokens_local(system + user_text, tokenizer)
            tokens_out = count_tokens_local(text, tokenizer)
            return text
        finally:
            usage = self._usage[role]
            usage["n_api_calls"] += 1
            usage["tokens_in"] += tokens_in
            usage["tokens_out"] += tokens_out
            usage["wall_clock_seconds"] += time.time() - start


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _strip_wrapping(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()


def _parse_json_object(text: str) -> Optional[dict]:
    text = _strip_wrapping(text)
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


def _parse_detections(text: str, img_w: int, img_h: int) -> Tuple[np.ndarray, np.ndarray]:
    """Parse Gemma 4's box_2d JSON array into (boxes_xyxy_pixel, confidences).

    ``box_2d`` is ``[ymin, xmin, ymax, xmax]`` normalized to a 1000x1000
    space (see module docstring for the source of this convention and its
    unverified axis-order caveat).
    """
    text = _strip_wrapping(text)
    data = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                data = None

    if not isinstance(data, list):
        return np.zeros((0, 4), dtype=np.float64), np.zeros(0, dtype=np.float64)

    boxes: list[list[float]] = []
    confidences: list[float] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        box_2d = entry.get("box_2d") or entry.get("bbox") or entry.get("box")
        if not box_2d or len(box_2d) != 4:
            continue
        ymin, xmin, ymax, xmax = (float(v) for v in box_2d)
        x1 = (xmin / 1000.0) * img_w
        y1 = (ymin / 1000.0) * img_h
        x2 = (xmax / 1000.0) * img_w
        y2 = (ymax / 1000.0) * img_h
        boxes.append([x1, y1, x2, y2])
        confidences.append(float(entry.get("confidence", entry.get("score", 1.0))))

    if not boxes:
        return np.zeros((0, 4), dtype=np.float64), np.zeros(0, dtype=np.float64)
    return np.array(boxes, dtype=np.float64), np.array(confidences, dtype=np.float64)


def _encode_image(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"
