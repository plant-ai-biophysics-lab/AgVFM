"""GroundingDINO model implementation using Hugging Face Transformers.

Wraps ``IDEA-Research/grounding-dino-base`` (or any compatible checkpoint) via
the HF ``AutoModelForZeroShotObjectDetection`` API so it satisfies the same
``BaseModel`` contract used by YOLOWorldModel and SAM3Model.

Prompt handling notes
---------------------
GroundingDINO tokenises each *phrase* as a separate query.  The model expects
phrases separated by ``. `` (period-space), and each phrase should end with a
period.  This wrapper normalises arbitrary prompt strings into that format:

    "a yellow cowpea flower" → "a yellow cowpea flower."
    "flower. bud."           → unchanged (already valid)

The ``box_threshold`` / ``text_threshold`` constructor parameters are the
primary detection thresholds used by the HF post-processor.  The
``conf_threshold`` passed by ``Evaluator.evaluate_prompt`` is applied as a
*secondary* post-hoc filter on the returned scores to keep the Evaluator
interface uniform across all model wrappers.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from agvfm.models.base import BaseModel


class GroundingDINOModel(BaseModel):
    """GroundingDINO zero-shot object-detection wrapper (HuggingFace Transformers)."""

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-base",
        device: Optional[str] = None,
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
        token: Optional[str] = None,
    ):
        """
        Initialise GroundingDINO from Hugging Face.

        Args:
            model_id:        HF model identifier.
            device:          Torch device string (default: cuda if available, else cpu).
            box_threshold:   Box-score threshold passed to the HF post-processor.
                             Detections below this are discarded before ``conf_threshold``
                             filtering; lower values yield more candidates.
            text_threshold:  Text-alignment threshold for the post-processor.
            token:           HF Hub token for private/gated models.  Falls back to
                             the ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` env vars.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_id = model_id
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

        try:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError:
            raise ImportError(
                "transformers >= 4.38 is required. "
                "Install with: pip install 'transformers>=4.38'"
            )

        # Load in float32; use autocast for float16 speed on CUDA.
        # This avoids all processor-output dtype mismatches (the processor
        # always returns float32 pixel tensors regardless of model dtype).
        self.use_autocast = device == "cuda"

        self.processor = AutoProcessor.from_pretrained(model_id, token=self.token)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id,
            token=self.token,
        ).to(device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Prompt normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_prompt(prompt: str) -> str:
        """Ensure the prompt ends with a period as GDino expects.

        GDino segments the input at `. ` boundaries.  We treat the entire
        prompt as a single phrase and simply append ``.`` if missing.
        """
        prompt = prompt.strip()
        if not prompt.endswith("."):
            prompt = prompt + "."
        return prompt

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def set_classes(self, class_names: List[str]) -> None:
        """No-op: GDino encodes classes from the text prompt at inference time."""
        pass

    @property
    def model_name(self) -> str:
        return f"GroundingDINO({self.model_id.split('/')[-1]})"

    def predict(
        self,
        image_path: Path,
        prompt: str,
        conf_threshold: float = 0.1,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run inference on a single image.

        Args:
            image_path:     Path to the input image.
            prompt:         Free-form text description of the target object(s).
            conf_threshold: Post-hoc confidence threshold applied *after* the HF
                            post-processor's own ``box_threshold`` filtering.
            **kwargs:       Unused; kept for interface compatibility.

        Returns:
            Tuple ``(boxes, confidences)`` where *boxes* is ``(N, 4)`` float64
            in **xyxy** pixel coordinates and *confidences* is ``(N,)`` float64.
        """
        image = Image.open(image_path).convert("RGB")
        text = self._normalise_prompt(prompt)

        inputs = self.processor(
            images=image,
            text=text,
            return_tensors="pt",
        ).to(self.device)

        autocast_ctx = torch.autocast(self.device) if self.use_autocast else torch.autocast(self.device, enabled=False)
        with torch.no_grad(), autocast_ctx:
            outputs = self.model(**inputs)

        # post_process_grounded_object_detection signature in this version of
        # GroundingDinoProcessor uses `threshold` (not `box_threshold`) as the
        # box-score gate, plus a separate `text_threshold` for text alignment.
        target_sizes = torch.tensor([image.size[::-1]], device=self.device)  # (H, W)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=target_sizes,
        )

        r = results[0]  # single image
        boxes = r["boxes"].cpu().numpy().astype(np.float64)     # (N, 4) xyxy
        scores = r["scores"].cpu().numpy().astype(np.float64)   # (N,)

        if len(boxes) == 0:
            return np.zeros((0, 4), dtype=np.float64), np.zeros(0, dtype=np.float64)

        # Secondary conf_threshold filter (keeps Evaluator interface uniform)
        keep = scores >= conf_threshold
        return boxes[keep], scores[keep]

    def predict_batch(
        self,
        image_paths: List[Path],
        prompt: str,
        conf_threshold: float = 0.1,
        **kwargs,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Run inference on multiple images (sequential loop).

        True batching across variable-resolution images would require padding
        to the same size, which can degrade GDino's spatial attention.  We use
        a sequential loop (same strategy as SAM3Model) for correctness.

        Args:
            image_paths:    List of paths to input images.
            prompt:         Text prompt.
            conf_threshold: Confidence threshold applied per image.

        Returns:
            List of ``(boxes, confidences)`` tuples, one per image.
        """
        return [
            self.predict(p, prompt, conf_threshold=conf_threshold, **kwargs)
            for p in image_paths
        ]
