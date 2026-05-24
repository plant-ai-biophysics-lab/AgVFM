"""OWLv2 model implementation using Hugging Face Transformers.

Wraps ``google/owlv2-base-patch16-ensemble`` (or any compatible OWLv2
checkpoint) via the HF ``Owlv2ForObjectDetection`` API so it satisfies
the same ``BaseModel`` contract used by YOLOWorldModel and SAM3Model.

Inference notes
---------------
OWLv2 is a vision–language model that takes a list of *text queries* for a
single image.  We pass the prompt as a single-element query list:

    processor(text=[[prompt]], images=image, ...)

``post_process_grounded_object_detection`` returns boxes already in **xyxy** pixel
coordinates, so no conversion is needed.

The ``conf_threshold`` argument passed by ``Evaluator.evaluate_prompt`` is
forwarded directly to the HF post-processor and also used as the return-level
filter, keeping the interface identical to all other wrappers.

Position-embedding limit
------------------------
The default ``Owlv2TextConfig`` sets ``max_position_embeddings = 16``.  Prompts
longer than ~12 tokens will be silently truncated by the tokenizer.  No config
patching is applied here; keep prompts short accordingly.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from agvfm.models.base import BaseModel


class OWLv2Model(BaseModel):
    """OWLv2 zero-shot object-detection wrapper (HuggingFace Transformers)."""

    def __init__(
        self,
        model_id: str = "google/owlv2-base-patch16-ensemble",
        device: Optional[str] = None,
        token: Optional[str] = None,
    ):
        """
        Initialise OWLv2 from Hugging Face.

        Args:
            model_id: HF model identifier.
            device:   Torch device string (default: cuda if available, else cpu).
            token:    HF Hub token for private/gated models.  Falls back to
                      the ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` env vars.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_id = model_id
        self.device = device
        self.token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

        try:
            from transformers import Owlv2ForObjectDetection, Owlv2Processor
        except ImportError:
            raise ImportError(
                "transformers >= 4.38 is required. "
                "Install with: pip install 'transformers>=4.38'"
            )

        # Load in float32; use autocast for float16 speed on CUDA.
        self.use_autocast = device == "cuda"

        self.processor = Owlv2Processor.from_pretrained(model_id, token=self.token)
        self.model = Owlv2ForObjectDetection.from_pretrained(
            model_id,
            token=self.token,
        ).to(device)
        self.model.eval()

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def set_classes(self, class_names: List[str]) -> None:
        """No-op: OWLv2 encodes classes from the text queries at inference time."""
        pass

    @property
    def model_name(self) -> str:
        return f"OWLv2({self.model_id.split('/')[-1]})"

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
            conf_threshold: Score threshold for ``post_process_grounded_object_detection``
                            and for the secondary keep-mask applied to the results.
            **kwargs:       Unused; kept for interface compatibility.

        Returns:
            Tuple ``(boxes, confidences)`` where *boxes* is ``(N, 4)`` float64
            in **xyxy** pixel coordinates and *confidences* is ``(N,)`` float64.
        """
        image = Image.open(image_path).convert("RGB")

        # OWLv2 expects a batch-of-query-lists: [[query1, query2, ...]]
        # We use a single query matching the full prompt string.
        inputs = self.processor(
            text=[[prompt]],
            images=image,
            return_tensors="pt",
        ).to(self.device)

        autocast_ctx = torch.autocast(self.device) if self.use_autocast else torch.autocast(self.device, enabled=False)
        with torch.no_grad(), autocast_ctx:
            outputs = self.model(**inputs)

        # post_process_grounded_object_detection lives on the PROCESSOR.
        # target_sizes: (H, W) — must be on CPU for this post-processor.
        target_sizes = torch.tensor([[image.height, image.width]])  # CPU (H, W)
        text_labels = [[prompt]]
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=conf_threshold,
            text_labels=text_labels,
        )

        r = results[0]  # single image
        boxes = r["boxes"].cpu().numpy().astype(np.float64)     # (N, 4) xyxy
        scores = r["scores"].cpu().numpy().astype(np.float64)   # (N,)

        if len(boxes) == 0:
            return np.zeros((0, 4), dtype=np.float64), np.zeros(0, dtype=np.float64)

        # Explicit keep-mask — post_process already filters, but this is a
        # defensive guard for floating-point edge cases near the threshold.
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

        OWLv2 images at different resolutions would require zero-padding to
        batch in a single forward pass, which can degrade patch-embedding
        quality.  We use a sequential loop (same strategy as SAM3 / GDino)
        for correctness.

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
