"""Shared data types and base class for the agvfm optimizer sub-package.

Defines the VFMBase interface (PIL-image in / Detection list out) used by the
gradient and meta-prompt optimizers, and AgVFMAdapter which bridges agvfm's
native models (file-path in / numpy out) to that interface.

Sample / DatasetSplit are the lightweight data containers used by the proxy-image
evaluation loop inside each optimizer.
"""

from __future__ import annotations

import logging
import os
import random
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Detection and data containers
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    bbox: list[float]   # [x1, y1, x2, y2] normalized 0-1
    score: float
    class_id: int
    class_name: str


@dataclass
class Sample:
    image: Image.Image
    annotations: list[dict]  # [{"bbox": [x, y, w, h] normalized, "category_id": int, "category_name": str}]


@dataclass
class DatasetSplit:
    name: str
    classes: list[str]
    train: Any   # _LazyAgMLData | _LazyDiskData | list[Sample]
    test: Any


# ---------------------------------------------------------------------------
# VFMBase — abstract interface used by optimizer loops
# ---------------------------------------------------------------------------

class VFMBase(ABC):
    """Abstract base for detection models used by the optimizer.

    Subclasses implement :meth:`predict` which takes a PIL image and a list of
    text prompts and returns detected objects.  :meth:`compute_map` (F1 proxy)
    is implemented here using :meth:`predict`.
    """

    name: str = ""

    @abstractmethod
    def predict(self, image: Image.Image, prompts: list[str]) -> list[Detection]:
        """Run inference on *image* with one prompt per class."""

    def compute_map(
        self,
        samples: list[Sample],
        prompts: list[str],
        iou_threshold: float = 0.5,
    ) -> float:
        """F1 score computed at *iou_threshold* across all *samples*.

        Used as a fast proxy for mAP inside the optimizer loops.
        """
        all_tp = all_fp = all_fn = 0

        for sample in samples:
            preds = self.predict(sample.image, prompts)
            tp, fp, fn = _match_detections(preds, sample.annotations, iou_threshold)
            all_tp += tp
            all_fp += fp
            all_fn += fn

        precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
        recall    = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# AgVFMAdapter — bridges agvfm native models to VFMBase
# ---------------------------------------------------------------------------

class AgVFMAdapter(VFMBase):
    """Wrap any agvfm BaseModel (file-path / numpy interface) as a VFMBase.

    agvfm models expect a file path and return numpy pixel-coordinate boxes.
    This adapter saves the PIL image to a temp .jpg, calls the agvfm model,
    and converts the result to normalized Detection objects.

    Parameters
    ----------
    agvfm_model:
        An instantiated agvfm model (YOLOWorldModel, GroundingDINOModel, …).
    name:
        Short identifier used in logging and output file names.
    conf_threshold:
        Forwarded to the agvfm model's ``predict()`` call.
    """

    def __init__(self, agvfm_model, name: str, conf_threshold: float = 0.1) -> None:
        self._agvfm_model = agvfm_model
        self.name = name
        self._conf_threshold = conf_threshold

    def predict(self, image: Image.Image, prompts: list[str]) -> list[Detection]:
        w, h = image.size
        detections: list[Detection] = []

        tmp_fd, tmp_name = tempfile.mkstemp(suffix=".jpg")
        os.close(tmp_fd)
        tmp_path = Path(tmp_name)
        try:
            image.save(tmp_path, format="JPEG", quality=95)
            for class_id, prompt in enumerate(prompts):
                boxes, confs = self._agvfm_model.predict(
                    tmp_path, prompt, conf_threshold=self._conf_threshold
                )
                for box, conf in zip(boxes, confs):
                    x1, y1, x2, y2 = box
                    detections.append(Detection(
                        bbox=[x1 / w, y1 / h, x2 / w, y2 / h],
                        score=float(conf),
                        class_id=class_id,
                        class_name=prompt,
                    ))
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        return detections


# ---------------------------------------------------------------------------
# Proxy image sampling
# ---------------------------------------------------------------------------

def sample_proxy_images(split: DatasetSplit, n: int, seed: int = 42) -> list[Sample]:
    """Return up to *n* proxy images from ``split.train``.

    Works with ``_LazyAgMLData``, ``_LazyDiskData``, and plain ``list[Sample]``.
    """
    train = split.train
    if hasattr(train, "load_n"):
        return train.load_n(n)
    rng = random.Random(seed)
    return rng.sample(list(train), min(n, len(train)))


# ---------------------------------------------------------------------------
# Detection matching helpers
# ---------------------------------------------------------------------------

def _iou(box_a: list[float], box_b: list[float]) -> float:
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _xywh_to_xyxy(bbox: list[float]) -> list[float]:
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def _match_detections(
    preds: list[Detection],
    gt: list[dict],
    iou_threshold: float,
) -> tuple[int, int, int]:
    matched_gt: set[int] = set()
    tp = 0
    for pred in sorted(preds, key=lambda d: d.score, reverse=True):
        best_iou = 0.0
        best_idx = -1
        for idx, ann in enumerate(gt):
            if idx in matched_gt:
                continue
            gt_box = _xywh_to_xyxy(ann["bbox"])
            iou = _iou(pred.bbox, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_iou >= iou_threshold and best_idx >= 0:
            tp += 1
            matched_gt.add(best_idx)
    fp = len(preds) - tp
    fn = len(gt) - tp
    return tp, max(fp, 0), max(fn, 0)
