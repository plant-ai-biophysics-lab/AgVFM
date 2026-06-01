"""YOLO-format disk dataset loader for the agvfm optimizer pipeline.

Provides :func:`load_disk_dataset` which reads image + ``.txt`` label files
from disk and returns a :class:`~agvfm.optimizer.types.DatasetSplit` compatible
with the proxy-image optimizer loops.

Label format (YOLO normalized)::

    <class_id> <x_center> <y_center> <width> <height>

All coordinates are normalized to [0, 1].  The resulting
:class:`~agvfm.optimizer.types.Sample` annotations use
``{"bbox": [x_topleft, y_topleft, w, h], "category_id": int, "category_name": str}``
with values in [0, 1].

Also supports COCO ``.json`` annotation files via ``convert_coco`` from
ultralytics (auto-conversion).

Usage
-----
    from agvfm.data.disk_loader import load_disk_dataset

    split = load_disk_dataset(
        name="cowpea_flower",
        classes=["flower"],
        images_dir=Path("/data/cowpea/images"),
        labels_dir=Path("/data/cowpea/labels"),
        train_split=0.8,
        seed=42,
    )
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any

from PIL import Image

from agvfm.optimizer.types import DatasetSplit, Sample

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ---------------------------------------------------------------------------
# Lazy disk wrapper
# ---------------------------------------------------------------------------

class _LazyDiskData:
    """Lazy wrapper over a list of (image_path, label_path) pairs."""

    def __init__(
        self,
        image_paths: list[Path],
        labels_dir: Path,
        classes: list[str],
    ) -> None:
        self._image_paths = image_paths
        self._labels_dir = labels_dir
        self._classes = classes

    def __len__(self) -> int:
        return len(self._image_paths)

    def __iter__(self):
        return iter(self._load_all())

    def __getitem__(self, idx):
        return self._load_all()[idx]

    def load_n(self, n: int) -> list[Sample]:
        """Load the first *n* samples sequentially."""
        return [
            _load_sample(p, self._labels_dir, self._classes)
            for p in self._image_paths[:n]
        ]

    def _load_all(self) -> list[Sample]:
        return [
            _load_sample(p, self._labels_dir, self._classes)
            for p in self._image_paths
        ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_disk_dataset(
    name: str,
    classes: list[str],
    images_dir: Path | str,
    labels_dir: Path | str,
    train_split: float = 0.8,
    seed: int = 42,
) -> DatasetSplit:
    """Load a YOLO-format on-disk dataset as a :class:`~agvfm.optimizer.types.DatasetSplit`.

    Parameters
    ----------
    name:
        Human-readable dataset name used in logging and result file names.
    classes:
        Ordered list of class names; index 0 = class_id 0 in label files.
    images_dir:
        Directory containing image files.
    labels_dir:
        Directory containing YOLO ``.txt`` label files (same stem as images).
        If it contains a COCO ``.json`` file, auto-converts to YOLO format first.
    train_split:
        Fraction of images used for the train (proxy) split.
    seed:
        Random seed for reproducible shuffling.
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    labels_dir = _convert_coco_if_needed(labels_dir)

    all_paths = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in _IMAGE_EXTENSIONS
    )
    if not all_paths:
        raise FileNotFoundError(
            f"No images found in {images_dir} "
            f"(expected extensions: {sorted(_IMAGE_EXTENSIONS)})"
        )

    rng = random.Random(seed)
    shuffled = list(all_paths)
    rng.shuffle(shuffled)

    n_train = max(1, int(len(shuffled) * train_split))
    train_paths = shuffled[:n_train]
    test_paths = shuffled[n_train:]

    return DatasetSplit(
        name=name,
        classes=classes,
        train=_LazyDiskData(train_paths, labels_dir, classes),
        test=_LazyDiskData(test_paths, labels_dir, classes),
    )


# ---------------------------------------------------------------------------
# COCO → YOLO conversion
# ---------------------------------------------------------------------------

def _convert_coco_if_needed(lbl_dir: Path) -> Path:
    """If *lbl_dir* contains a COCO .json file, convert to YOLO .txt format."""
    json_files = list(lbl_dir.glob("*.json"))
    if not json_files:
        return lbl_dir

    ann_file = json_files[0]
    output_dir = lbl_dir / "coco_converted" / "labels" / ann_file.stem
    if output_dir.exists() and any(output_dir.glob("*.txt")):
        logger.info(f"Converted COCO labels already exist: {output_dir}")
        return output_dir

    logger.info(f"Converting COCO → YOLO: {ann_file}")
    from ultralytics.data.converter import convert_coco
    orig_cwd = os.getcwd()
    try:
        os.chdir(str(lbl_dir))
        convert_coco(labels_dir=str(lbl_dir), use_segments=False, use_keypoints=False)
    finally:
        os.chdir(orig_cwd)

    if not output_dir.exists():
        raise RuntimeError(
            f"convert_coco did not produce expected directory: {output_dir}"
        )
    return output_dir


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_sample(image_path: Path, labels_dir: Path, classes: list[str]) -> Sample:
    image = Image.open(image_path).convert("RGB")
    label_path = labels_dir / (image_path.stem + ".txt")
    return Sample(image=image, annotations=_parse_yolo_label(label_path, classes))


def _parse_yolo_label(label_path: Path, classes: list[str]) -> list[dict]:
    """Parse a YOLO .txt label into normalized [x_topleft, y_topleft, w, h] annotations."""
    annotations: list[dict] = []
    if not label_path.exists():
        return annotations
    with open(label_path) as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])
            x = cx - bw / 2
            y = cy - bh / 2
            class_name = classes[class_id] if class_id < len(classes) else str(class_id)
            annotations.append({
                "bbox": [x, y, bw, bh],
                "category_id": class_id,
                "category_name": class_name,
            })
    return annotations
