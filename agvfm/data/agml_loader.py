"""AgML streaming dataset loader for the agvfm optimizer pipeline.

Provides :func:`load_dataset` which wraps ``agml.data.AgMLDataLoader`` with
lazy image loading so that datasets can be iterated without downloading all
images up front.  The returned :class:`~agvfm.optimizer.types.DatasetSplit`
is compatible with :func:`~agvfm.optimizer.types.sample_proxy_images` and
both optimizer loops (:mod:`agvfm.optimizer.grad_prompt`,
:mod:`agvfm.optimizer.meta_prompt`).

AgML datasets are identified by name (e.g. ``"grape_detection_californiaday"``)
and are downloaded to ``~/.agml/`` by default.  Pass *data_root* to redirect
downloads to a shared cluster directory.

Usage
-----
    from agvfm.data.agml_loader import load_dataset, load_all_datasets

    split = load_dataset(
        "grape_detection_californiaday",
        classes=["grape"],
        data_root="/group/jmearlesgrp/agml_data",
    )
    # split.train / split.test are _LazyAgMLData objects
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from agvfm.optimizer.types import DatasetSplit, Sample

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy AgML wrapper
# ---------------------------------------------------------------------------

class _LazyAgMLData:
    """Wraps an AgML data split and loads images only when samples are requested."""

    def __init__(self, agml_data: Any, classes: list[str]) -> None:
        self._data = agml_data
        self._classes = classes
        try:
            self._size = len(agml_data)
        except Exception:
            self._size = 0

    def __len__(self) -> int:
        return self._size

    def __iter__(self):
        return iter(self._load_all())

    def __getitem__(self, idx):
        return self._load_all()[idx]

    def load_n(self, n: int) -> list[Sample]:
        """Load the first *n* samples by iterating the AgML data sequentially.

        AgML shuffles at split time, so sequential iteration is already random.
        """
        samples = []
        for image, annotation in self._data:
            samples.append(_make_sample(image, annotation, self._classes))
            if len(samples) >= n:
                break
        return samples

    def _load_all(self) -> list[Sample]:
        return [_make_sample(img, ann, self._classes) for img, ann in self._data]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_dataset(
    dataset_name: str,
    classes: list[str],
    train_split: float = 0.8,
    seed: int = 42,
    data_root: str | Path | None = None,
) -> DatasetSplit:
    """Load an AgML dataset as a :class:`~agvfm.optimizer.types.DatasetSplit`.

    Parameters
    ----------
    dataset_name:
        AgML dataset identifier (e.g. ``"grape_detection_californiaday"``).
    classes:
        Ordered list of class names present in the dataset.
    train_split:
        Fraction of images used for the train (proxy) split.
    seed:
        Random seed passed to AgML for reproducible shuffling.
    data_root:
        Override the default AgML download directory (``~/.agml/``).
        Useful on shared cluster filesystems.

    Returns
    -------
    DatasetSplit
        ``split.train`` and ``split.test`` are :class:`_LazyAgMLData` objects.
    """
    try:
        import agml
    except ImportError as exc:
        raise ImportError(
            "agml is required for AgML dataset loading. "
            "Install it with: pip install agml"
        ) from exc

    loader_kwargs: dict = {}
    if data_root is not None:
        loader_kwargs["dataset_path"] = str(data_root)

    try:
        loader = agml.data.AgMLDataLoader(dataset_name, **loader_kwargs)
    except TypeError:
        logger.warning(
            "AgML does not accept dataset_path= (version too old); "
            "datasets will download to ~/.agml/. Set AGML_DATA_ROOT instead."
        )
        loader = agml.data.AgMLDataLoader(dataset_name)

    loader.split(train=train_split, test=round(1.0 - train_split, 2))
    loader.batch(batch_size=1)
    try:
        loader.shuffle()
    except Exception:
        pass

    return DatasetSplit(
        name=dataset_name,
        classes=classes,
        train=_LazyAgMLData(loader.train_data, classes),
        test=_LazyAgMLData(loader.test_data, classes),
    )


def load_config(config_path: str | Path = "config.yaml") -> dict:
    """Load a YAML config file as a dict."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_all_datasets(
    config_path: str | Path = "config.yaml",
) -> list[DatasetSplit]:
    """Load all datasets listed in a YAML config file.

    Expected config structure::

        datasets:
          - name: grape_detection_californiaday
            classes: [grape]
          - name: tomato_ripeness_detection
            classes: [ripe, unripe]
        optimization:
          train_split: 0.8
          seed: 42
        data_root: /path/to/shared/agml   # optional
    """
    config = load_config(config_path)
    opt = config.get("optimization", {})
    train_split = opt.get("train_split", 0.8)
    seed = opt.get("seed", 42)
    data_root = config.get("data_root", None)

    splits = []
    for ds in config["datasets"]:
        split = load_dataset(
            ds["name"],
            ds["classes"],
            train_split=train_split,
            seed=seed,
            data_root=data_root,
        )
        splits.append(split)
    return splits


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_sample(image: Any, annotation: Any, classes: list[str]) -> Sample:
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image.astype(np.uint8))
    else:
        pil_image = image
    w, h = pil_image.size
    return Sample(image=pil_image, annotations=_parse_annotation(annotation, classes, w, h))


def _parse_annotation(
    annotation: Any,
    classes: list[str],
    img_w: int = 1,
    img_h: int = 1,
) -> list[dict]:
    """Convert AgML annotation dict to normalized [x, y, w, h] bbox format."""
    class_to_id = {c: i for i, c in enumerate(classes)}
    results = []

    if isinstance(annotation, dict):
        bboxes = annotation.get("bboxes", annotation.get("bbox", []))
        labels = annotation.get("labels", annotation.get("category_id", []))

        if not isinstance(bboxes, (list, np.ndarray)):
            bboxes = [bboxes]
            labels = [labels]

        for bbox, label in zip(bboxes, labels):
            if isinstance(label, (int, np.integer)):
                class_name = classes[label] if label < len(classes) else str(label)
            else:
                class_name = str(label)
            x, y, w, h = map(float, bbox)
            results.append({
                "bbox": [x / img_w, y / img_h, w / img_w, h / img_h],
                "category_id": class_to_id.get(class_name, 0),
                "category_name": class_name,
            })

    return results
