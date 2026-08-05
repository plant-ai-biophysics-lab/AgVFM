"""HuggingFace-backed dataset loader for the agvfm optimizer pipeline.

Provides :func:`load_dataset`, which loads a Project-AgML dataset from the
HuggingFace Hub via ``agml.data.hf_loader.HuggingFaceDataLoader`` (the
project has moved off the legacy ``agml.data.AgMLDataLoader`` streaming API)
and wraps it into a :class:`~agvfm.optimizer.types.DatasetSplit` compatible
with :func:`~agvfm.optimizer.types.sample_proxy_images` and every optimizer
loop (:mod:`agvfm.optimizer.loop`, :mod:`agvfm.optimizer.grad_prompt`,
:mod:`agvfm.optimizer.meta_prompt`, :mod:`agvfm.optimizer.prompt_template`,
:mod:`agvfm.optimizer.meta_prompt_template`).

Datasets are identified by their HuggingFace Hub repo id, e.g.
``"Project-AgML/grape_detection_californiaday"``. A bare name with no ``"/"``
(e.g. ``"grape_detection_californiaday"``, matching the naming convention
used throughout this repo's configs/CLI examples) is automatically resolved
under the ``Project-AgML`` org; pass a fully-qualified ``"org/name"`` to load
from elsewhere. Datasets are cached to HuggingFace's default cache
(``~/.cache/huggingface/``) unless *data_root* is given.

Usage
-----
    from agvfm.data.agml_loader import load_dataset, load_all_datasets

    split = load_dataset(
        "grape_detection_californiaday",   # -> Project-AgML/grape_detection_californiaday
        classes=["grape"],
        data_root="/group/jmearlesgrp/hf_cache",
    )
    # split.train / split.test are _LazyHFData objects

Caveat
------
Object-detection annotation schemas are not identical across every
Project-AgML HF dataset. This module assumes the common
``{"image": ..., "objects": {"bbox": [[x, y, w, h], ...], "categories": [...]}}``
layout (pixel-space ``[x, y, w, h]`` boxes) seen on datasets such as
``Project-AgML/grape_detection_californiaday`` — see
:func:`_parse_hf_annotation` for the exact fallback/defensive logic. This has
not been verified end-to-end against a live download in this environment;
sanity-check the parsed ``Sample.annotations`` against a real dataset before
trusting results from it (see ``AGENT.md``).
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

_DEFAULT_HF_ORG = "Project-AgML"


def _resolve_hf_repo_id(dataset_name: str) -> str:
    """Prefix a bare dataset name with the default Project-AgML HF org."""
    return dataset_name if "/" in dataset_name else f"{_DEFAULT_HF_ORG}/{dataset_name}"


# ---------------------------------------------------------------------------
# Lazy HF dataset wrapper
# ---------------------------------------------------------------------------

class _LazyHFData:
    """Wraps a single split of a HuggingFace ``datasets.Dataset``.

    HF's Arrow/parquet-backed ``Dataset`` already decodes image columns
    lazily on row access, so this wrapper's job is just to convert each raw
    row into our :class:`~agvfm.optimizer.types.Sample` shape and expose the
    ``load_n``/``__len__``/``__iter__`` surface the rest of the optimizer
    pipeline expects (see :func:`~agvfm.data.agml_loader._load_pool`).
    """

    def __init__(self, hf_dataset: Any, classes: list[str], category_feature: Any = None) -> None:
        self._data = hf_dataset
        self._classes = classes
        self._category_feature = category_feature
        try:
            self._size = len(hf_dataset)
        except Exception:
            self._size = 0

    def __len__(self) -> int:
        return self._size

    def __iter__(self):
        for i in range(self._size):
            yield _make_sample(self._data[i], self._classes, self._category_feature)

    def __getitem__(self, idx):
        return _make_sample(self._data[idx], self._classes, self._category_feature)

    def load_n(self, n: int | None) -> list[Sample]:
        """Load up to *n* samples (or all when *n* is None).

        ``Dataset.train_test_split`` shuffles by default, so sequential
        access here is already effectively random.
        """
        limit = self._size if n is None else min(n, self._size)
        return [_make_sample(self._data[i], self._classes, self._category_feature) for i in range(limit)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_dataset(
    dataset_name: str,
    classes: list[str],
    train_split: float = 0.8,
    seed: int = 42,
    data_root: str | Path | None = None,
    hf_config: str | None = None,
) -> DatasetSplit:
    """Load a Project-AgML HuggingFace dataset as a :class:`~agvfm.optimizer.types.DatasetSplit`.

    Parameters
    ----------
    dataset_name:
        HF Hub repo id (e.g. ``"Project-AgML/grape_detection_californiaday"``)
        or a bare name (e.g. ``"grape_detection_californiaday"``), which is
        resolved under the ``Project-AgML`` org.
    classes:
        Ordered list of class names present in the dataset. Used as a
        fallback for category-name resolution when the HF dataset's
        ``categories`` feature isn't a ``ClassLabel`` we can decode directly.
    train_split:
        Fraction of images used for the train (proxy) split.
    seed:
        Random seed for reproducible train/test splitting.
    data_root:
        Override HuggingFace's default cache directory
        (``~/.cache/huggingface/``). Useful on shared cluster filesystems.
    hf_config:
        Optional HF dataset config/subset name, for datasets that expose
        more than one (e.g. an augmented variant).

    Returns
    -------
    DatasetSplit
        ``split.train`` and ``split.test`` are :class:`_LazyHFData` objects.
    """
    try:
        from agml.data.hf_loader import HuggingFaceDataLoader
    except ImportError as exc:
        raise ImportError(
            "agml is required for dataset loading. Install it with: pip install agml"
        ) from exc

    repo_id = _resolve_hf_repo_id(dataset_name)
    loader = HuggingFaceDataLoader(
        repo_id,
        config=hf_config,
        cache_dir=str(data_root) if data_root is not None else None,
    )

    ds_dict = loader.dataset
    if "train" in ds_dict and "test" in ds_dict:
        train_ds, test_ds = ds_dict["train"], ds_dict["test"]
    else:
        base = ds_dict["train"] if "train" in ds_dict else ds_dict[next(iter(ds_dict.keys()))]
        split = base.train_test_split(test_size=round(1.0 - train_split, 4), seed=seed)
        train_ds, test_ds = split["train"], split["test"]

    category_feature = _resolve_category_feature(train_ds)

    return DatasetSplit(
        name=dataset_name,
        classes=classes,
        train=_LazyHFData(train_ds, classes, category_feature),
        test=_LazyHFData(test_ds, classes, category_feature),
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
        data_root: /path/to/shared/hf_cache   # optional
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

def _resolve_category_feature(ds: Any) -> Any:
    """Best-effort lookup of the ``ClassLabel`` feature backing ``objects.categories``.

    Lets us decode integer category codes back to names via ``int2str()``
    when the HF dataset defines one; returns ``None`` (triggering the
    positional-index fallback in :func:`_parse_hf_annotation`) if the
    feature schema doesn't match the expected shape.
    """
    try:
        obj_feature = ds.features.get("objects")
        if obj_feature is None:
            return None
        cat_feature = obj_feature.get("categories", obj_feature.get("category"))
        if cat_feature is not None and hasattr(cat_feature, "feature"):
            inner = cat_feature.feature
            return inner if hasattr(inner, "int2str") else None
        if cat_feature is not None and hasattr(cat_feature, "int2str"):
            return cat_feature
    except Exception:
        pass
    return None


def _make_sample(row: dict, classes: list[str], category_feature: Any = None) -> Sample:
    image = row["image"]
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image.astype(np.uint8))
    else:
        pil_image = image
    w, h = pil_image.size
    return Sample(
        image=pil_image,
        annotations=_parse_hf_annotation(row, classes, w, h, category_feature),
    )


def _parse_hf_annotation(
    row: dict,
    classes: list[str],
    img_w: int = 1,
    img_h: int = 1,
    category_feature: Any = None,
) -> list[dict]:
    """Convert a HF dataset row's ``objects`` column to normalized [x, y, w, h] bbox format.

    Assumes the common Project-AgML detection layout —
    ``objects: {"bbox": [[x, y, w, h], ...], "categories": [...]}`` — falling
    back to a few alternate key spellings (``bboxes``/``category``/
    ``category_id``) defensively since not every dataset on the hub has been
    checked against this parser.

    Two dataset-specific quirks handled defensively, found by inspecting raw
    values directly (not assumed) across several Project-AgML datasets:

    - **Coordinate scale is not consistent across datasets.** Most datasets
      checked use pixel-space ``[x, y, w, h]`` (values scale with image
      size), but ``GEMINI_cowpea_flower_detection`` stores already-normalized
      [0, 1] coordinates — dividing those by ``img_w``/``img_h`` again would
      silently shrink every box to ~0, corrupting mAP for that dataset
      specifically without raising any error. Detected per-row via the raw
      max coordinate value rather than hardcoded per-dataset, so it's not
      assumed to be an exhaustive list of exceptions.
    - **``categories`` can be shorter than ``bbox``** (seen on
      ``GEMINI_cowpea_flower_detection``: 4 boxes, 1 category entry) — zipping
      the two lists directly would silently drop the extra boxes. Broadcasts
      the last category value across any extra boxes instead (reasonable for
      the single-class datasets where this has been observed; logs a warning
      so a genuine multi-class mismatch doesn't fail silently).
    """
    class_to_id = {c: i for i, c in enumerate(classes)}
    objects = row.get("objects")
    results: list[dict] = []
    if not isinstance(objects, dict):
        return results

    bboxes = objects.get("bbox", objects.get("bboxes", []))
    labels = objects.get("categories", objects.get("category", objects.get("category_id", [])))

    if not isinstance(bboxes, (list, np.ndarray)):
        bboxes = [bboxes]
        labels = [labels]
    labels = list(labels)

    if 0 < len(labels) < len(bboxes):
        logger.warning(
            f"{len(bboxes)} boxes but only {len(labels)} category label(s) — "
            f"broadcasting the last label across the remaining boxes rather than dropping them."
        )
        labels = labels + [labels[-1]] * (len(bboxes) - len(labels))

    all_coords = [v for bbox in bboxes for v in bbox]
    already_normalized = bool(all_coords) and max(all_coords) <= 1.5

    for bbox, label in zip(bboxes, labels):
        if isinstance(label, (int, np.integer)):
            class_name = None
            if category_feature is not None:
                try:
                    class_name = category_feature.int2str(int(label))
                except Exception:
                    class_name = None
            if class_name is None:
                class_name = classes[label] if label < len(classes) else str(label)
        else:
            class_name = str(label)
        x, y, w, h = map(float, bbox)
        if already_normalized:
            norm_bbox = [x, y, w, h]
        else:
            norm_bbox = [x / img_w, y / img_h, w / img_w, h / img_h]
        results.append({
            "bbox": norm_bbox,
            "category_id": class_to_id.get(class_name, 0),
            "category_name": class_name,
        })

    return results


# ---------------------------------------------------------------------------
# Proxy sampling helpers (cross-dataset template optimisation)
# ---------------------------------------------------------------------------

def _load_pool(split: DatasetSplit) -> list[Sample]:
    train = split.train
    if hasattr(train, "load_n"):
        return train.load_n(None)
    return list(train)


def sample_proxy_images_for_class(
    split: DatasetSplit, class_name: str, n: int | None, seed: int = 42
) -> list[Sample]:
    """Like :func:`~agvfm.optimizer.types.sample_proxy_images` but restricted to *class_name*.

    Needed for multi-class datasets, where most images in the pool won't
    contain every class — sampling from the unfiltered pool would silently
    hand the cross-dataset template optimiser a bunch of irrelevant images.
    Falls back to the unfiltered pool if nothing matches (e.g. unexpected
    annotation format), rather than returning an empty pool outright.
    """
    pool = _load_pool(split)
    filtered = [s for s in pool if any(a["category_name"] == class_name for a in s.annotations)]
    if not filtered:
        filtered = pool
    if n is None:
        return filtered
    rng = random.Random(seed)
    return rng.sample(filtered, min(n, len(filtered)))


def kfold_proxy_splits(
    split: DatasetSplit,
    k: int,
    proxy_images: int | None,
    seed: int = 42,
) -> list[tuple[list[Sample], list[Sample]]]:
    """Return k ``(proxy_train, held_out)`` splits from the training pool.

    Each fold holds out 1/k of the shuffled pool for evaluation; the
    remaining (k-1)/k samples form the proxy set used during optimisation.
    If *proxy_images* is set and smaller than the available proxy pool, the
    proxy set is subsampled.

    When k == 1 the held-out set equals the proxy set (degenerate case,
    equivalent to a single run with no true hold-out).
    """
    pool = _load_pool(split)
    rng = random.Random(seed)
    shuffled = list(pool)
    rng.shuffle(shuffled)
    n_total = len(shuffled)

    if k == 1:
        proxy = rng.sample(shuffled, min(proxy_images, n_total)) if proxy_images else shuffled
        return [(proxy, proxy)]

    fold_size = n_total // k
    folds: list[tuple[list[Sample], list[Sample]]] = []
    for i in range(k):
        val_start = i * fold_size
        val_end = val_start + fold_size if i < k - 1 else n_total
        held_out = shuffled[val_start:val_end]
        proxy_pool = shuffled[:val_start] + shuffled[val_end:]
        if proxy_images is not None and len(proxy_pool) > proxy_images:
            proxy_pool = rng.sample(proxy_pool, proxy_images)
        folds.append((proxy_pool, held_out))
    return folds
