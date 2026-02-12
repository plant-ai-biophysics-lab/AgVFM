"""Bounding box and detection visualization."""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from PIL import Image

from vlme.data.labels import parse_yolo_label


def _default_predict_kwargs() -> dict:
    """Default arguments for model.predict(); user dict overrides these. Device: cuda if available, else mps (Mac), else cpu."""
    out = dict(imgsz=1280, conf=0.1, max_det=500, verbose=False, iou=0.3)
    try:
        import torch
        if torch.cuda.is_available():
            out["device"] = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            out["device"] = "mps"
        # else leave unset so ultralytics uses cpu
    except Exception:
        pass
    return out


# Default arguments for model.predict(); user dict overrides these.
DEFAULT_PREDICT_KWARGS = _default_predict_kwargs()


def _draw_gt_boxes(ax, gt_boxes: List[Tuple[float, float, float, float, int]], class_names: List[str]):
    """Draw ground-truth boxes (green) and class labels on ax."""
    for (x1, y1, x2, y2, cid) in gt_boxes:
        rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="lime", facecolor="none")
        ax.add_patch(rect)
        name = class_names[cid] if cid < len(class_names) else str(cid)
        ax.text(x1, y1 - 4, name, color="lime", fontsize=8, weight="bold")


def _draw_pred_boxes(
    ax,
    pred_xyxy: np.ndarray,
    pred_cls: Optional[np.ndarray] = None,
    prediction_class_names: Optional[List[str]] = None,
):
    """
    Draw predicted boxes (red, dashed) on ax.
    If prediction_class_names is provided, skip background (""), and label each box with its class name.
    """
    n = len(pred_xyxy)
    cls_ids = pred_cls if pred_cls is not None else np.zeros(n, dtype=np.int64)
    for i, (x1, y1, x2, y2) in enumerate(pred_xyxy):
        name = ""
        if prediction_class_names is not None and i < len(cls_ids):
            cid = int(cls_ids[i])
            name = prediction_class_names[cid] if cid < len(prediction_class_names) else ""
        if name == "":
            continue
        rect = Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1.5, edgecolor="red", facecolor="none", linestyle="--",
        )
        ax.add_patch(rect)
        ax.text(x1, y1 - 4, name, color="red", fontsize=7, weight="bold")


def plot_gt_vs_predictions(
    image_paths: List[Path],
    labels_dir: Union[str, Path],
    model,
    class_names: List[str],
    *,
    prediction_class_names: Optional[List[str]] = None,
    predict_kwargs: Optional[dict] = None,
    ncols: int = 2,
    figsize_per_subplot: Tuple[float, float] = (6.0, 5.0),
    show: bool = True,
    return_results: bool = False,
):
    """
    Plot a grid of images with ground-truth boxes (green) and model predicted boxes (red dashed).

    Args:
        image_paths: List of paths to images.
        labels_dir: Directory containing YOLO .txt label files (same stem as image names).
        model: Loaded YOLO World (or compatible) model with .predict() returning results with .boxes.xyxy.
        class_names: Names for GT class IDs (e.g. ["flower"]).
        prediction_class_names: Names for model prediction classes (same order as model.set_classes()).
            Boxes with class name "" (background) are not drawn. If provided, each predicted box is labeled.
        predict_kwargs: Optional dict passed to model.predict(). Keys can include imgsz, conf, iou, max_det, verbose.
            Defaults: imgsz=1280, conf=0.1, iou=0.3, max_det=500, verbose=False. User dict overrides these.
        ncols: Number of columns in the grid.
        figsize_per_subplot: (width, height) per subplot.
        show: If True, call plt.show().
        return_results: If True, return a list of YOLO World result objects (one per image) for diagnostics.

    Returns:
        (fig, axes) when return_results is False. (fig, axes, results_list) when return_results is True,
        where results_list[i] is the ultralytics Results object for image_paths[i].
    """
    labels_dir = Path(labels_dir)
    kwargs = dict(DEFAULT_PREDICT_KWARGS)
    kwargs.update(predict_kwargs or {})
    predict_kwargs = kwargs

    n = len(image_paths)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per_subplot[0] * ncols, figsize_per_subplot[1] * nrows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    results_list: List[object] = [] if return_results else None

    for i, img_path in enumerate(image_paths):
        ax = axes[i]
        img = Image.open(img_path).convert("RGB")
        img_arr = np.asarray(img)
        ax.imshow(img_arr)

        # Ground truth
        label_path = labels_dir / (img_path.stem + ".txt")
        gt_boxes = parse_yolo_label(label_path, img.width, img.height)
        _draw_gt_boxes(ax, gt_boxes, class_names)

        # Predictions
        results = model.predict(str(img_path), **predict_kwargs)
        if return_results:
            results_list.append(results[0])
        if results[0].boxes.xyxy.numel() > 0:
            pred_xyxy = results[0].boxes.xyxy.cpu().numpy()
            pred_cls = results[0].boxes.cls.cpu().numpy() if results[0].boxes.cls.numel() else None
            _draw_pred_boxes(ax, pred_xyxy, pred_cls=pred_cls, prediction_class_names=prediction_class_names)

        ax.set_title(img_path.name, fontsize=9)
        ax.axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    legend_elements = [Line2D([0], [0], color="lime", lw=2, label="Label (GT)")]
    if prediction_class_names:
        for name in prediction_class_names:
            if name:
                legend_elements.append(Line2D([0], [0], color="red", lw=2, linestyle="--", label=name))
    else:
        legend_elements.append(Line2D([0], [0], color="red", lw=2, linestyle="--", label="YOLO World"))
    fig.legend(handles=legend_elements, loc="upper center", ncol=min(len(legend_elements), 4))
    plt.tight_layout()
    if show:
        plt.show()
    if return_results:
        return fig, axes, results_list
    return fig, axes
