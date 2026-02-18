"""
Evaluation pipeline: run model on full or limited image set, compute mAP@0.5, P, R, optional visualizations.
"""

import random
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from PIL import Image

from vlme.data.labels import parse_yolo_label
from vlme.data.sampling import get_image_paths, sample_image_paths
from vlme.evaluation.metrics import compute_metrics_at_iou
from vlme.models.yolo_world import load_yolo_world
try:
    from vlme.models.huggingface import GroundingDinoWrapper, OwlV2Wrapper
except ImportError:
    GroundingDinoWrapper = None
    OwlV2Wrapper = None

from vlme.run import OPEN_SET_CLASS_NAMES
from vlme.viz.boxes import DEFAULT_PREDICT_KWARGS, plot_gt_vs_predictions
from vlme.viz.eval_plots import plot_eval_summary


def _collect_gt_and_pred(
    image_paths: List[Path],
    labels_dir: Path,
    model,
    prediction_class_names: Optional[List[str]],
    predict_kwargs: dict,
) -> tuple:
    """
    Run model on each path, load GT; return list_gt_xyxy, list_pred_xyxy, list_pred_conf.
    Predictions with class "" are excluded.
    """
    list_gt_xyxy: List[np.ndarray] = []
    list_pred_xyxy: List[np.ndarray] = []
    list_pred_conf: List[np.ndarray] = []

    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")
        label_path = labels_dir / (img_path.stem + ".txt")
        gt_boxes = parse_yolo_label(label_path, img.width, img.height)
        gt_xyxy = np.array([[x1, y1, x2, y2] for (x1, y1, x2, y2, _) in gt_boxes], dtype=np.float64)
        if len(gt_xyxy) == 0:
            gt_xyxy = np.zeros((0, 4), dtype=np.float64)
        list_gt_xyxy.append(gt_xyxy)

        results = model.predict(str(img_path), **predict_kwargs)
        r = results[0]
        if r.boxes.xyxy.numel() == 0:
            list_pred_xyxy.append(np.zeros((0, 4), dtype=np.float64))
            list_pred_conf.append(np.zeros(0, dtype=np.float64))
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()
        cls_ = r.boxes.cls.cpu().numpy().astype(int)
        if prediction_class_names is not None:
            keep = np.array(
                [cls_[i] < len(prediction_class_names) and prediction_class_names[cls_[i]] != "" for i in range(len(cls_))],
                dtype=bool,
            )
            xyxy = xyxy[keep]
            conf = conf[keep]
        list_pred_xyxy.append(xyxy.astype(np.float64))
        list_pred_conf.append(conf.astype(np.float64))

    return list_gt_xyxy, list_pred_xyxy, list_pred_conf


def run_evaluation(
    images_dir: Union[str, Path],
    labels_dir: Union[str, Path],
    num_images: Optional[int] = None,
    random_seed: int = 42,
    class_names: Optional[List[str]] = None,
    yolo_classes: Optional[List[str]] = None,
    weights_path: Union[str, Path] = "model_weights/yolov8x-worldv2.pt",
    model_type: str = "yolo",
    predict_kwargs: Optional[dict] = None,
    iou_threshold: float = 0.5,
    plot_summary: bool = True,
    plot_pr_curve: bool = True,
    plot_sample: int = 0,
    show: bool = True,
) -> dict:
    """
    Run evaluation: mAP@0.5, precision, recall. Optionally plot summary and/or qualitative sample.

    Args:
        images_dir: Directory of test images.
        labels_dir: Directory of YOLO .txt labels (same stem as images).
        num_images: If None, use full set of images; else use this many (sampled with random_seed).
        random_seed: Seed for sampling when num_images is not None.
        class_names: Names for GT class IDs (default ["flower"]).
        yolo_classes: Text prompts for YOLO World (default OPEN_SET_CLASS_NAMES).
        weights_path: Path to YOLO World .pt weights or HF model ID.
        model_type: "yolo", "grounding_dino", or "owlv2".
        predict_kwargs: Optional dict for model.predict() (merged over defaults).
        iou_threshold: IoU threshold for mAP/P/R (default 0.5).
        plot_summary: If True, show metrics summary figure.
        plot_pr_curve: If True and plot_summary, include P-R curve in summary.
        plot_sample: If > 0, also run plot_gt_vs_predictions on this many sampled images (separate figure).
        show: Whether to call plt.show() on figures.

    Returns:
        dict with metrics (precision, recall, map, total_tp, total_fp, total_fn, n_images, n_gt_total)
        and optionally "image_paths" (list of paths evaluated), "list_gt_xyxy", "list_pred_xyxy", "list_pred_conf"
        for further diagnostics.
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    class_names = class_names or ["flower"]
    yolo_classes = yolo_classes or OPEN_SET_CLASS_NAMES
    kwargs = dict(DEFAULT_PREDICT_KWARGS)
    kwargs.update(predict_kwargs or {})
    predict_kwargs = kwargs

    if num_images is None:
        image_paths = get_image_paths(images_dir)
    else:
        image_paths = sample_image_paths(images_dir, num_images, random_seed)

    if len(image_paths) == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "map": 0.0,
            "total_tp": 0,
            "total_fp": 0,
            "total_fn": 0,
            "n_images": 0,
            "n_gt_total": 0,
        }

    if model_type == "yolo":
        model = load_yolo_world(weights_path, classes=yolo_classes)
    elif model_type == "grounding_dino":
        if GroundingDinoWrapper is None:
            raise ImportError("Please install transformers and torch to use GroundingDinoWrapper.")
        # weights_path e.g. "IDEA-Research/grounding-dino-base"
        model = GroundingDinoWrapper(str(weights_path))
        # Ensure we pass classes to predict for HF models
        predict_kwargs["classes"] = yolo_classes
    elif model_type == "owlv2":
        if OwlV2Wrapper is None:
            raise ImportError("Please install transformers and torch to use OwlV2Wrapper.")
        # weights_path e.g. "google/owlv2-base-patch16-ensemble"
        model = OwlV2Wrapper(str(weights_path))
        predict_kwargs["classes"] = yolo_classes
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    list_gt_xyxy, list_pred_xyxy, list_pred_conf = _collect_gt_and_pred(
        image_paths, labels_dir, model, yolo_classes, predict_kwargs
    )

    metrics = compute_metrics_at_iou(
        list_gt_xyxy, list_pred_xyxy, list_pred_conf, iou_threshold=iou_threshold
    )

    if plot_summary:
        plot_eval_summary(
            metrics,
            show_pr_curve=plot_pr_curve,
            iou_threshold=iou_threshold,
            list_gt_xyxy=list_gt_xyxy,
            list_pred_xyxy=list_pred_xyxy,
            list_pred_conf=list_pred_conf,
            show=show,
        )

    if plot_sample > 0:
        n_sample = min(plot_sample, len(image_paths))
        rng = random.Random(random_seed)
        sample_paths = rng.sample(image_paths, n_sample)
        plot_gt_vs_predictions(
            sample_paths,
            labels_dir,
            model,
            class_names,
            prediction_class_names=yolo_classes,
            predict_kwargs=predict_kwargs,
            show=show,
        )

    out = dict(metrics)
    out["image_paths"] = image_paths
    out["list_gt_xyxy"] = list_gt_xyxy
    out["list_pred_xyxy"] = list_pred_xyxy
    out["list_pred_conf"] = list_pred_conf
    return out
