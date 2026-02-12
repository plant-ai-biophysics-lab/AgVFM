"""
High-level pipelines: sample images, run model, plot GT vs predictions.
"""

from pathlib import Path
from typing import List, Optional, Union

from vlme.data.sampling import sample_image_paths
from vlme.models.yolo_world import load_yolo_world
from vlme.viz.boxes import plot_gt_vs_predictions

# Open-set class names for YOLO World. Empty string is background (positive examples only; not drawn).
OPEN_SET_CLASS_NAMES = [
    "small yellow flower",
    "small white flower",
    "small yellow bud",
    "small white bud",
    "",
]


def run_gt_vs_predictions(
    images_dir: Union[str, Path],
    labels_dir: Union[str, Path],
    num_images: int = 6,
    random_seed: int = 42,
    class_names: Optional[List[str]] = None,
    yolo_classes: Optional[List[str]] = None,
    weights_path: Union[str, Path] = "model_weights/yolov8x-worldv2.pt",
    predict_kwargs: Optional[dict] = None,
    ncols: int = 2,
    show: bool = True,
    return_results: bool = False,
):
    """
    Sample images, run YOLO World, and plot ground-truth vs predicted boxes.

    Args:
        images_dir: Directory of test images.
        labels_dir: Directory of YOLO .txt labels (same stem as images).
        num_images: Number of images to sample.
        random_seed: Seed for reproducible sampling.
        class_names: Names for GT class IDs (default ["flower"]).
        yolo_classes: Text prompts for YOLO World (default OPEN_SET_CLASS_NAMES: flowers, buds; "" = background, not drawn).
        weights_path: Path to YOLO World .pt weights.
        predict_kwargs: Optional dict of arguments for model.predict(). Merged over defaults. Common keys:
            imgsz (int): input size, e.g. 640, 1280; default 1280.
            conf (float): confidence threshold; default 0.1.
            iou (float): NMS IoU threshold; default 0.3.
            max_det (int): max detections per image; default 500.
            verbose (bool): default False.
            Example: predict_kwargs=dict(imgsz=640, conf=0.25, iou=0.5).
        ncols: Subplot columns.
        show: Whether to call plt.show().
        return_results: If True, return a list of YOLO World result objects (one per image) for diagnostics.

    Returns:
        (fig, axes) when return_results is False. (fig, axes, results_list) when return_results is True,
        where results_list[i] is the ultralytics Results object for the i-th processed image.
    """
    class_names = class_names or ["flower"]
    yolo_classes = yolo_classes or OPEN_SET_CLASS_NAMES

    model = load_yolo_world(weights_path, classes=yolo_classes)
    image_paths = sample_image_paths(images_dir, num_images, random_seed)
    return plot_gt_vs_predictions(
        image_paths,
        labels_dir,
        model,
        class_names,
        prediction_class_names=yolo_classes,
        predict_kwargs=predict_kwargs,
        ncols=ncols,
        show=show,
        return_results=return_results,
    )
