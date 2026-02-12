"""YOLO World model loading and inference."""

from pathlib import Path
from typing import List, Optional, Union

from ultralytics import YOLOWorld


def load_yolo_world(
    weights_path: Union[str, Path] = "model_weights/yolov8x-worldv2.pt",
    classes: Optional[List[str]] = None,
):
    """
    Load YOLO World model and optionally set text classes.

    Args:
        weights_path: Path to .pt weights.
        classes: Optional list of class names for open-vocab detection.
                 If None, model is returned without set_classes.

    Returns:
        Loaded YOLOWorld model instance.
    """
    model = YOLOWorld(str(weights_path))
    if classes is not None:
        model.set_classes(classes)
    return model
