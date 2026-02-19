"""YOLO-format label parsing and dataset utilities."""

from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image


def parse_yolo_label(
    label_path: Path,
    img_w: int,
    img_h: int,
) -> List[Tuple[float, float, float, float, int]]:
    """
    Parse YOLO format label file (class_id x_center y_center width height, normalized).

    Args:
        label_path: Path to .txt label file
        img_w: Image width in pixels
        img_h: Image height in pixels

    Returns:
        List of (x1, y1, x2, y2, class_id) tuples in pixel coordinates (xyxy format).
    """
    boxes: List[Tuple[float, float, float, float, int]] = []
    if not Path(label_path).exists():
        return boxes

    for line in open(label_path):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        cid = int(parts[0])
        xc, yc, w, h = map(float, parts[1:5])
        x1 = (xc - w / 2) * img_w
        y1 = (yc - h / 2) * img_h
        x2 = (xc + w / 2) * img_w
        y2 = (yc + h / 2) * img_h
        boxes.append((x1, y1, x2, y2, cid))

    return boxes


def load_ground_truth(
    image_path: Path,
    labels_dir: Path,
) -> np.ndarray:
    """
    Load ground truth boxes for an image.

    Args:
        image_path: Path to image file
        labels_dir: Directory containing YOLO format label files

    Returns:
        Array of shape (N, 4) with boxes in xyxy format, or (0, 4) if no labels.
    """
    img = Image.open(image_path).convert("RGB")
    label_path = labels_dir / (image_path.stem + ".txt")
    boxes = parse_yolo_label(label_path, img.width, img.height)
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float64)
    # Return only xyxy coordinates (drop class_id)
    return np.array([[x1, y1, x2, y2] for (x1, y1, x2, y2, _) in boxes], dtype=np.float64)


def get_image_paths(images_dir: Path, extensions: Tuple[str, ...] = (".jpg", ".png")) -> List[Path]:
    """
    Collect all image paths from a directory.

    Args:
        images_dir: Directory containing images
        extensions: Image file extensions to include

    Returns:
        Sorted list of image paths
    """
    paths = []
    for ext in extensions:
        paths.extend(sorted(Path(images_dir).glob(f"*{ext}")))
    return sorted(set(paths))
