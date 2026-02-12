"""YOLO-format label parsing."""

from pathlib import Path
from typing import List, Tuple


def parse_yolo_label(
    label_path: Path,
    img_w: int,
    img_h: int,
) -> List[Tuple[float, float, float, float, int]]:
    """
    Parse YOLOv8 .txt (class_id x_center y_center width height, normalized).

    Returns list of (x1, y1, x2, y2, class_id) in pixel coordinates.
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
