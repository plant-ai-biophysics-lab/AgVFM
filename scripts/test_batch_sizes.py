#!/usr/bin/env python3
"""
Test maximum batch sizes for GPU memory constraints.
Run from project root to avoid import conflicts.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# Now import torch (before agvfm.utils which has logging.py)
import torch

# Now import our utility
from agvfm.utils.batch_size_test import find_optimal_batch_sizes

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test maximum batch sizes for GPU memory")
    parser.add_argument(
        "--model",
        type=str,
        choices=["yolo_world", "sam3", "both"],
        default="both",
        help="Model to test",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[640, 640],
        metavar=("WIDTH", "HEIGHT"),
        help="Test image crop size (default: 640 640)",
    )
    parser.add_argument(
        "--safety-margin",
        type=float,
        default=0.8,
        help="Safety margin for recommended batch size (default: 0.8 = 80%%)",
    )
    
    args = parser.parse_args()
    
    yolo_model = None
    sam3_model = None
    
    if args.model in ["yolo_world", "both"]:
        from agvfm.models.yolo_world import YOLOWorldModel
        print("Loading YOLO World model...")
        yolo_model = YOLOWorldModel()
    
    if args.model in ["sam3", "both"]:
        from agvfm.models.sam3 import SAM3Model
        print("Loading SAM3 model...")
        sam3_model = SAM3Model()
    
    results = find_optimal_batch_sizes(
        yolo_model=yolo_model,
        sam3_model=sam3_model,
        image_size=tuple(args.image_size),
        safety_margin=args.safety_margin,
    )
