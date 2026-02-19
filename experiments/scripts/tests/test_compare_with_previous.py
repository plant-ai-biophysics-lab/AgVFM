#!/usr/bin/env python3
"""
Compare our implementation with the previous working code to find the issue.

Previous code:
- Sets classes ONCE at initialization
- Calls model.predict() directly with string path
- Uses holdout test set (138 images)

If we get zero detections, something is fundamentally wrong.
"""

import sys
from pathlib import Path

# Add project root to path
agvfm2_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(agvfm2_root))

from ultralytics import YOLOWorld
from agvfm.data import get_image_paths, load_ground_truth
from PIL import Image
import numpy as np

# Dataset paths
dataset_root = Path("/data2/jmearles/AgVFM2/_data/T4_REAL_FLOWER_SOBJ.v18-train_2023_test_location_genotype")
test_images_dir = dataset_root / "test" / "images"
labels_dir = dataset_root / "test" / "labels"
dev_manifest = dataset_root / "dev" / "manifest.txt"

# Get holdout test paths
if dev_manifest.exists():
    with open(dev_manifest) as f:
        dev_names = {line.strip() for line in f if line.strip()}
    all_images = get_image_paths(test_images_dir)
    image_paths = [p for p in all_images if p.name not in dev_names]
else:
    image_paths = get_image_paths(test_images_dir)

print("=" * 70)
print("COMPARISON TEST: Previous Working Code vs Our Implementation")
print("=" * 70)

# Test 1: Previous working approach
print("\nTEST 1: Previous Working Code (sets classes ONCE)")
print("-" * 70)

model_prev = YOLOWorld("model_weights/yolo_world/yolov8x-worldv2.pt")
model_prev.set_classes(["a cowpea flower"])  # Set ONCE

print(f"Testing on {len(image_paths)} holdout test images")

total_preds_prev = 0
tested = 0
for img_path in image_paths[:20]:  # Test first 20
    gt_boxes = load_ground_truth(img_path, labels_dir)
    
    if len(gt_boxes) == 0:
        continue
    
    tested += 1
    results = model_prev.predict(str(img_path), conf=0.1, iou=0.5, verbose=False)
    r = results[0]
    if r.boxes.xyxy.numel() > 0:
        boxes = r.boxes.xyxy.cpu().numpy()
        total_preds_prev += len(boxes)
        if tested <= 3:
            print(f"  Image {tested}: {len(boxes)} detections")

print(f"\nPrevious code: {total_preds_prev} total predictions on {tested} images with GT")

# Test 2: Our wrapper (but set classes ONCE like previous)
print("\nTEST 2: Our Wrapper (sets classes ONCE)")
print("-" * 70)

from agvfm.models.yolo_world import YOLOWorldModel

model_ours = YOLOWorldModel(weights_path="model_weights/yolo_world/yolov8x-worldv2.pt")
model_ours.set_classes(["a cowpea flower"])  # Set ONCE

total_preds_ours = 0
tested = 0
for img_path in image_paths[:20]:
    gt_boxes = load_ground_truth(img_path, labels_dir)
    
    if len(gt_boxes) == 0:
        continue
    
    tested += 1
    # Use model's predict directly (bypass our wrapper's set_classes logic)
    results = model_ours.model.predict(str(img_path), conf=0.1, iou=0.5, verbose=False)
    r = results[0]
    if r.boxes.xyxy.numel() > 0:
        boxes = r.boxes.xyxy.cpu().numpy()
        total_preds_ours += len(boxes)
        if tested <= 3:
            print(f"  Image {tested}: {len(boxes)} detections")

print(f"\nOur wrapper: {total_preds_ours} total predictions on {tested} images with GT")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Previous code predictions: {total_preds_prev}")
print(f"Our wrapper predictions: {total_preds_ours}")

if total_preds_prev == 0:
    print("\n⚠️  Previous code also gets ZERO detections!")
    print("   This suggests:")
    print("   1. The weights file might be different")
    print("   2. These specific images don't have flowers")
    print("   3. Something else is wrong")
elif total_preds_ours == 0 and total_preds_prev > 0:
    print("\n❌ PROBLEM: Our wrapper gets zero but previous code works!")
    print("   There's a bug in our implementation.")
elif total_preds_ours > 0 and total_preds_prev > 0:
    print("\n✅ Both work! The issue is with calling set_classes repeatedly.")
else:
    print("\n⚠️  Both get zero detections - need to investigate further")
