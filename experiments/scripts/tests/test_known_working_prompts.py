#!/usr/bin/env python3
"""
Test with prompts that we KNOW worked in previous experiments.

Previous results showed:
- "a cowpea flower" (C3): mAP=0.159, F1=0.299
- "a single yellow cowpea flower with open petals" (C2): mAP=0.332, F1=0.463

If these don't work, something is fundamentally wrong.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from agvfm.models.yolo_world import YOLOWorldModel
from agvfm.data import get_image_paths, load_ground_truth
from agvfm.evaluation.metrics import box_iou, match_predictions_to_gt
def test_known_working_prompts():
    """Test with prompts that definitely worked before."""
    print("=" * 70)
    print("Testing with KNOWN WORKING PROMPTS from previous experiments")
    print("=" * 70)
    
    # Use holdout test set (what previous experiments used)
    # Path from AgVFM (sibling directory)
    dataset_root = project_root / "_data" / "T4_REAL_FLOWER_SOBJ.v18-train_2023_test_location_genotype"
    test_images_dir = dataset_root / "test" / "images"
    labels_dir = dataset_root / "test" / "labels"
    dev_manifest = dataset_root / "dev" / "manifest.txt"
    
    if not test_images_dir.exists():
        print(f"✗ Dataset not found: {test_images_dir}")
        return False
    
    # Get holdout test paths (excludes dev images)
    if dev_manifest.exists():
        with open(dev_manifest) as f:
            dev_names = {line.strip() for line in f if line.strip()}
        all_images = get_image_paths(test_images_dir)
        image_paths = [p for p in all_images if p.name not in dev_names]
        print(f"Using holdout test set: {len(image_paths)} images (excluded {len(all_images) - len(image_paths)} dev images)")
    else:
        image_paths = get_image_paths(test_images_dir)
        print(f"Using all test images: {len(image_paths)} images (no dev manifest found)")
    
    # Prompts that DEFINITELY worked before
    test_prompts = [
        ("a cowpea flower", "C3 baseline - should get mAP~0.16"),
        ("a single yellow cowpea flower with open petals", "C2 - should get mAP~0.33"),
    ]
    
    print(f"\nTesting on first 10 images with GT from holdout test set...")
    
    # Find images with GT
    test_images = []
    for img_path in image_paths[:50]:  # Check more to find ones with GT
        gt = load_ground_truth(img_path, labels_dir)
        if len(gt) > 0:
            test_images.append((img_path, gt))
            if len(test_images) >= 10:
                break
    
    if len(test_images) == 0:
        print("✗ No images with GT found!")
        return False
    
    print(f"Found {len(test_images)} images with GT\n")
    
    # Load model
    print("Loading model...")
    model = YOLOWorldModel(weights_path="model_weights/yolo_world/yolov8x-worldv2.pt")
    
    all_results = []
    
    for prompt, description in test_prompts:
        print("\n" + "-" * 70)
        print(f"Testing: '{prompt}'")
        print(f"Expected: {description}")
        print("-" * 70)
        
        model.set_classes([prompt])
        
        total_preds = 0
        total_gt = 0
        total_tp = 0
        
        for img_path, gt_boxes in test_images:
            try:
                pred_boxes, pred_confs = model.predict(img_path, prompt, conf_threshold=0.1)
                total_preds += len(pred_boxes)
                total_gt += len(gt_boxes)
                
                if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                    tp_mask, _ = match_predictions_to_gt(pred_boxes, pred_confs, gt_boxes, iou_threshold=0.5)
                    total_tp += tp_mask.sum()
            except Exception as e:
                print(f"  ✗ Error on {img_path.name}: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        precision = total_tp / total_preds if total_preds > 0 else 0.0
        recall = total_tp / total_gt if total_gt > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"\nResults on {len(test_images)} images:")
        print(f"  Total predictions: {total_preds}")
        print(f"  Total GT: {total_gt}")
        print(f"  Total TP: {total_tp}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1: {f1:.4f}")
        
        all_results.append({
            'prompt': prompt,
            'predictions': total_preds,
            'gt': total_gt,
            'tp': total_tp,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        })
        
        if total_preds == 0:
            print(f"  ⚠️  WARNING: Zero predictions! This prompt worked before.")
        elif total_tp == 0:
            print(f"  ⚠️  WARNING: Zero true positives! Predictions don't match GT.")
        else:
            print(f"  ✓ Got {total_tp} true positives")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    all_working = True
    for result in all_results:
        status = "✓" if result['predictions'] > 0 and result['tp'] > 0 else "✗"
        print(f"{status} {result['prompt']:50s} Preds: {result['predictions']:3d}, TP: {result['tp']:3d}, F1: {result['f1']:.4f}")
        if result['predictions'] == 0 or result['tp'] == 0:
            all_working = False
    
    if all_working:
        print("\n✅ All known working prompts are working!")
    else:
        print("\n❌ PROBLEM: Known working prompts are not working!")
        print("   Something is wrong with the model or dataset.")
    
    return all_working


if __name__ == "__main__":
    success = test_known_working_prompts()
    sys.exit(0 if success else 1)
