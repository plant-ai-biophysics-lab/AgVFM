#!/usr/bin/env python3
"""
Test YOLO World model on 3 images to verify it works before running full experiments.

This script tests:
1. Model loading
2. Setting classes
3. Making predictions
4. Computing metrics
5. Full evaluation pipeline
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from agvfm.models.yolo_world import YOLOWorldModel
from agvfm.data import get_image_paths, load_ground_truth
from agvfm.evaluation.metrics import box_iou, match_predictions_to_gt, compute_metrics_at_iou
from agvfm.experiments import Evaluator


def test_model_loading():
    """Test 1: Model loads without errors."""
    print("\n" + "=" * 70)
    print("TEST 1: Model Loading")
    print("=" * 70)
    try:
        model = YOLOWorldModel(weights_path="model_weights/yolo_world/yolov8x-worldv2.pt")
        print("✓ Model loaded successfully")
        return model
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_set_classes(model):
    """Test 2: Setting classes works."""
    print("\n" + "=" * 70)
    print("TEST 2: Setting Classes")
    print("=" * 70)
    try:
        model.set_classes(["a flower"])
        print("✓ set_classes(['a flower']) succeeded")
        
        model.set_classes(["a cowpea flower"])
        print("✓ set_classes(['a cowpea flower']) succeeded")
        
        return True
    except Exception as e:
        print(f"✗ set_classes failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_single_prediction(model, test_images_dir, labels_dir):
    """Test 3: Single prediction works."""
    print("\n" + "=" * 70)
    print("TEST 3: Single Prediction")
    print("=" * 70)
    
    # Find an image with GT
    image_paths = get_image_paths(test_images_dir)
    test_img = None
    for img_path in image_paths[:10]:
        gt = load_ground_truth(img_path, labels_dir)
        if len(gt) > 0:
            test_img = img_path
            gt_boxes = gt
            break
    
    if test_img is None:
        print("✗ No image with GT found")
        return False
    
    print(f"Testing on: {test_img.name} ({len(gt_boxes)} GT boxes)")
    
    try:
        # Test prediction
        model.set_classes(["a cowpea flower"])
        pred_boxes, pred_confs = model.predict(test_img, "a cowpea flower", conf_threshold=0.1)
        
        print(f"✓ Prediction succeeded: {len(pred_boxes)} detections")
        
        if len(pred_boxes) > 0:
            print(f"  First box: {pred_boxes[0]}")
            print(f"  First conf: {pred_confs[0]:.4f}")
            
            # Check IoU
            if len(gt_boxes) > 0:
                ious = box_iou(pred_boxes, gt_boxes)
                max_ious = ious.max(axis=1)
                tp_count = (max_ious >= 0.5).sum()
                print(f"  Max IoU: {max_ious.max():.4f}")
                print(f"  TP (IoU >= 0.5): {tp_count}/{len(pred_boxes)}")
        else:
            print("  ⚠️  No detections (may be expected with simple prompts)")
        
        return True
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_prompts(model, test_images_dir, labels_dir):
    """Test 4: Multiple prompts work without device errors."""
    print("\n" + "=" * 70)
    print("TEST 4: Multiple Prompts (Device Stability)")
    print("=" * 70)
    
    # Get 3 images with GT
    image_paths = get_image_paths(test_images_dir)
    test_images = []
    for img_path in image_paths[:20]:
        gt = load_ground_truth(img_path, labels_dir)
        if len(gt) > 0:
            test_images.append((img_path, gt))
            if len(test_images) >= 3:
                break
    
    if len(test_images) < 3:
        print(f"✗ Only found {len(test_images)} images with GT (need 3)")
        return False
    
    print(f"Testing on {len(test_images)} images with different prompts...")
    
    prompts = [
        "a flower",
        "a cowpea flower",
        "a yellow flower",
    ]
    
    try:
        for i, (img_path, gt_boxes) in enumerate(test_images):
            prompt = prompts[i % len(prompts)]
            print(f"\n  Image {i+1}: {img_path.name}")
            print(f"    Prompt: '{prompt}'")
            print(f"    GT boxes: {len(gt_boxes)}")
            
            pred_boxes, pred_confs = model.predict(img_path, prompt, conf_threshold=0.1)
            print(f"    Predictions: {len(pred_boxes)}")
            
            if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                ious = box_iou(pred_boxes, gt_boxes)
                tp_count = (ious.max(axis=1) >= 0.5).sum()
                print(f"    TP: {tp_count}")
        
        print("\n✓ All prompts processed without device errors")
        return True
    except Exception as e:
        print(f"\n✗ Multiple prompts failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluator(test_images_dir, labels_dir):
    """Test 5: Full evaluator pipeline works."""
    print("\n" + "=" * 70)
    print("TEST 5: Full Evaluator Pipeline")
    print("=" * 70)
    
    try:
        # Create model
        model = YOLOWorldModel(weights_path="model_weights/yolo_world/yolov8x-worldv2.pt")
        
        # Filter to just 3 images with GT
        all_image_paths = get_image_paths(test_images_dir)
        test_image_paths = []
        for img_path in all_image_paths[:20]:
            gt = load_ground_truth(img_path, labels_dir)
            if len(gt) > 0:
                test_image_paths.append(img_path)
                if len(test_image_paths) >= 3:
                    break
        
        if len(test_image_paths) < 3:
            print(f"✗ Only found {len(test_image_paths)} images with GT")
            return False
        
        print(f"Testing evaluator on {len(test_image_paths)} images...")
        
        # Create evaluator with filtered paths
        evaluator = Evaluator(
            model=model,
            images_dir=test_images_dir,
            labels_dir=labels_dir,
            conf_threshold=0.1,
            image_paths=test_image_paths,  # Use filtered list
        )
        
        # Run evaluation
        result = evaluator.evaluate_prompt(
            prompt="a cowpea flower",
            iou_thresholds=[0.3, 0.5],
        )
        
        print(f"\n✓ Evaluator completed successfully")
        print(f"  Images: {result['n_images']}")
        print(f"  Predictions: {result['total_predictions']}")
        print(f"  Ground truth: {result['total_ground_truth']}")
        
        metrics_05 = result['metrics_by_iou']['iou_0.5']
        print(f"  mAP@0.5: {metrics_05['map']:.4f}")
        print(f"  Precision: {metrics_05['precision']:.4f}")
        print(f"  Recall: {metrics_05['recall']:.4f}")
        print(f"  F1: {metrics_05['f1']:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Evaluator failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("YOLO World Model - Comprehensive Test Suite")
    print("=" * 70)
    print("\nTesting on 3 images to verify model works before full experiments")
    
    # Find dataset
    test_images_dir = Path("/data2/jmearles/AgVFM2/_data/T4_REAL_FLOWER_SOBJ.v18-train_2023_test_location_genotype/test/images")
    labels_dir = test_images_dir.parent / "labels"
    
    if not test_images_dir.exists():
        print(f"\n✗ Dataset not found: {test_images_dir}")
        sys.exit(1)
    
    print(f"\nDataset: {test_images_dir}")
    
    # Run tests
    results = {}
    
    # Test 1: Model loading
    model = test_model_loading()
    results['model_loading'] = model is not None
    if not model:
        print("\n" + "=" * 70)
        print("❌ CRITICAL: Model loading failed. Cannot continue.")
        print("=" * 70)
        sys.exit(1)
    
    # Test 2: Setting classes
    results['set_classes'] = test_set_classes(model)
    
    # Test 3: Single prediction
    results['single_prediction'] = test_single_prediction(model, test_images_dir, labels_dir)
    
    # Test 4: Multiple prompts
    results['multiple_prompts'] = test_multiple_prompts(model, test_images_dir, labels_dir)
    
    # Test 5: Full evaluator
    results['evaluator'] = test_evaluator(test_images_dir, labels_dir)
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name:20s}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ ALL TESTS PASSED - Model is ready for full experiments")
    else:
        print("❌ SOME TESTS FAILED - Fix issues before running full experiments")
    print("=" * 70)
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
