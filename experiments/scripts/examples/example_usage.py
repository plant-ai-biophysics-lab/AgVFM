"""
Example script demonstrating AgVFM usage.

This script shows how to:
1. Load models (YOLO World, SAM3)
2. Create an evaluator
3. Run factor analysis
4. Evaluate prompts
5. Run confidence threshold sweeps
"""

from pathlib import Path

from agvfm.experiments import Evaluator, run_factor_analysis
from agvfm.models import SAM3Model, YOLOWorldModel


def main():
    """Example usage of AgVFM framework."""

    # Data paths (adjust to your dataset location)
    data_root = Path("_data/T4_REAL_FLOWER_SOBJ.v18-train_2023_test_location_genotype")
    images_dir = data_root / "test" / "images"
    labels_dir = data_root / "test" / "labels"

    print("=" * 80)
    print("AgVFM Example Usage")
    print("=" * 80)
    print()

    # Example 1: YOLO World evaluation
    print("Example 1: YOLO World Evaluation")
    print("-" * 80)

    yolo_model = YOLOWorldModel(weights_path="model_weights/yolo_world/yolov8x-worldv2.pt")
    evaluator = Evaluator(
        model=yolo_model,
        images_dir=images_dir,
        labels_dir=labels_dir,
        conf_threshold=0.1,
    )

    # Evaluate a single prompt
    results = evaluator.evaluate_prompt(
        prompt="a single yellow cowpea flower with open petals",
        iou_thresholds=[0.3, 0.5],
    )

    print(f"Prompt: {results['prompt']}")
    print(f"mAP@0.5: {results['metrics_by_iou']['iou_0.5']['map']:.4f}")
    print(f"F1@0.5: {results['metrics_by_iou']['iou_0.5']['f1']:.4f}")
    print(f"Counting R²: {results['counting']['r2']:.4f}")
    print()

    # Example 2: SAM3 evaluation
    print("Example 2: SAM3 Evaluation")
    print("-" * 80)

    sam3_model = SAM3Model(model_id="facebook/sam3", device="cuda")
    sam3_evaluator = Evaluator(
        model=sam3_model,
        images_dir=images_dir,
        labels_dir=labels_dir,
        conf_threshold=0.1,
    )

    sam3_results = sam3_evaluator.evaluate_prompt(
        prompt="a single yellow cowpea flower with open petals",
        iou_thresholds=[0.3, 0.5],
    )

    print(f"Prompt: {sam3_results['prompt']}")
    print(f"mAP@0.5: {sam3_results['metrics_by_iou']['iou_0.5']['map']:.4f}")
    print(f"F1@0.5: {sam3_results['metrics_by_iou']['iou_0.5']['f1']:.4f}")
    print()

    # Example 3: Confidence threshold sweep
    print("Example 3: Confidence Threshold Sweep")
    print("-" * 80)

    conf_sweep = evaluator.evaluate_confidence_sweep(
        prompt="a single yellow cowpea flower with open petals",
        conf_thresholds=[0.1, 0.2, 0.3, 0.4, 0.5],
        iou_threshold=0.5,
    )

    print("Confidence Threshold Analysis:")
    for conf_thresh, result in conf_sweep["results_by_conf"].items():
        metrics = result["metrics"]
        print(f"  conf={conf_thresh:.1f}: mAP={metrics['map']:.4f}, F1={metrics['f1']:.4f}")
    print()

    # Example 4: Factor analysis (commented out - takes longer)
    # print("Example 4: Factor Analysis")
    # print("-" * 80)
    # factor_results = run_factor_analysis(evaluator, iou_thresholds=[0.5])
    # print(f"Completed {factor_results['n_combinations']} combinations")
    # print()

    print("=" * 80)
    print("Examples complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
