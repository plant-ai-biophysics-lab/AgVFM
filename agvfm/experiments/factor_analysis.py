"""Factor analysis experiments (Experiment 7 style)."""

from pathlib import Path
from typing import Dict, List

from agvfm.config.experiments import FACTOR_AXES, build_prompt_from_components, generate_factor_combinations
from agvfm.experiments.evaluator import Evaluator


def run_factor_analysis(
    evaluator: Evaluator,
    iou_thresholds: List[float] = None,
) -> Dict:
    """
    Run one-factor-at-a-time (OFAT) factor analysis.

    Args:
        evaluator: Evaluator instance
        iou_thresholds: List of IoU thresholds for evaluation

    Returns:
        Dictionary with results for each factor combination
    """
    if iou_thresholds is None:
        iou_thresholds = [0.3, 0.5]

    # Generate all combinations
    combinations = generate_factor_combinations()

    print("=" * 80)
    print("Factor Analysis: One-Factor-at-a-Time")
    print("=" * 80)
    print(f"Total combinations: {len(combinations)}")
    print(f"IoU thresholds: {iou_thresholds}")
    print()

    results = {}

    for idx, combo in enumerate(combinations, 1):
        prompt = build_prompt_from_components(combo)
        combo_name = "_".join([f"{k}:{v}" for k, v in combo.items() if v])
        if not combo_name:
            combo_name = "baseline"

        print(f"[{idx}/{len(combinations)}] {combo_name}")
        print(f"  Prompt: {prompt}")

        eval_result = evaluator.evaluate_prompt(
            prompt=prompt,
            iou_thresholds=iou_thresholds,
        )

        results[combo_name] = {
            "components": combo,
            "prompt": prompt,
            **eval_result,
        }

        # Print summary
        metrics_05 = eval_result["metrics_by_iou"]["iou_0.5"]
        print(f"  mAP@0.5: {metrics_05['map']:.4f}, F1: {metrics_05['f1']:.4f}")
        print()

    return {
        "experiment_type": "factor_analysis",
        "model": evaluator.model.model_name,
        "n_combinations": len(combinations),
        "iou_thresholds": iou_thresholds,
        "results": results,
    }


def analyze_factor_contributions(factor_results: Dict) -> Dict:
    """
    Analyze contribution of each factor axis.

    Args:
        factor_results: Results from run_factor_analysis()

    Returns:
        Dictionary with per-axis contributions
    """
    results = factor_results["results"]
    baseline_result = results.get("baseline", {})

    contributions = {}

    for axis in FACTOR_AXES:
        axis_name = axis.name
        axis_contributions = []

        for combo_name, result in results.items():
            if combo_name == "baseline":
                continue

            components = result.get("components", {})
            if axis_name not in components:
                continue

            # Compare to baseline
            baseline_map = baseline_result.get("metrics_by_iou", {}).get("iou_0.5", {}).get("map", 0.0)
            combo_map = result.get("metrics_by_iou", {}).get("iou_0.5", {}).get("map", 0.0)
            delta = combo_map - baseline_map

            axis_contributions.append({
                "value": components[axis_name],
                "delta_map": delta,
                "map": combo_map,
            })

        contributions[axis_name] = axis_contributions

    return contributions
