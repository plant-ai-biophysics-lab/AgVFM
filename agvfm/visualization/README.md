# Visualization Module

General-purpose visualization utilities for experiment results analysis.

## Overview

The `agvfm.visualization` module provides **general-purpose** plotting functions that can be reused across different experiments:
- Precision-Recall curves
- Confidence threshold sweeps
- Counting metrics comparison

**Note:** Experiment-specific visualization code (like Phase 1 factor analysis plots or Phase 2 combination plots) lives in `experiments/scripts/visualization/` alongside the experiment scripts. This keeps the `agvfm` library general-purpose and reusable.

## Usage

### Precision-Recall Curves

```python
from agvfm.visualization import plot_pr_curve

# List of result dictionaries from evaluate_prompt()
results_list = [result1, result2, result3]
labels = ["Config 1", "Config 2", "Config 3"]

plot_pr_curve(
    results_list,
    labels=labels,
    iou_threshold=0.5,
    save_path="plots/pr_curve.png"
)
```

### Confidence Threshold Sweeps

```python
from agvfm.visualization import plot_confidence_sweep

# Dictionary with confidence thresholds as keys
sweep_results = {
    "0.1": result_dict_1,
    "0.2": result_dict_2,
    # ...
}

plot_confidence_sweep(
    sweep_results,
    metrics=["map", "f1", "precision", "recall"],
    save_path="plots/confidence_sweep.png"
)
```

### Counting Metrics

```python
from agvfm.visualization import plot_counting_metrics

plot_counting_metrics(
    results_list,
    labels=["Config 1", "Config 2"],
    save_path="plots/counting_metrics.png"
)
```

## Experiment-Specific Visualizations

For experiment-specific visualizations (Phase 1 factor analysis, Phase 2 combinations), use the scripts in `experiments/scripts/visualization/`:

```bash
# Phase 1 visualizations
python experiments/scripts/visualization/visualize_phase1_results.py

# Phase 2 visualizations
python experiments/scripts/visualization/visualize_phase2_results.py
```

These scripts use experiment-specific plotting modules located in the same directory.

## Design Principles

- **Reusable:** Functions work with any result dictionary structure
- **Configurable:** Customizable metrics, thresholds, labels, save paths
- **Publication-ready:** High DPI (300), proper styling, tight layout
- **Modular:** Each plot type in separate function, easy to extend
- **Separation of concerns:** General-purpose code in `agvfm/`, experiment-specific code in `experiments/`