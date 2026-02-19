# Visualization Module

Reusable visualization utilities for experiment results analysis.

## Overview

The `agvfm.visualization` module provides plotting functions for:
- Factor analysis results (Phase 1)
- Precision-Recall curves (Phase 2)
- Confidence threshold sweeps (Phase 3)
- Counting metrics comparison

## Usage

### Factor Analysis Visualizations

```python
from agvfm.visualization import plot_factor_contributions, plot_factor_comparison
import json

# Load results
with open('results/phase1_factor_analysis/yolo_world_factor_analysis.json') as f:
    yolo_results = json.load(f)

# Plot single model factor contributions
plot_factor_contributions(
    yolo_results,
    metric="map",  # or "f1", "precision", "recall"
    iou_threshold=0.5,
    save_path="plots/yolo_factor_contributions.png"
)

# Compare two models side-by-side
with open('results/phase1_factor_analysis/sam3_factor_analysis.json') as f:
    sam3_results = json.load(f)

plot_factor_comparison(
    yolo_results,
    sam3_results,
    metric="map",
    iou_threshold=0.5,
    save_path="plots/factor_comparison.png"
)
```

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

## Quick Script

Use the provided script to generate all Phase 1 visualizations:

```bash
python notebooks/scripts/visualize_phase1_results.py
```

This will:
- Load YOLO World and SAM3 results
- Generate factor contribution plots for each model
- Generate side-by-side comparison plots
- Save all plots to `notebooks/results/phase1_factor_analysis/plots/`

## Design Principles

- **Reusable:** Functions work with any result dictionary structure
- **Configurable:** Customizable metrics, thresholds, labels, save paths
- **Publication-ready:** High DPI (300), proper styling, tight layout
- **Modular:** Each plot type in separate function, easy to extend
