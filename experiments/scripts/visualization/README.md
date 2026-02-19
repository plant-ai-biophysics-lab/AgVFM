# Visualization Scripts

Scripts for generating plots and visualizations from experiment results.

## Scripts

### `visualize_phase1_results.py`

Generates Phase 1 factor analysis visualizations:
- Factor contribution plots (mAP and F1) for each model
- Side-by-side model comparison plots
- All saved to `experiments/results/phase1_factor_analysis/plots/`

**Usage:**
```bash
python experiments/scripts/visualization/visualize_phase1_results.py
```

**Output:**
- `yolo_world_factor_contributions_map.png`
- `yolo_world_factor_contributions_f1.png`
- `sam3_factor_contributions_map.png`
- `sam3_factor_contributions_f1.png`
- `factor_comparison_map.png`
- `factor_comparison_f1.png`

## Visualization API

For programmatic visualization, use the `agvfm.visualization` package:

```python
from agvfm.visualization.factor_analysis import plot_factor_contributions
from agvfm.visualization.metrics import plot_pr_curve, plot_confidence_sweep

# See agvfm/visualization/README.md for detailed API documentation
```
