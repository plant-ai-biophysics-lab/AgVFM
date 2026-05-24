# Visualization Scripts

Scripts and modules for generating plots and visualizations from experiment results.

## Directory Structure

```
visualization/
├── visualize_phase1_results.py    # Phase 1 visualization script
├── visualize_phase2_results.py    # Phase 2 visualization script
├── phase1_factor_analysis.py      # Phase 1 plotting functions (experiment-specific)
├── phase2_combinations.py          # Phase 2 plotting functions (experiment-specific)
└── README.md
```

## Scripts

### `visualize_phase1_results.py`

Generates Phase 1 factor analysis visualizations:
- Factor contribution plots (mAP, F1, Precision, Recall) for each model
- Side-by-side model comparison plots
- All saved to `experiments/results/phase1_factor_analysis/plots/`

**Usage:**
```bash
python experiments/scripts/visualization/visualize_phase1_results.py
```

**Output:**
- `yolo_world_factor_contributions_all_metrics.png` (2x2 grid)
- `yolo_world_factor_contributions_map.png`
- `yolo_world_factor_contributions_f1.png`
- `yolo_world_factor_contributions_precision.png`
- `yolo_world_factor_contributions_recall.png`
- Similar plots for SAM3
- `factor_comparison_map.png` and `factor_comparison_f1.png` (if both models run)

### `visualize_phase2_results.py`

Generates Phase 2 combination and absorber visualizations:
- Combination performance plots (all 13 configs ranked by metric)
- Absorber comparison plots (text negation vs absorber architecture)
- Precision-Recall tradeoff plot with F1 iso-lines
- All saved to `experiments/results/phase2_combinations/plots/`

**Usage:**
```bash
python experiments/scripts/visualization/visualize_phase2_results.py
```

**Output:**
- `yolo_world_combinations_all_metrics.png` (2x2 grid)
- `yolo_world_combinations_map.png`, `f1.png`, `precision.png`, `recall.png`
- `yolo_world_absorber_comparison_*.png` (for each metric)
- `yolo_world_precision_recall_tradeoff.png`

## Visualization Modules

### Experiment-Specific Modules

These modules contain plotting functions specific to each experiment phase:

- **`phase1_factor_analysis.py`**: Functions for Phase 1 OFAT factor analysis plots
- **`phase2_combinations.py`**: Functions for Phase 2 combination and absorber plots

These are imported by the visualization scripts and are experiment-specific (not part of the general `agvfm` library).

### General-Purpose Visualization

For general-purpose visualization utilities (PR curves, confidence sweeps, counting metrics), use the `agvfm.visualization` package:

```python
from agvfm.visualization import plot_pr_curve, plot_confidence_sweep, plot_counting_metrics

# See agvfm/visualization/README.md for detailed API documentation
```

## Design Philosophy

- **Experiment-specific code** (like `phase1_factor_analysis.py`, `phase2_combinations.py`) lives in `experiments/scripts/visualization/` alongside the scripts that use them
- **General-purpose utilities** (like PR curves, confidence sweeps) live in `agvfm/visualization/` as part of the reusable library
- This keeps the `agvfm` library general-purpose while experiment-specific analysis code stays in the experiments directory
