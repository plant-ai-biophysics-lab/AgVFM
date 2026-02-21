# Experiments Overview

> **Purpose:** Track overall experiment progress and provide quick reference

## Experiment Phases

### Phase 1: Factor Analysis ✅
- **Status:** ✅ Complete (Both YOLO World and SAM3)
- **Objective:** OFAT analysis across 7 axes
- **Script:** `experiments/scripts/experiments/phase1/run_factor_analysis.py`
- **Results:** `experiments/results/phase1_factor_analysis/`
- **Plots:** `experiments/results/phase1_factor_analysis/plots/`
- **Details:** [experiments_phase1.md](experiments_phase1.md)

### Phase 2: Combination Tests & Absorber Architecture ✅
- **Status:** ✅ Complete (YOLO World: 21 combinations + 7 absorbers; SAM3: 21 combinations)
- **Objective:** Systematic combinations, negation strategies, strategic optimization
- **Script:** `experiments/scripts/experiments/phase2/run_combinations.py`
- **Results:** `experiments/results/phase2_combinations/`
- **Plots:** `experiments/results/phase2_combinations/plots/`
- **Visualization:** `experiments/scripts/visualization/visualize_phase2_results.py`
- **Best performers:**
  - **YOLO World:** `C2_not_bud_calyx` (mAP@0.5: 0.4123)
  - **SAM3:** `C_color_taxonomy_anatomy_tiny` (mAP@0.5: 0.5364)
- **Details:** [experiments_phase2.md](experiments_phase2.md)

### Phase 3: Confidence Threshold Analysis
- [experiments_phase3.md](experiments_phase3.md)

## Phase 4: Unlabeled Threshold Selection
- [experiments_phase4.md](experiments_phase4.md) ✅
- **Status:** ✅ Complete (Both YOLO World and SAM3)
- **Objective:** Threshold sensitivity analysis for zero-shot deployment
- **Script:** `experiments/scripts/experiments/phase3/run_confidence_sweeps.py`
- **Results:** `experiments/results/phase3_confidence_sweeps/`
- **Plots:** `experiments/results/phase3_confidence_sweeps/plots/`
- **Visualization:** `experiments/scripts/visualization/visualize_phase3_results.py`
- **Configs:** Top 3 configs per model from Phase 2 (by mAP@0.5)
- **Thresholds:** 18 values (0.05 to 0.90, step 0.05)
- **Details:** [experiments_phase3.md](experiments_phase3.md)

## Quick Start

### Running Phase 1

```bash
# Run both models
python experiments/scripts/experiments/phase1/run_factor_analysis.py --model all --full-test-set

# Run just YOLO World
python experiments/scripts/experiments/phase1/run_factor_analysis.py --model yolo_world --full-test-set

# Run just SAM3
python experiments/scripts/experiments/phase1/run_factor_analysis.py --model sam3 --full-test-set

# Start fresh (don't resume)
python experiments/scripts/experiments/phase1/run_factor_analysis.py --model all --full-test-set --no-resume
```

### Running Phase 2

```bash
# Run both models (combinations only)
python experiments/scripts/experiments/phase2/run_combinations.py --model all --full-test-set

# Run just YOLO World (combinations + absorbers)
python experiments/scripts/experiments/phase2/run_combinations.py --model yolo_world --full-test-set

# Run just SAM3 (combinations only)
python experiments/scripts/experiments/phase2/run_combinations.py --model sam3 --full-test-set

# Start fresh (don't resume)
python experiments/scripts/experiments/phase2/run_combinations.py --model all --full-test-set --no-resume
```

### Dataset Location

The script auto-detects dataset in:
1. `../AgVFM/_data/...` (sibling directory)
2. `./_data/...` (current directory)

Or specify with `--data-root`:
```bash
python experiments/scripts/experiments/phase1/run_factor_analysis.py --data-root /path/to/dataset
```

## Results Location

All results saved to: `experiments/results/`

- Phase 1: `experiments/results/phase1_factor_analysis/`
- Phase 2: `experiments/results/phase2_combinations/`
- Phase 3: `experiments/results/phase3_confidence_sweeps/`

## Visualization

Generate plots from results:

```bash
# Phase 1 visualizations
python experiments/scripts/visualization/visualize_phase1_results.py

# Phase 2 visualizations
python experiments/scripts/visualization/visualize_phase2_results.py

# Phase 3 visualizations
python experiments/scripts/visualization/visualize_phase3_results.py --model all
```

**Note:** Experiment-specific visualization code lives in `experiments/scripts/visualization/` alongside the scripts. General-purpose visualization utilities are in `agvfm/visualization/`.

## Notes

- **Incremental saving:** Results saved after each config (can resume if interrupted)
- **Progress tracking:** See individual phase docs for detailed progress
- **Reproducibility:** All configs and results saved in JSON format
