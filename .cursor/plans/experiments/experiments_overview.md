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

### Phase 2: Combination Tests & Absorber Architecture 📋
- **Status:** Waiting for Phase 1
- **Objective:** Systematic combinations, negation strategies
- **Details:** [experiments_phase2.md](experiments_phase2.md)

### Phase 3: Confidence Threshold Analysis 📋
- **Status:** Waiting for Phase 2
- **Objective:** Threshold sensitivity analysis
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

## Notes

- **Incremental saving:** Results saved after each config (can resume if interrupted)
- **Progress tracking:** See individual phase docs for detailed progress
- **Reproducibility:** All configs and results saved in JSON format
