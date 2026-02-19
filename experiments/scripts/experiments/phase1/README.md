# Phase 1: Factor Analysis Experiments

One-factor-at-a-time (OFAT) analysis to identify which prompt components matter most for each model.

## Quick Start

```bash
# Run both models (recommended)
cd /data2/jmearles/AgVFM2
python experiments/scripts/experiments/phase1/run_factor_analysis.py --model all --full-test-set

# Run just one model
python experiments/scripts/experiments/phase1/run_factor_analysis.py --model yolo_world --full-test-set
python experiments/scripts/experiments/phase1/run_factor_analysis.py --model sam3 --full-test-set
```

## Features

- **Incremental saving:** Results saved after each config (can resume if interrupted)
- **Progress tracking:** Shows progress as `[X/34] config_name`
- **Auto-resume:** Skips already-completed configs by default
- **Error handling:** Continues with next config if one fails
- **Full logging:** Detailed logs in `experiments/results/phase1_factor_analysis/logs/`

## Options

```bash
--model {yolo_world,sam3,all}    # Which model(s) to run
--yolo-weights PATH              # YOLO World weights path
--sam3-model-id ID               # SAM3 Hugging Face model ID
--device {cuda,cpu}              # Device to run on
--results-dir PATH               # Custom results directory
--no-resume                      # Start fresh (don't resume)
--data-root PATH                 # Dataset root directory
--full-test-set                  # Use full test set (158 images) instead of holdout (138)
```

## Dataset Auto-Detection

The script automatically looks for dataset in:
1. `./_data/T4_REAL_FLOWER_SOBJ.v18-train_2023_test_location_genotype/` (AgVFM2)
2. `DATA_ROOT` environment variable (if set)
3. `../AgVFM/_data/T4_REAL_FLOWER_SOBJ.v18-train_2023_test_location_genotype/` (fallback)

Or specify with `--data-root`.

## Results

Results saved to: `experiments/results/phase1_factor_analysis/`

- `yolo_world_factor_analysis.json` - Complete YOLO World results
- `sam3_factor_analysis.json` - Complete SAM3 results

Each result includes:
- Prompt and components
- Metrics at IoU 0.3, 0.5, and 0.5:0.95
- Counting metrics (R², RMSE, MAE, etc.)
- Test set info (full vs holdout, n_images)

## Monitoring Progress

```bash
# Quick status check
bash experiments/scripts/monitoring/check_experiment_progress.sh

# Watch log file
tail -f experiments/results/phase1_factor_analysis/logs/phase1_factor_analysis_*.log

# Check results count
jq '.results | keys | length' experiments/results/phase1_factor_analysis/yolo_world_factor_analysis.json
```

## Configuration

Tests 34 configurations (1 baseline + 33 variations) across 7 factor axes:
- Taxonomy (8 values)
- Color (5 values)
- Size (4 values)
- Phenology (6 values)
- Negation (6 values)
- Anatomy (5 values)
- Grammar (6 values)

See `.cursor/plans/approach/approach_phase1.md` for full details.
