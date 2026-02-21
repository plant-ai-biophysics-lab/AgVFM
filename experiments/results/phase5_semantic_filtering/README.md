# Phase 5 Semantic Filtering Results

## Directory Structure

- `results/` - All JSON result files (test and full dataset)
- `plots/` - All visualization plots
- `logs/` - Log files from runs (if any)

## Results Files

### Test Results (5 images)
- `results/semantic_filtering_results.json` - Baseline (q=0.10, per-image BH-FDR)
- `results/semantic_filtering_results_q0.1.json` - Per-image BH-FDR with q=0.1
- `results/semantic_filtering_results_q0.2.json` - Per-image BH-FDR with q=0.2
- `results/semantic_filtering_results_q0.3.json` - Per-image BH-FDR with q=0.3
- `results/semantic_filtering_results_q0.5.json` - Per-image BH-FDR with q=0.5
- `results/semantic_filtering_results_p0.05.json` - Fixed p-value cutoff (p ≤ 0.05)
- `results/semantic_filtering_results_p0.1.json` - Fixed p-value cutoff (p ≤ 0.10)
- `results/semantic_filtering_results_top3.json` - Top-3 by margin
- `results/semantic_filtering_results_top5.json` - Top-5 by margin
- `results/semantic_filtering_results_top10.json` - Top-10 by margin
- `results/semantic_filtering_results_global_q0.1.json` - Global BH-FDR with q=0.1
- `results/semantic_filtering_results_global_q0.2.json` - Global BH-FDR with q=0.2
- `results/semantic_filtering_results_global_q0.3.json` - Global BH-FDR with q=0.3

### Full Dataset Results (158 images)
- `results/semantic_filtering_results_q0.3_full.json` - BH-FDR q=0.3 (full dataset)
- `results/semantic_filtering_results_q0.5_full.json` - BH-FDR q=0.5 (full dataset)
- `results/semantic_filtering_results_p0.1_full.json` - Fixed p=0.1 (full dataset)
- `results/semantic_filtering_results_top5_full.json` - Top-5 (full dataset)

## Running Experiments

### Single Experiment

```bash
python experiments/scripts/experiments/phase5/run_semantic_filtering.py \
    --selection-method bh_fdr \
    --q-fdr 0.3 \
    --results-filename semantic_filtering_results_q0.3.json \
    --output-dir experiments/results/phase5_semantic_filtering
```

### Selected Ablations (Full Dataset)

Run the 4 selected methods sequentially:

```bash
python experiments/scripts/experiments/phase5/run_selected_ablations.py
```

Or in background (allows SSH disconnect):

```bash
nohup python experiments/scripts/experiments/phase5/run_selected_ablations.py \
    > experiments/results/phase5_semantic_filtering/selected_ablations.log 2>&1 &
```

### All Ablations (Test Mode)

```bash
# Test mode (5 images)
python experiments/scripts/experiments/phase5/run_ablations.py --test-mode

# Full dataset
python experiments/scripts/experiments/phase5/run_ablations.py

# Background mode
python experiments/scripts/experiments/phase5/run_ablations.py --background
```

## Selection Methods

1. **bh_fdr**: Per-image Benjamini-Hochberg FDR control
2. **fixed_p**: Fixed p-value cutoff per image
3. **top_k**: Top-k detections by semantic margin
4. **global_bh**: Global BH-FDR across all candidates

## Monitoring Background Runs

### Check Process Status
```bash
ps aux | grep run_semantic_filtering | grep -v grep
```

### Monitor Logs
```bash
tail -f experiments/results/phase5_semantic_filtering/selected_ablations.log
```

### Check Results
```bash
ls -lht experiments/results/phase5_semantic_filtering/results/*_full.json
```

## Generating Visualizations

### Individual Method Plots
```bash
python experiments/scripts/experiments/phase5/visualize_results.py \
    --results-file results/semantic_filtering_results_q0.3_full.json \
    --output-dir experiments/results/phase5_semantic_filtering
```

### Comparison Plots (All Methods)
```bash
python experiments/scripts/experiments/phase5/visualize_comparison.py \
    --results-dir experiments/results/phase5_semantic_filtering/results \
    --compute-metrics
```

### All Plots
```bash
python experiments/scripts/experiments/phase5/generate_all_plots.py \
    --results-dir experiments/results/phase5_semantic_filtering/results \
    --output-dir experiments/results/phase5_semantic_filtering/plots \
    --compute-metrics
```

## Configuration

- **Model:** YOLO World
- **Confidence floor:** 0.01
- **Background samples (K_bg):** 128 per image
- **Crop scales:** [0.0, 0.2]
- **Batch size:** 15 (YOLO World)

## Notes

- All results use the same base configuration
- Only the selection method and parameters vary
- JSON files are stored in `results/` subdirectory
- Plots are generated in the `plots/` subdirectory
- Test results (5 images) use standard filenames
- Full dataset results use `_full.json` suffix

## Test Results Summary

See `ABLATION_RESULTS.md` for detailed test mode results (5 images) comparing all methods.
