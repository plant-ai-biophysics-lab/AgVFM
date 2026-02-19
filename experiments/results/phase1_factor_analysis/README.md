# Phase 1 Factor Analysis Results

## Directory Structure

```
phase1_factor_analysis/
├── yolo_world_factor_analysis.json    # Main results file (YOLO World)
├── sam3_factor_analysis.json          # Main results file (SAM3, if run)
├── logs/                              # All log files
│   ├── run_YYYYMMDD_HHMMSS.log       # Execution logs
│   └── phase1_factor_analysis_*.log  # Detailed experiment logs
├── plots/                             # Generated visualizations
│   ├── *_all_metrics.png             # All metrics (mAP, F1, Precision, Recall)
│   ├── *_factor_contributions_*.png # Individual metric plots
│   └── factor_comparison_*.png        # Model comparison plots
└── archive/                           # Archived old results
    └── YYYYMMDD_HHMMSS/              # Timestamped archives
```

## Results Files

- **yolo_world_factor_analysis.json**: Complete results for YOLO World model
  - Contains all 34 configurations (baseline + one-factor-at-a-time variations)
  - Metrics: mAP, F1, Precision, Recall at IoU thresholds 0.3 and 0.5
  - Each configuration includes: components, prompt, and evaluation metrics

## Logs

All execution logs are stored in the `logs/` directory:
- `run_*.log`: Main execution log with progress updates
- `phase1_factor_analysis_*.log`: Detailed experiment logs from the logging system

## Visualizations

Run the visualization script to generate plots:
```bash
python experiments/scripts/visualization/visualize_phase1_results.py
```

This generates:
- All-metrics plots (2x2 grid: mAP, F1, Precision, Recall)
- Individual metric plots
- Model comparison plots (if both models are run)
