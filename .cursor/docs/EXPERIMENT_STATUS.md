# Experiment Status

**Last Updated:** $(date)

## Phase 1: Factor Analysis

### Status: 🟢 RUNNING

**Started:** Background process running  
**Model:** YOLO World  
**Dataset:** Holdout test set (138 images)

### Quick Status Check

```bash
# Check experiment status
bash notebooks/scripts/check_experiment_status.sh

# Monitor live progress
tail -f notebooks/results/phase1_factor_analysis/run.log

# Check results
ls -lh notebooks/results/phase1_factor_analysis/*.json
jq '.results | length' notebooks/results/phase1_factor_analysis/yolo_world_factor_analysis.json
```

### Results Location

- **Results:** `notebooks/results/phase1_factor_analysis/`
- **Logs:** `notebooks/results/phase1_factor_analysis/logs/`
- **Main log:** `notebooks/results/phase1_factor_analysis/run.log`

### Features

✅ **Incremental saving** - Results saved after each config (can resume if interrupted)  
✅ **Auto-resume** - Skips already-completed configs  
✅ **Comprehensive logging** - Both file and console output  
✅ **Background execution** - Safe to disconnect SSH  

### Package Organization

All reusable code is in the `agvfm` package:

- `agvfm.data` - Data loading and sampling utilities
- `agvfm.models` - Model wrappers (YOLO World, SAM3)
- `agvfm.evaluation` - Metrics computation
- `agvfm.experiments` - Experiment runners
- `agvfm.visualization` - Plotting functions
- `agvfm.utils` - Logging and utilities

### Next Steps

1. **Monitor progress** - Use `check_experiment_status.sh` or check log files
2. **After completion** - Run visualization script:
   ```bash
   python notebooks/scripts/visualize_phase1_results.py
   ```
3. **Review results** - Check `.cursor/plans/experiments/experiments_phase1.md`
