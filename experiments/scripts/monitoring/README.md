# Monitoring Scripts

Scripts for checking experiment status and progress.

## Scripts

### `check_experiment_progress.sh`

Comprehensive progress checker showing:
- Whether experiment is running
- Progress (X/34 configurations)
- Test set info
- Recent log activity
- Monitor commands

**Usage:**
```bash
bash experiments/scripts/monitoring/check_experiment_progress.sh
```

### `check_experiment_status.sh`

Quick status check showing:
- Process status
- Latest log output
- Results file sizes and counts
- Log file listing

**Usage:**
```bash
bash experiments/scripts/monitoring/check_experiment_status.sh
```

## Manual Monitoring

### Watch log file
```bash
tail -f experiments/results/phase1_factor_analysis/logs/phase1_factor_analysis_*.log
```

### Check results count
```bash
jq '.results | keys | length' experiments/results/phase1_factor_analysis/yolo_world_factor_analysis.json
```

### Check process
```bash
ps aux | grep run_factor_analysis
```
