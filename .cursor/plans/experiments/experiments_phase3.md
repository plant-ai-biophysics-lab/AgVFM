# Phase 3: Confidence Threshold Analysis

> **Status:** 🔄 In Progress (YOLO World: ✅ Complete, SAM3: ⏳ Running)  
> **Objective:** Threshold sensitivity analysis for zero-shot deployment  
> **Details:** [approach_phase3.md](../approach/approach_phase3.md)

## Progress Tracking

### YOLO World - Confidence Sweeps
- **Status:** ✅ Complete (3/3 configs, 18 thresholds each)
- **Configs tested:**
  - `C2_not_bud_calyx`: "a single yellow cowpea flower with open petals, not a bud, not the green calyx" (mAP@0.5: 0.4123)
  - `C1`: "a single yellow cowpea flower with open petals, not a bud, not the green calyx, not a leaf" (mAP@0.5: 0.3830)
  - `C2_not_calyx`: "a single yellow cowpea flower with open petals, not the green calyx" (mAP@0.5: 0.3807)
- **Thresholds:** 0.05 to 0.90 in steps of 0.05 (18 values)
- **Test set:** Full (178 images: 158 test + 20 dev)
- **Results:** `experiments/results/phase3_confidence_sweeps/yolo_world_confidence_sweeps.json`
- **Plots:** `experiments/results/phase3_confidence_sweeps/plots/yolo_world_*.png` (8 plots generated)
- **Key findings:**
  - Best F1 thresholds: 0.30-0.45 (varies by config)
  - Low sensitivity (robust to threshold choice)
  - conf=0.50 matches GT density (~6.4 predictions/image)

### SAM3 - Confidence Sweeps
- **Status:** ⏳ In Progress
- **Configs to test:**
  - `C2_not_bud`: "a single yellow cowpea flower with open petals, not a bud" (mAP@0.5: 0.5407)
  - `C_color_taxonomy_anatomy_tiny`: "a tiny yellow cowpea flower with open petals" (mAP@0.5: 0.5364)
  - `C2`: "a single yellow cowpea flower with open petals" (mAP@0.5: 0.4925)
- **Thresholds:** 0.05 to 0.90 in steps of 0.05 (18 values)
- **Test set:** Full (178 images: 158 test + 20 dev)
- **Results:** `experiments/results/phase3_confidence_sweeps/sam3_confidence_sweeps.json`
- **Progress:** Running in background

## Quick Stats

- **Test set:** Full test set (178 images: 158 test + 20 dev)
- **Threshold range:** 0.05 to 0.90 (18 values, step 0.05)
- **Total evaluations per model:** 3 configs × 18 thresholds = 54 evaluations
- **Expected time:** ~2-3 hours per model (depends on batch size and model speed)
- **Results location:** `experiments/results/phase3_confidence_sweeps/`
- **Visualization:** `experiments/scripts/visualization/visualize_phase3_results.py`

## Configurations

**Selected from Phase 2 top performers (by mAP@0.5):**

**YOLO World (top 3):**
1. `C2_not_bud_calyx` - Partial negation (bud+calyx) - mAP@0.5: 0.4123
2. `C1` - Full negation (bud+calyx+leaf) - mAP@0.5: 0.3830
3. `C2_not_calyx` - Single negation (calyx) - mAP@0.5: 0.3807

**SAM3 (top 3):**
1. `C2_not_bud` - Single negation (bud) - mAP@0.5: 0.5407
2. `C_color_taxonomy_anatomy_tiny` - Size variant - mAP@0.5: 0.5364
3. `C2` - Four-component baseline - mAP@0.5: 0.4925

## Implementation Details

### Runner Script
- **Location:** `experiments/scripts/experiments/phase3/run_confidence_sweeps.py`
- **Usage:** `python experiments/scripts/experiments/phase3/run_confidence_sweeps.py --model {yolo_world|sam3|all} --full-test-set`
- **Features:**
  - Incremental saving after each threshold (not just after full sweep)
  - Resume functionality (skips completed thresholds automatically)
  - Progress tracking and logging
  - Model-specific batch sizes (15 for YOLO World, 1 for SAM3)

### Visualization Script
- **Location:** `experiments/scripts/visualization/visualize_phase3_results.py`
- **Usage:** `python experiments/scripts/visualization/visualize_phase3_results.py --model {yolo_world|sam3|all}`
- **Output:** 8 plots per model saved to `experiments/results/phase3_confidence_sweeps/plots/`

### Results Format
- JSON files with structure: `{"results": {config_name: {conf_thresholds: [...], results_by_conf: {...}}}}`
- Each threshold result includes: metrics (mAP, F1, P, R), counting metrics, total_predictions
- Results saved incrementally to prevent data loss

## Notes

- **Incremental saving:** Results saved after each threshold completes (not just after full sweep)
- **Resume capability:** Script automatically skips completed thresholds when resuming
- **Logging:** Model-specific log files in `logs/` directory with timestamps
- **Error handling:** Corrupted JSON files are backed up and experiment restarts cleanly
