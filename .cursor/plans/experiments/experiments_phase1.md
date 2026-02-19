# Phase 1: Factor Analysis Experiments

> **Status:** ✅ Complete (Both YOLO World and SAM3)  
> **Objective:** One-factor-at-a-time (OFAT) analysis across 7 axes for YOLO World and SAM3

## Progress Tracking

### YOLO World
- **Status:** ✅ Complete
- **Configs tested:** 34 / 34
- **Test set:** Full (158 images)
- **Last updated:** 2026-02-18
- **Results:** `experiments/results/phase1_factor_analysis/yolo_world_factor_analysis.json`
- **Plots:** `experiments/results/phase1_factor_analysis/plots/`

### SAM3
- **Status:** ✅ Complete
- **Configs tested:** 34 / 34
- **Test set:** Full (158 images)
- **Last updated:** 2026-02-19
- **Results:** `experiments/results/phase1_factor_analysis/sam3_factor_analysis.json`
- **Plots:** `experiments/results/phase1_factor_analysis/plots/`

## Quick Stats

- **Total configs per model:** 34 (1 baseline + 33 variations across 7 axes)
- **Test set:** Full test set (158 images, includes dev images)
- **Expected time:** ~15-20 minutes per model with batch_size=15
- **Results location:** `experiments/results/phase1_factor_analysis/`

## Configurations Tested

### Baseline (1 config)
- `baseline`: `"a flower"` (all axes at baseline: taxonomy="flower", others="")

### Taxonomy Axis (7 configs)
- `taxonomy:cowpea flower`: `"a cowpea flower"`
- `taxonomy:bean flower`: `"a bean flower"`
- `taxonomy:pea flower`: `"a pea flower"`
- `taxonomy:legume flower`: `"a legume flower"`
- `taxonomy:black-eyed pea flower`: `"a black-eyed pea flower"`
- `taxonomy:vigna unguiculata flower`: `"a vigna unguiculata flower"`
- `taxonomy:crop flower`: `"a crop flower"`

### Color Axis (4 configs)
- `color:yellow`: `"a yellow flower"`
- `color:white`: `"a white flower"`
- `color:cream`: `"a cream flower"`
- `color:purple`: `"a purple flower"`

### Size Axis (3 configs)
- `size:large`: `"a large flower"`
- `size:small`: `"a small flower"`
- `size:tiny`: `"a tiny flower"`

### Phenology Axis (5 configs)
- `phenology:bud`: `"a flower bud"`
- `phenology:open`: `"a flower open"`
- `phenology:closed bud`: `"a flower closed bud"`
- `phenology:blooming`: `"a flower blooming"`
- `phenology:in bloom`: `"a flower in bloom"`

### Negation Axis (5 configs)
- `negation:not a bud, not the green calyx, not a leaf`: `"a flower not a bud, not the green calyx, not a leaf"`
- `negation:not a bud, not the green calyx`: `"a flower not a bud, not the green calyx"`
- `negation:not a leaf`: `"a flower not a leaf"`
- `negation:not a bud`: `"a flower not a bud"`
- `negation:not the green calyx`: `"a flower not the green calyx"`

### Anatomy Axis (4 configs)
- `anatomy:with open petals`: `"a flower with open petals"`
- `anatomy:with visible petals`: `"a flower with visible petals"`
- `anatomy:with petals and stamens`: `"a flower with petals and stamens"`
- `anatomy:corolla`: `"a flower corolla"`

### Grammar Axis (5 configs)
- `grammar:a single`: `"a single flower"`
- `grammar:`: `"flower"` (bare noun, no article)
- `grammar:a photo of a`: `"a photo of a flower"`
- `grammar:one`: `"one flower"`
- `grammar:close-up of a`: `"close-up of a flower"`

## Results Summary

### YOLO World Results
- ✅ All 34 configurations completed
- Metrics computed: mAP, F1, Precision, Recall at IoU 0.3 and 0.5
- Visualizations generated: All-metrics plot (4 rows), individual metric plots
- Baseline (mAP@0.5): 0.0909
- Key findings:
  - **Best performers:** "cowpea flower" (taxonomy), "yellow" (color), "tiny" (size), "bud" (phenology), "with open petals" (anatomy), "a single" (grammar)
  - **Worst performers:** "vigna unguiculata flower", "crop flower" (taxonomy), "purple" (color), "large" (size), "corolla" (anatomy), "a photo of a", "close-up of a" (grammar)

### SAM3 Results
- ✅ All 34 configurations completed
- Metrics computed: mAP, F1, Precision, Recall at IoU 0.3 and 0.5
- Visualizations generated: All-metrics plot (4 rows), individual metric plots, comparison plots with YOLO World
- Baseline performance: See plots for baseline mAP@0.5 (higher than YOLO World baseline)
- Key findings: (See plots for detailed analysis)
  - SAM3 shows different sensitivity patterns compared to YOLO World
  - Comparison plots available: `factor_comparison_map.png`, `factor_comparison_f1.png`

## Notes

- Running experiments with: `python experiments/scripts/experiments/phase1/run_factor_analysis.py --model yolo_world --full-test-set`
- Can resume interrupted runs (checks existing results)
- Results saved incrementally after each config
- Batch size: 6 for SAM3 (true batch processing), 15 for YOLO World
- Visualization: `python experiments/scripts/visualization/visualize_phase1_results.py`