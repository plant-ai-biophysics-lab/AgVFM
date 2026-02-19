# Phase 3: Confidence Threshold Analysis

> **Objective:** Systematically analyze how model performance changes with confidence threshold to understand zero-shot generalizability and identify optimal thresholds per model/configuration

## Experimental Design

### Overview

Confidence threshold is a critical hyperparameter for zero-shot deployment. This phase:
1. **Sweeps confidence thresholds** across a range for each model
2. **Analyzes sensitivity** - how much performance changes with threshold
3. **Identifies optimal thresholds** per model and configuration
4. **Assesses generalizability** - threshold stability across configs

### Why This Matters

- **Zero-shot deployment:** No labeled data to tune thresholds
- **Model differences:** YOLO World and SAM3 may have different confidence distributions
- **Config differences:** Best threshold may vary by prompt/architecture
- **Generalizability:** Stable thresholds across configs indicate robust models

### Test Configurations

#### 3.1 Best Configs from Phase 2

Test confidence sweeps on:
1. **YOLO World best configs:**
   - `C2`: `"a single yellow cowpea flower with open petals"` (best F1 without negation)
   - `C1`: `"a single yellow cowpea flower with open petals, not a bud, not the green calyx, not a leaf"` (kitchen sink)
   - `H3b`: Absorber architecture (bud+calyx)
   - `H2a`: Absorber architecture (leaf+stem)
   - `C3`: `"a cowpea flower"` (baseline)

2. **SAM3 best configs:**
   - `C2`: `"a single yellow cowpea flower with open petals"` (best from Phase 2)
   - `C3`: `"a cowpea flower"` (baseline)
   - Note: SAM3 may not support all YOLO World configs (e.g., complex negation)

#### 3.2 Confidence Threshold Range

**Sweep range:** 0.05 to 0.95 in steps of 0.05
- **Total thresholds:** 19 values (0.05, 0.10, 0.15, ..., 0.90, 0.95)
- **Fine-grained around optimal:** May need finer steps (0.01) around best threshold

**Rationale:**
- Low thresholds (0.05-0.20): High recall, many false positives
- Medium thresholds (0.20-0.50): Balance
- High thresholds (0.50-0.95): High precision, many false negatives

### Evaluation Protocol

#### Dataset
- **Test set:** Full holdout test set (138 images, 839 ground truth flowers)
- **Same images:** Use identical images across all thresholds for fair comparison

#### Metrics to Compute

For each confidence threshold:

**Detection Metrics (at IoU=0.5):**
- **mAP@0.5** - Primary metric
- **F1@0.5** - Optimization target
- **Precision@0.5** - How precision changes
- **Recall@0.5** - How recall changes
- **Total Predictions** - Count of detections above threshold
- **False Positives** - Count of FPs
- **False Negatives** - Count of FNs

**Counting Metrics:**
- **R², RMSE, MAE, MAPE, Slope, Intercept, Bias**
- **Note:** Counting may be sensitive to threshold (affects total count)

#### Model-Specific Settings

**YOLO World:**
- **Weights:** `yolov8x-worldv2.pt`
- **NMS IoU:** 0.5 (standard)
- **Confidence threshold:** Varied (0.05 to 0.95)
- **Inference:** Run once per image, filter by confidence threshold post-inference

**SAM3:**
- **Model:** `facebook/sam3` (Hugging Face)
- **Confidence threshold:** Varied (0.05 to 0.95)
- **Note:** SAM3 outputs confidence scores; filter masks/boxes by threshold

## Analysis Plan

### 3.1 Performance Curves

For each config, plot:
1. **mAP vs Confidence Threshold** - Identify optimal mAP threshold
2. **F1 vs Confidence Threshold** - Identify optimal F1 threshold
3. **Precision vs Confidence Threshold** - How precision changes
4. **Recall vs Confidence Threshold** - How recall changes
5. **Total Predictions vs Confidence Threshold** - Detection volume

**Key questions:**
- Is optimal threshold the same for mAP vs F1?
- How sensitive is performance to threshold choice?
- Are there clear "sweet spots" or gradual changes?

### 3.2 Threshold Sensitivity Analysis

**Sensitivity metric:** How much does performance change with ±0.1 threshold change?

For each config:
- **Optimal threshold:** Threshold with best F1
- **Sensitivity window:** F1 at optimal ± 0.1 threshold
- **Robustness score:** How much F1 drops with suboptimal threshold

**Interpretation:**
- **Low sensitivity:** Robust to threshold choice (good for zero-shot)
- **High sensitivity:** Requires careful threshold tuning (problematic for zero-shot)

### 3.3 Model Comparison

**Compare YOLO World vs SAM3:**
1. **Optimal thresholds:** Do they differ?
2. **Sensitivity:** Which model is more robust?
3. **Confidence distributions:** Analyze raw confidence scores
4. **Threshold stability:** Do optimal thresholds vary by config?

### 3.4 Configuration Comparison

**Compare across configs:**
1. **Absorber vs non-absorber:** Do absorber configs have different optimal thresholds?
2. **Text negation vs no negation:** How does negation affect threshold sensitivity?
3. **Baseline vs optimized:** Do optimized prompts need different thresholds?

### 3.5 Confidence Distribution Analysis

**Analyze raw confidence scores:**
1. **Distribution shape:** Bimodal (TP vs FP modes)?
2. **TP vs FP separation:** How well do confidences separate true/false positives?
3. **Optimal threshold location:** Where does it fall in the distribution?

**Expected (from previous work):**
- Configs without absorbers: Bimodal distribution, threshold tuning helps (+35% F1 for C1)
- Configs with absorbers: Already near-optimal, less threshold sensitivity

## Implementation Details

### Code Structure

```python
from agvfm.experiments import Evaluator
from agvfm.models import YOLO WorldModel, SAM3Model

# Confidence threshold sweep
conf_thresholds = [0.05, 0.10, 0.15, ..., 0.90, 0.95]

for config in best_configs:
    # YOLO World
    yolo_sweep = yolo_evaluator.evaluate_confidence_sweep(
        prompt=config.prompt,
        conf_thresholds=conf_thresholds,
        iou_threshold=0.5,
        absorber_classes=config.absorber_classes,
        target_indices=config.target_indices,
    )
    
    # SAM3
    sam3_sweep = sam3_evaluator.evaluate_confidence_sweep(
        prompt=config.prompt,
        conf_thresholds=conf_thresholds,
        iou_threshold=0.5,
    )
```

### Results Storage

Save results to: `notebooks/results/phase3_confidence_sweeps/`

**Per config:**
- `yolo_world_C2_confidence_sweep.json`
- `yolo_world_H3b_confidence_sweep.json`
- `sam3_C2_confidence_sweep.json`
- etc.

**Analysis:**
- `confidence_sensitivity_analysis.json` - Sensitivity metrics
- `optimal_thresholds.json` - Optimal thresholds per config
- `model_comparison_confidence.json` - YOLO World vs SAM3 comparison

### Output Format

Each sweep result:
```json
{
  "config_name": "C2",
  "prompt": "a single yellow cowpea flower with open petals",
  "model": "YOLO World",
  "iou_threshold": 0.5,
  "conf_thresholds": [0.05, 0.10, ...],
  "results_by_conf": {
    "0.05": {
      "metrics": {"map": 0.xxx, "f1": 0.xxx, "precision": 0.xxx, "recall": 0.xxx},
      "counting": {...},
      "total_predictions": 1234
    },
    ...
  },
  "optimal_threshold": {
    "f1": 0.50,
    "map": 0.45,
    "conf": 0.25
  }
}
```

## Deliverables

1. **Performance curves:** mAP, F1, P, R vs confidence threshold for each config
2. **Optimal threshold table:** Best threshold per config and metric
3. **Sensitivity analysis:** How robust each config is to threshold choice
4. **Model comparison:** YOLO World vs SAM3 threshold behavior
5. **Confidence distribution plots:** TP vs FP confidence distributions
6. **Recommendations:** Optimal thresholds for zero-shot deployment

## Expected Findings (from previous work)

### Threshold Sensitivity
- **Non-absorber configs:** High sensitivity, threshold tuning helps significantly
  - C1 (kitchen sink): +35% F1 from threshold tuning (optimal ~0.38)
- **Absorber configs:** Low sensitivity, already near-optimal
  - H3b: Minimal improvement from threshold tuning

### Confidence Distributions
- **Bimodal:** TP and FP modes clearly separated
- **Absorber margin:** Better signal than raw confidence for absorber configs

### Model Differences
- **YOLO World:** Well-calibrated confidence scores
- **SAM3:** May have different confidence distribution (to be determined)

## Success Criteria

Phase 3 is complete when:
- ✅ Confidence sweeps completed for all best configs
- ✅ Performance curves generated and analyzed
- ✅ Optimal thresholds identified per config
- ✅ Sensitivity analysis completed
- ✅ Model comparison documented
- ✅ Confidence distributions analyzed
- ✅ Recommendations provided for zero-shot deployment
- ✅ Results saved in structured format

## Next Steps (Future Work)

After Phase 3, potential next steps:
- **Grounding gap analysis:** Understand why models fail
- **Cross-species validation:** Test on additional species
- **Real-world deployment:** Apply optimal thresholds in field settings
- **Model improvements:** Use findings to guide future model development
