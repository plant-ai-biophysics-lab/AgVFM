# Phase 3: Confidence Threshold Analysis

> **Objective:** Systematically analyze how model performance changes with confidence threshold to understand zero-shot generalizability and provide unlabeled operating rules for deployment (no label-based threshold selection)

## Experimental Design

### Overview

Confidence threshold is a critical hyperparameter for zero-shot deployment. This phase:
1. **Sweeps confidence thresholds** across a range for each model
2. **Analyzes sensitivity** - how much performance changes with threshold
3. **Reports performance at pre-declared thresholds** (no label-based selection)
4. **Provides unlabeled operating rules** (fixed thresholds, top-K, percentile rules)
5. **Diagnoses FP sources** and confidence distributions
6. **Assesses generalizability** - threshold stability across configs

### Why This Matters

- **Zero-shot deployment:** No labeled data to tune thresholds
- **Model differences:** YOLO World and SAM3 may have different confidence distributions
- **Config differences:** Best threshold may vary by prompt/architecture
- **Generalizability:** Stable thresholds across configs indicate robust models

### Test Configurations

#### 3.1 Top 3 Configs by mAP@0.5 from Phase 2

Test confidence sweeps on top performers:

1. **YOLO World top 3 (by mAP@0.5):**
   - `C2_not_bud_calyx`: `"a single yellow cowpea flower with open petals, not a bud, not the green calyx"` (mAP: 0.4123)
   - `C1`: `"a single yellow cowpea flower with open petals, not a bud, not the green calyx, not a leaf"` (mAP: 0.3830)
   - `C2_not_calyx`: `"a single yellow cowpea flower with open petals, not the green calyx"` (mAP: 0.3807)

2. **SAM3 top 3 (by mAP@0.5):**
   - `C2_not_bud`: `"a single yellow cowpea flower with open petals, not a bud"` (mAP: 0.5407)
   - `C_color_taxonomy_anatomy_tiny`: `"a tiny yellow cowpea flower with open petals"` (mAP: 0.5364)
   - `C2`: `"a single yellow cowpea flower with open petals"` (mAP: 0.4925)

#### 3.2 Confidence Threshold Range

**Sweep range:** 0.05 to 0.90 in steps of 0.05
- **Total thresholds:** 18 values (0.05, 0.10, 0.15, ..., 0.85, 0.90)
- **Rationale:** Covers full range from low (high recall) to high (high precision) thresholds
- **Note:** Fine-grained steps (0.01) around optimal can be added if needed for specific analysis

**Rationale:**
- Low thresholds (0.05-0.20): High recall, many false positives
- Medium thresholds (0.20-0.50): Balance
- High thresholds (0.50-0.95): High precision, many false negatives

### Evaluation Protocol

#### Dataset
- **Test set:** Full test set (178 images: 158 test + 20 dev) - consistent with Phase 1 and Phase 2
- **Same images:** Use identical images across all thresholds for fair comparison
- **Image paths:** Automatically detected from `_data/` directory or via `--data-root` argument

#### Metrics to Compute

For each confidence threshold:

**Detection Metrics (at IoU=0.5):**
- **mAP@0.5** - Primary metric (measures ranking quality)
- **F1@0.5** - Optimization target
- **Precision@0.5** - How precision changes with threshold
- **Recall@0.5** - How recall changes with threshold
- **Total Predictions** - Count of detections above threshold
- **False Positives** - Count of FPs
- **False Negatives** - Count of FNs
- **Predictions per Image** - Distribution of detection counts

**Counting Metrics:**
- **R², RMSE, MAE, MAPE, Slope, Intercept, Bias**
- **Note:** Counting may be sensitive to threshold (affects total count)

#### Model-Specific Settings

**YOLO World:**
- **Weights:** `yolov8x-worldv2.pt` (default: `model_weights/yolo_world/yolov8x-worldv2.pt`)
- **NMS IoU:** 0.5 (standard)
- **Confidence threshold:** Varied (0.05 to 0.90, step 0.05)
- **Batch size:** 15 (efficient batch processing)
- **Inference:** Run once per image, filter by confidence threshold post-inference

**SAM3:**
- **Model:** `facebook/sam3` (Hugging Face)
- **Confidence threshold:** Varied (0.05 to 0.90, step 0.05)
- **Batch size:** 1 (sequential processing due to batching limitations)
- **Note:** SAM3 outputs confidence scores; filter masks/boxes by threshold

## Analysis Plan

### 3.1 Performance Curves

For each config, plot:
1. **mAP vs Confidence Threshold** - Shows ranking quality across thresholds
2. **F1 vs Confidence Threshold** - Shows F1-optimal operating points
3. **Precision vs Confidence Threshold** - How precision changes
4. **Recall vs Confidence Threshold** - How recall changes
5. **Total Predictions vs Confidence Threshold** - Detection volume
6. **Precision-Recall Curves** - Full PR curves at different thresholds

**Key questions:**
- How sensitive is performance to threshold choice?
- Are there clear "sweet spots" or gradual changes?
- How do mAP and point metrics (P/R/F1) relate at different thresholds?

### 3.2 Threshold Sensitivity Analysis

**Sensitivity metric:** How much does performance change with ±0.1 threshold change?

For each config:
- **Performance at pre-declared thresholds:** Report F1/P/R at 0.05, 0.10, 0.20, 0.30, 0.50
- **Sensitivity windows:** F1 change with ±0.1 threshold around key points
- **Robustness score:** How much F1 drops with suboptimal threshold

**Interpretation:**
- **Low sensitivity:** Robust to threshold choice (good for zero-shot)
- **High sensitivity:** Requires careful threshold tuning (problematic for zero-shot)
- **Note:** We report performance at fixed thresholds, not "optimal" thresholds selected from test labels

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
3. **Confidence quantiles:** 25th, 50th, 75th, 90th, 95th percentiles
4. **Per-image statistics:** Mean, median, max confidence per image

### 3.6 Unlabeled Operating Rules

**Provide deployment rules that don't require labels:**

1. **Fixed Threshold Options:**
   - Report performance at pre-declared thresholds: 0.05, 0.10, 0.20, 0.30, 0.50
   - Let users choose based on their precision/recall needs

2. **Top-K Rules:**
   - Keep top K detections per image (K = 3, 5, 10)
   - Report performance for each K value

3. **Percentile Rules:**
   - Keep top 50th, 75th, 90th percentile of confidences
   - Report performance for each percentile

4. **Elbow Detection (unlabeled):**
   - Find natural breakpoints in confidence distributions
   - Use unlabeled methods (e.g., gap statistic, silhouette score)

### 3.7 False Positive Source Diagnostics

**Analyze where FPs come from:**
1. **Duplicate detections:** How many FPs are near-duplicates of other detections?
2. **Near-miss analysis:** How many FPs are close to GT (IoU 0.3-0.5)?
3. **Background confusion:** How many FPs are clearly background?
4. **Per-image FP distribution:** Are FPs concentrated in certain images?
5. **IoU-to-nearest-GT histograms:** How close are FPs to actual objects?

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

Save results to: `experiments/results/phase3_confidence_sweeps/`

**Per config:**
- `yolo_world_C2_not_bud_calyx_confidence_sweep.json`
- `yolo_world_C1_confidence_sweep.json`
- `yolo_world_C2_not_calyx_confidence_sweep.json`
- `sam3_C2_not_bud_confidence_sweep.json`
- `sam3_C_color_taxonomy_anatomy_tiny_confidence_sweep.json`
- `sam3_C2_confidence_sweep.json`

**Analysis:**
- `confidence_sensitivity_analysis.json` - Sensitivity metrics
- `unlabeled_operating_rules.json` - Performance at fixed thresholds, top-K, percentiles
- `confidence_distributions.json` - Distribution statistics
- `fp_source_analysis.json` - False positive diagnostics
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
  "performance_at_fixed_thresholds": {
    "0.05": {"f1": 0.xx, "precision": 0.xx, "recall": 0.xx},
    "0.10": {"f1": 0.xx, "precision": 0.xx, "recall": 0.xx},
    "0.20": {"f1": 0.xx, "precision": 0.xx, "recall": 0.xx},
    "0.30": {"f1": 0.xx, "precision": 0.xx, "recall": 0.xx},
    "0.50": {"f1": 0.xx, "precision": 0.xx, "recall": 0.xx}
  },
  "unlabeled_operating_rules": {
    "top_k": {"3": {...}, "5": {...}, "10": {...}},
    "percentiles": {"50th": {...}, "75th": {...}, "90th": {...}}
  }
}
```

## Deliverables

1. **Performance curves:** mAP, F1, P, R vs confidence threshold for each config
2. **Pre-declared threshold table:** Performance at fixed thresholds (0.05, 0.10, 0.20, 0.30, 0.50)
3. **Sensitivity analysis:** How robust each config is to threshold choice
4. **Unlabeled operating rules:** Top-K and percentile-based rules with performance metrics
5. **Model comparison:** YOLO World vs SAM3 threshold behavior
6. **Confidence distribution plots:** TP vs FP confidence distributions, quantiles
7. **FP source diagnostics:** Analysis of where false positives come from
8. **Deployment recommendations:** Zero-shot operating rules (no label-based threshold selection)

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

## Results Summary

### YOLO World Phase 3 Results
- **Optimal F1 thresholds:** 0.30-0.45 (varies by config)
- **Best mAP thresholds:** All at conf=0.05 (low threshold, high recall)
- **Sensitivity:** Low (<0.02 F1 change per ±0.05 threshold) - robust to threshold choice
- **At conf=0.50:** F1: 0.36-0.40, mAP: 0.40-0.41, P: 0.48-0.51, R: 0.42-0.49
- **Detection volume at conf=0.50:** ~6.6 predictions/image (matches GT density ~6.4/image)
- **Key finding:** All configs show low threshold sensitivity, making them robust for zero-shot deployment

### SAM3 Phase 3 Results
- **Optimal F1 thresholds:** 0.40-0.45 (similar to YOLO World)
- **Best mAP thresholds:** 0.05-0.15 (low thresholds, high recall)
- **Sensitivity:** Moderate (0.10-0.12 F1 change per ±0.05 threshold) - more sensitive than YOLO World
- **At conf=0.50:** F1: 0.51-0.58, mAP: 0.34-0.43, P: 0.59-0.62, R: 0.43-0.57
- **Detection volume at conf=0.50:** 4.4-6.1 predictions/image (varies by config)
- **Key finding:** Higher F1 than YOLO World at optimal thresholds, but more sensitive to threshold choice

### Model Comparison
- **YOLO World:** Lower sensitivity (robust), better recall at low thresholds (R≈0.70), balanced F1
- **SAM3:** Higher F1 at optimal thresholds (0.51-0.58 vs 0.36-0.40), higher precision at high thresholds (0.72-0.75 vs 0.48-0.51), but more sensitive to threshold choice
- **Threshold selection:** YOLO World more forgiving (low sensitivity), SAM3 requires more careful threshold selection

## Success Criteria

Phase 3 is complete when:
- ✅ Confidence sweeps completed for top 3 configs per model (6 total)
- ✅ Performance curves generated and analyzed
- ✅ Performance at pre-declared thresholds reported (no label-based selection)
- ✅ Sensitivity analysis completed
- ✅ Model comparison documented
- ✅ Results saved in structured format
- ⏳ Unlabeled operating rules (top-K, percentiles) - future work
- ⏳ Confidence distributions analyzed - future work
- ⏳ FP source diagnostics - future work

## Next Steps (Future Work)

After Phase 3, potential next steps:
- **Grounding gap analysis:** Understand why models fail
- **Cross-species validation:** Test on additional species
- **Real-world deployment:** Apply optimal thresholds in field settings
- **Model improvements:** Use findings to guide future model development
