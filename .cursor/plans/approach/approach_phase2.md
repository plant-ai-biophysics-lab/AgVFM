# Phase 2: Combination Tests & Absorber Architecture

> **Objective:** Systematically combine best components from Phase 1, test negation strategies (text vs absorber architecture), and optimize for F1 rather than just mAP

## Experimental Design

### Overview

Phase 2 builds on Phase 1 findings by:
1. **Combining best components** from factor analysis
2. **Testing negation strategies** - text negation vs absorber architecture
3. **Multi-class assembly** - testing if detecting multiple classes helps
4. **Absorber architecture** - detect confusers as separate classes, filter to target

### Test Configurations

#### 2.1 Systematic Prompt Combinations

**Approach:** Combine best-performing components from Phase 1 into systematic combinations.

**Configurations to test:**

1. **Baseline controls:**
   - `C3`: `"a cowpea flower"` (species baseline)

2. **Two-component combinations:**
   - Color + Taxonomy: `"a yellow cowpea flower"`
   - Color + Anatomy: `"a yellow flower with open petals"`
   - Color + Grammar: `"a single yellow flower"`
   - Taxonomy + Anatomy: `"a cowpea flower with open petals"`
   - Taxonomy + Grammar: `"a single cowpea flower"`
   - Anatomy + Grammar: `"a single flower with open petals"`

3. **Three-component combinations:**
   - Color + Taxonomy + Anatomy: `"a yellow cowpea flower with open petals"`
   - Color + Taxonomy + Grammar: `"a single yellow cowpea flower"`
   - Color + Anatomy + Grammar: `"a single yellow flower with open petals"`
   - Taxonomy + Anatomy + Grammar: `"a single cowpea flower with open petals"`

4. **Four-component combination (best without negation):**
   - `C2`: `"a single yellow cowpea flower with open petals"`

5. **Kitchen sink (all components with text negation):**
   - `C1`: `"a single yellow cowpea flower with open petals, not a bud, not the green calyx, not a leaf"`

**Expected findings:**
- Components interact super-additively (sum of parts < whole)
- Text negation boosts recall but hurts precision
- Best F1 may differ from best mAP

#### 2.2 Negation Strategy Comparison

**Objective:** Compare text negation vs absorber architecture for handling confusers.

**Text Negation Configs:**
- `C1`: Full text negation (kitchen sink)
- `C1_partial`: Partial negation variants (test which clauses matter)

**Absorber Architecture Configs:**
- **H3b:** Bud + calyx absorbers
  - Target: `"a single yellow cowpea flower with open petals"`
  - Absorbers: `["a cowpea flower bud", "a green calyx"]`
  - Filter: Keep only target class (index 0)
  
- **H2a:** Leaf + stem absorbers
  - Target: `"a single yellow cowpea flower with open petals"`
  - Absorbers: `["a leaf", "a stem"]`
  - Filter: Keep only target class (index 0)

- **H3c:** Bud + calyx + leaf absorbers
  - Target: `"a single yellow cowpea flower with open petals"`
  - Absorbers: `["a cowpea flower bud", "a green calyx", "a leaf"]`
  - Filter: Keep only target class (index 0)

- **H5:** All absorbers (bud, calyx, leaf, stem, soil)
  - Target: `"a single yellow cowpea flower with open petals"`
  - Absorbers: `["a cowpea flower bud", "a green calyx", "a leaf", "a stem", "soil"]`
  - Filter: Keep only target class (index 0)

**Key comparison:**
- Text negation (`C1`) vs Absorber (`H3b`, `H2a`) - same target prompt
- Measure: F1, Precision, Recall, False Positives
- **Expected:** Absorbers should have higher precision, fewer FPs, similar or better F1

**Note:** Absorber architecture only works for YOLO World (multi-class detection). SAM3 will only test text negation variants.

#### 2.3 Multi-Class Assembly Tests

**Objective:** Test if detecting multiple classes simultaneously helps (vs single-class).

**Configurations:**
1. **Single-class baseline:** `"a single yellow cowpea flower with open petals"`

2. **Two-class sets:**
   - Flower + Bud: `["a single yellow cowpea flower with open petals", "a cowpea flower bud"]`
   - Yellow + White: `["a yellow cowpea flower", "a white cowpea flower"]`

3. **Three-class sets:**
   - Flower + Bud + Calyx: `["a single yellow cowpea flower with open petals", "a cowpea flower bud", "a green calyx"]`

**Evaluation:** 
- Match all non-background detections to single GT class
- Compare to single-class performance

**Expected finding:** Multi-class should not help (from previous work: single-class wins on mAP)

## Evaluation Protocol

### Dataset
- **Test set:** Full holdout test set (138 images, 839 ground truth flowers)
- **No sampling:** Evaluate on all images

### Metrics to Compute

For each configuration:

#### Detection Metrics (at multiple IoU thresholds)
- **mAP@0.3** - Relaxed IoU
- **mAP@0.5** - Standard IoU
- **mAP@0.5:0.95** - COCO-style average
- **Precision@0.5** - Critical for comparing negation strategies
- **Recall@0.5** - Critical for comparing negation strategies
- **F1@0.5** - Primary optimization target (not just mAP)

#### Counting Metrics
- **R², RMSE, MAE, MAPE, Slope, Intercept, Bias**

### Model-Specific Settings

#### YOLO World
- **Weights:** `yolov8x-worldv2.pt`
- **Confidence threshold:** 0.1
- **NMS IoU:** 0.5 (standard)
- **Single-class:** Use `predict()` method
- **Multi-class/Absorber:** Use `predict_multi_class()` method with `target_indices` filtering

#### SAM3
- **Model:** `facebook/sam3` (Hugging Face)
- **Confidence threshold:** 0.1
- **Note:** SAM3 does not support multi-class detection, so absorber architecture tests are YOLO World only
- **Text negation:** Test same prompts as YOLO World for comparison

## Analysis Plan

### 2.1 Combination Analysis

1. **Component interaction effects:**
   - Compare actual performance vs additive prediction
   - Identify super-additive combinations
   - Visualize component stacking (cumulative mAP/F1 as components added)

2. **Precision-Recall tradeoff:**
   - Plot P-R scatter with F1 iso-lines
   - Identify mAP-optimal vs F1-optimal configs
   - Document that mAP ≠ F1 optimum

### 2.2 Negation Strategy Comparison

1. **Text negation vs Absorber:**
   - Side-by-side comparison table
   - Key metrics: F1, Precision, Recall, False Positives
   - **Expected:** Absorbers should have:
     - Similar or better F1
     - Higher precision
     - Fewer false positives
     - Similar recall

2. **Absorber depth analysis:**
   - Compare 1 absorber vs 2 vs 3 vs 5
   - Identify optimal absorber set
   - Precision vs recall tradeoff

### 2.3 Multi-Class Analysis

1. **Single vs Multi-class:**
   - Compare mAP, F1, P, R
   - **Expected:** Single-class should win

2. **Class dilution effect:**
   - Measure if more classes hurt performance
   - Analyze duplicate FPs from class-aware NMS

## Expected Findings (from previous work)

### Combination Effects
- **Super-additive interactions:** Components together > sum of parts
- **Best combination:** `"a single yellow cowpea flower with open petals"` (C2) - best F1 without negation
- **Kitchen sink:** `C1` has highest mAP but lower F1 due to precision collapse

### Negation Strategies
- **Text negation:** Boosts recall (+106% FP) but hurts precision (-29%)
- **Absorber architecture:** Replaces text negation, +32% F1, -50% FP
- **Best absorber:** H3b (bud+calyx) or H2a (leaf+stem) depending on dataset

### Multi-Class
- **Single-class wins:** More classes monotonically hurt mAP
- **Class dilution:** Each additional class dilutes attention

## Implementation Details

### Code Structure

```python
from agvfm.config.experiments import COMBINATION_CONFIGS, ABSORBER_CONFIGS
from agvfm.experiments import Evaluator
from agvfm.models import YOLOWorldModel, SAM3Model

# Combination tests
for config in COMBINATION_CONFIGS:
    # YOLO World
    yolo_results = yolo_evaluator.evaluate_prompt(
        prompt=config.prompt,
        iou_thresholds=[0.3, 0.5],
    )
    
    # SAM3
    sam3_results = sam3_evaluator.evaluate_prompt(
        prompt=config.prompt,
        iou_thresholds=[0.3, 0.5],
    )

# Absorber tests (YOLO World only)
for config in ABSORBER_CONFIGS:
    yolo_results = yolo_evaluator.evaluate_prompt(
        prompt=config.prompt,
        absorber_classes=config.absorber_classes,
        target_indices=config.target_indices,
        iou_thresholds=[0.3, 0.5],
    )
```

### Results Storage

Save results to: `notebooks/results/phase2_combinations/`

**Per model:**
- `yolo_world_combinations.json` - All combination results
- `yolo_world_absorbers.json` - Absorber architecture results
- `sam3_combinations.json` - SAM3 combination results (no absorbers)

**Analysis:**
- `negation_strategy_comparison.json` - Text vs absorber comparison
- `component_interactions.json` - Interaction effects analysis
- `precision_recall_analysis.json` - P-R tradeoff analysis

### Output Format

Each result entry:
```json
{
  "config_name": "C2",
  "prompt": "a single yellow cowpea flower with open petals",
  "absorber_classes": null,
  "target_indices": null,
  "model": "YOLO World",
  "metrics_by_iou": {...},
  "map_coco": 0.xxx,
  "counting": {...}
}
```

## Deliverables

1. **Combination performance table:** All configs ranked by F1 and mAP
2. **Negation strategy comparison:** Text vs absorber side-by-side
3. **P-R scatter plot:** With F1 iso-lines, showing tradeoffs
4. **Component interaction plot:** Showing super-additive effects
5. **Best configs per model:** Optimal prompts for YOLO World vs SAM3
6. **Prompt engineering rules (updated):** What combinations work for each model

## Success Criteria

Phase 2 is complete when:
- ✅ All combination configs tested for both models
- ✅ Absorber architecture tested for YOLO World
- ✅ Negation strategy comparison completed
- ✅ Multi-class tests completed
- ✅ Best F1 configs identified per model
- ✅ Precision-recall tradeoff documented
- ✅ Results saved in structured format
- ✅ Key differences between models documented

## Next Steps (Phase 3)

After Phase 2, we will:
- Take best configs from Phase 2
- Sweep confidence thresholds to find optimal values
- Analyze model sensitivity to confidence threshold
- Understand zero-shot generalizability implications
