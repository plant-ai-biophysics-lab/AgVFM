# Phase 1: Structured Factor Analysis

> **Objective:** One-factor-at-a-time (OFAT) analysis to identify which prompt components matter most for each model architecture (YOLO World vs SAM3)

## Experimental Design

### Factor Axes (7 total)

We systematically vary one factor at a time while keeping all others at baseline. This allows us to isolate the contribution of each component.

**Note:** We test all values from the original Experiment 7 to ensure comprehensive coverage and comparability.

| Axis | Values | Baseline | Purpose |
|------|--------|----------|---------|
| **Taxonomy** | `"cowpea flower"` (species_common), `"bean flower"` (genus_bean), `"pea flower"` (genus_pea), `"legume flower"` (family_common), `"black-eyed pea flower"` (alt_common), `"flower"` (baseline), `"vigna unguiculata flower"` (scientific), `"crop flower"` (generic_crop) | `"flower"` | Test taxonomic specificity (species > genus > family > generic) |
| **Color** | `"yellow"` (best), `"white"`, `"cream"`, `""` (baseline), `"purple"` (worst) | `""` | Test if specifying dominant color helps |
| **Size** | `""` (baseline, best), `"large"`, `"small"`, `"tiny"` (worst) | `""` | Test if explicit size language helps (expected to hurt) |
| **Phenology** | `"bud"` (best), `"open"`, `""` (baseline), `"closed bud"`, `"blooming"`, `"in bloom"` (worst, expected to fail) | `""` | Test developmental stage terms (expected to fail for "in bloom") |
| **Negation** | `"not a bud, not the green calyx, not a leaf"` (best, bud+calyx+leaf), `"not a bud, not the green calyx"` (bud+calyx), `"not a leaf"` (leaf_only), `"not a bud"` (bud_only), `"not the green calyx"` (calyx_only), `""` (baseline) | `""` | Test text negation for confuser exclusion (vs absorber architecture in Phase 2) |
| **Anatomy** | `"with open petals"` (best, open_petals), `"with visible petals"` (visible_petals), `"with petals and stamens"` (petals+stamens), `""` (baseline), `"corolla"` (worst, expected to fail) | `""` | Test positive structural descriptors |
| **Grammar** | `"a single"` (best, single), `"a"` (baseline, article), `""` (bare_noun), `"a photo of a"` (photo_of), `"one"` (one), `"close-up of a"` (closeup_of, worst) | `"a"` | Test article/quantifier/framing variations |

### Prompt Construction

Prompts are built from components in this order:
1. Grammar (article/framing)
2. Size (if present)
3. Phenology (if "open" or "blooming" - goes before taxonomy)
4. Color (if present)
5. Taxonomy
6. Phenology (if "bud", "closed bud", or "in bloom" - special placement)
7. Anatomy (if present)
8. Negation (appended at end)

**Special cases:**
- **Phenology "bud":** `"a cowpea flower bud"` (placed after taxonomy)
- **Phenology "closed bud":** `"a closed cowpea bud"` (replaces "flower" with "bud")
- **Phenology "open":** `"an open cowpea flower"` (article changes to "an", placed before taxonomy)
- **Phenology "in bloom":** `"a cowpea flower in bloom"` (placed after taxonomy)
- **Anatomy "corolla":** `"a cowpea corolla"` (replaces "flower" with "corolla")
- **Grammar "":** Bare noun, no article (e.g., `"cowpea flower"`)

**Baseline prompt:** `"a flower"` (all axes at baseline)

**Example combinations:**
- Baseline: `"a flower"`
- Color only: `"a yellow flower"`
- Taxonomy only: `"a cowpea flower"`
- Color + Taxonomy: `"a yellow cowpea flower"` (not in Phase 1 - that's Phase 2)

### Test Configurations

**Total configurations:** 1 baseline + (sum of non-baseline values per axis)

For each axis:
- Baseline: All axes at baseline → `"a flower"`
- Axis variations: Vary that axis while keeping others at baseline

**Example:**
- Baseline: `"a flower"` (all axes at baseline)
- Taxonomy axis: `"a cowpea flower"`, `"a bean flower"`, `"a pea flower"`, `"a legume flower"`, `"a black-eyed pea flower"`, `"a vigna unguiculata flower"`, `"a crop flower"` (7 tests)
- Color axis: `"a yellow flower"`, `"a white flower"`, `"a cream flower"`, `"a purple flower"` (4 tests)
- Size axis: `"a large flower"`, `"a small flower"`, `"a tiny flower"` (3 tests)
- Phenology axis: `"a flower bud"`, `"an open flower"`, `"a closed flower bud"`, `"a blooming flower"`, `"a flower in bloom"` (5 tests)
- Negation axis: `"a flower, not a bud, not the green calyx, not a leaf"`, `"a flower, not a bud, not the green calyx"`, `"a flower, not a leaf"`, `"a flower, not a bud"`, `"a flower, not the green calyx"` (5 tests)
- Anatomy axis: `"a flower with open petals"`, `"a flower with visible petals"`, `"a flower with petals and stamens"`, `"a corolla"` (4 tests)
- Grammar axis: `"a single flower"`, `"flower"` (bare noun), `"a photo of a flower"`, `"one flower"`, `"close-up of a flower"` (5 tests)

**Total configurations:** 1 baseline + 7 + 4 + 3 + 5 + 5 + 4 + 5 = **34 prompt configurations per model**

This matches the original Experiment 7 (34 prompts evaluated).

## Evaluation Protocol

### Dataset
- **Test set:** Full holdout test set (138 images, 839 ground truth flowers)
- **No sampling:** Evaluate on all images
- **Format:** YOLO format labels

### Metrics to Compute

For each configuration, compute:

#### Detection Metrics (at multiple IoU thresholds)
- **mAP@0.3** - Relaxed IoU for fine-grained objects
- **mAP@0.5** - Standard IoU threshold
- **mAP@0.5:0.95** - COCO-style average (0.5 to 0.95 in steps of 0.05)
- **Precision@0.5** - At IoU=0.5
- **Recall@0.5** - At IoU=0.5
- **F1@0.5** - At IoU=0.5

#### Counting Metrics
- **R²** - Coefficient of determination (predicted vs actual counts)
- **RMSE** - Root mean squared error
- **MAE** - Mean absolute error
- **MAPE** - Mean absolute percentage error
- **Slope** - Linear regression slope (ideal = 1.0)
- **Intercept** - Linear regression intercept (ideal = 0.0)
- **Bias** - Mean prediction error

### Model-Specific Settings

#### YOLO World
- **Weights:** `yolov8x-worldv2.pt`
- **Confidence threshold:** 0.1 (low threshold to capture all predictions)
- **NMS IoU:** 0.5 (standard)
- **Inference:** Single-class detection per prompt

#### SAM3
- **Model:** `facebook/sam3` (Hugging Face)
- **Confidence threshold:** 0.1 (for consistency)
- **Inference:** Text-to-segmentation, convert masks to bounding boxes
- **Note:** May not support all prompt variations (e.g., complex negation)

## Analysis Plan

### Per-Axis Contribution Analysis

For each axis, compute:
1. **Baseline performance:** mAP@0.5, F1@0.5 with all axes at baseline
2. **Axis variations:** Performance for each non-baseline value
3. **Delta from baseline:** `ΔmAP = mAP(variation) - mAP(baseline)`
4. **Best value per axis:** Which value gives highest mAP/F1

### Model Comparison

For each axis:
- **YOLO World sensitivity:** How much does performance change?
- **SAM3 sensitivity:** How much does performance change?
- **Model differences:** Which model is more sensitive to each axis?

### Expected Findings (from previous work)

Based on original Experiment 7 (YOLO World, 158 images), we expect:

- **Taxonomy:** Very strong positive effect
  - Best: `"cowpea flower"` (+0.137 mAP vs `"flower"` baseline)
  - Genus (`"bean flower"`, `"pea flower"`) intermediate
  - Scientific (`"vigna unguiculata flower"`) and generic (`"crop flower"`) fail completely (0 mAP)
  
- **Color:** Strong positive effect
  - Best: `"yellow"` (+0.095 mAP vs baseline)
  - `"white"` and `"cream"` also help
  - `"purple"` hurts significantly (-0.141 mAP)
  
- **Anatomy:** Moderate positive effect
  - Best: `"with open petals"` (+0.058 mAP)
  - `"with visible petals"` also helps
  - `"corolla"` fails completely (0 mAP)
  
- **Grammar:** Moderate positive effect
  - Best: `"a single"` (+0.062 mAP)
  - Bare noun (`""`) and framing (`"a photo of a"`, `"close-up of a"`) hurt
  
- **Negation:** Strong positive effect
  - Best: `"not a bud, not the green calyx, not a leaf"` (+0.066 mAP)
  - Individual negations also help (but less)
  - Note: May be replaced by absorber architecture in Phase 2
  
- **Phenology:** Mixed results
  - `"bud"` and `"open"` slightly help
  - `"in bloom"` fails catastrophically (-0.143 mAP) — CLIP interprets as scene concept
  
- **Size:** Negative effect
  - All size modifiers hurt performance
  - Worst: `"tiny"` (-0.060 mAP)
  - Baseline (no size) is best

**Key question:** Do these patterns hold for SAM3, or does SAM3 respond differently? Also, how do results compare on the holdout test set (138 images) vs original (158 images)?

## Implementation Details

### Code Structure

```python
from agvfm.config.experiments import FACTOR_AXES, generate_factor_combinations, build_prompt_from_components
from agvfm.experiments import run_factor_analysis, analyze_factor_contributions
from agvfm.models import YOLOWorldModel, SAM3Model
from agvfm.experiments import Evaluator

# Generate all combinations
combinations = generate_factor_combinations()

# Run for YOLO World
yolo_model = YOLOWorldModel(weights_path="yolov8x-worldv2.pt")
yolo_evaluator = Evaluator(yolo_model, images_dir, labels_dir)
yolo_results = run_factor_analysis(yolo_evaluator, iou_thresholds=[0.3, 0.5])

# Run for SAM3
sam3_model = SAM3Model(model_id="facebook/sam3")
sam3_evaluator = Evaluator(sam3_model, images_dir, labels_dir)
sam3_results = run_factor_analysis(sam3_evaluator, iou_thresholds=[0.3, 0.5])

# Analyze contributions
yolo_contributions = analyze_factor_contributions(yolo_results)
sam3_contributions = analyze_factor_contributions(sam3_results)
```

### Results Storage

Save results to: `notebooks/results/phase1_factor_analysis/`

**Per model:**
- `yolo_world_factor_analysis.json` - Full results
- `sam3_factor_analysis.json` - Full results

**Analysis:**
- `factor_contributions_comparison.json` - Side-by-side comparison
- `factor_contributions_comparison.csv` - For easy analysis

### Output Format

Each result entry:
```json
{
  "combo_name": "color:yellow",
  "components": {"color": "yellow", "taxonomy": "flower", ...},
  "prompt": "a yellow flower",
  "model": "YOLO World",
  "metrics_by_iou": {
    "iou_0.3": {"map": 0.xxx, "f1": 0.xxx, ...},
    "iou_0.5": {"map": 0.xxx, "f1": 0.xxx, ...}
  },
  "map_coco": 0.xxx,
  "counting": {"r2": 0.xxx, "rmse": 0.xxx, ...}
}
```

## Deliverables

1. **Factor contribution plots:** Per-axis mAP/F1 contributions for each model
2. **Model comparison table:** Side-by-side comparison of axis sensitivities
3. **Prompt engineering rules (draft):** What works for YOLO World vs SAM3
4. **Baseline performance:** Establish baseline for each model
5. **Dead ends identified:** What doesn't work (e.g., phenology, size modifiers)

## Success Criteria

Phase 1 is complete when:
- ✅ All factor combinations tested for both models
- ✅ Per-axis contributions computed and visualized
- ✅ Model-specific sensitivities identified
- ✅ Baseline performance established
- ✅ Results saved in structured format
- ✅ Key differences between models documented

## Next Steps (Phase 2)

After Phase 1, we will:
- Use best-performing components to build combination prompts
- Test absorber architecture (YOLO World only, likely)
- Compare text negation vs absorber architecture
- Optimize for F1, not just mAP
