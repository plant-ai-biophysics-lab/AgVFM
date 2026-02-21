# Phase 2: Combination Tests & Absorber Architecture

> **Status:** ✅ Complete (YOLO World: combinations + absorbers; SAM3: combinations)  
> **Objective:** Systematic combinations, negation strategies, absorber architecture  
> **Details:** [approach_phase2.md](../approach/approach_phase2.md)

## Progress Tracking

### YOLO World - Combinations
- **Status:** ✅ Complete (21/21 total: 13 original + 8 strategic)
- **Configs tested:** 13/13 original (1 baseline + 6 two-component + 4 three-component + 1 four-component + 1 kitchen sink)
- **Strategic experiments:** 8/8 complete (size variants, partial negation)
- **Best performer:** `C2_not_bud_calyx` (mAP@0.5: 0.4123) - **NEW BEST**, beats C1 (0.3830)
- **Key finding:** Partial negation (2 clauses) > Full negation (3 clauses) for YOLO World
- **Test set:** Full (158 images)
- **Results:** `experiments/results/phase2_combinations/yolo_world_combinations.json`
- **Plots:** `experiments/results/phase2_combinations/plots/`

### YOLO World - Absorbers
- **Status:** ✅ Complete (7/7 total: 4 original + 3 strategic)
- **Configs tested:** 4/4 original (H3b, H2a, H3c, H5)
- **Strategic experiments:** 3/3 complete (H3b with simpler base prompts)
- **Key finding:** Simpler base prompts did NOT improve absorber performance (all = 0.0909 mAP@0.5)
- **Best performer:** H3b (mAP@0.5: 0.1359) - still much worse than text negation
- **Test set:** Full (158 images)
- **Results:** `experiments/results/phase2_combinations/yolo_world_absorbers.json`
- **Plots:** Included in Phase 2 visualization plots

### YOLO World - Multi-Class
- **Status:** ⏳ Not Started
- **Configs to test:** 4 (single-class baseline + 2 two-class + 1 three-class)
- **Test set:** Full (158 images)
- **Results:** `experiments/results/phase2_combinations/yolo_world_multiclass.json`

### SAM3 - Combinations
- **Status:** ✅ Complete (21/21 total: 13 original + 8 strategic, but only 2 SAM3-specific)
- **Configs tested:** 13/13 original (1 baseline + 6 two-component + 4 three-component + 1 four-component + 1 kitchen sink)
- **Strategic experiments:** 2/2 complete (size variants for negation-free optimization)
- **Best performer:** `C_color_taxonomy_anatomy_tiny` (mAP@0.5: 0.5364) - **NEW BEST**, beats C2 (0.4925)
- **Key finding:** Size component ("tiny") helps when added to `C_color_taxonomy_anatomy` (+0.0453 mAP)
- **Test set:** Full (158 images)
- **Results:** `experiments/results/phase2_combinations/sam3_combinations.json`
- **Plots:** `experiments/results/phase2_combinations/plots/sam3_*.png`

## Quick Stats

- **Total configs (YOLO World):** 21 (13 combinations + 4 absorbers + 4 multi-class)
- **Total configs (SAM3):** 13 (combinations only, no absorbers/multi-class)
- **Test set:** Full test set (158 images)
- **Expected time:** ~20-30 minutes per model
- **Results location:** `experiments/results/phase2_combinations/`

## Configurations

### Combination Configs (Section 2.1)

1. **Baseline:**
   - `C3`: `"a cowpea flower"`

2. **Two-component (6 configs):**
   - `C_color_taxonomy`: `"a yellow cowpea flower"` (Color + Taxonomy)
   - `C_color_anatomy`: `"a yellow flower with open petals"` (Color + Anatomy)
   - `C_color_grammar`: `"a single yellow flower"` (Color + Grammar)
   - `C_taxonomy_anatomy`: `"a cowpea flower with open petals"` (Taxonomy + Anatomy)
   - `C_taxonomy_grammar`: `"a single cowpea flower"` (Taxonomy + Grammar)
   - `C_anatomy_grammar`: `"a single flower with open petals"` (Anatomy + Grammar)

3. **Three-component (4 configs):**
   - `C_color_taxonomy_anatomy`: `"a yellow cowpea flower with open petals"` (Color + Taxonomy + Anatomy)
   - `C_color_taxonomy_grammar`: `"a single yellow cowpea flower"` (Color + Taxonomy + Grammar)
   - `C_color_anatomy_grammar`: `"a single yellow flower with open petals"` (Color + Anatomy + Grammar)
   - `C_taxonomy_anatomy_grammar`: `"a single cowpea flower with open petals"` (Taxonomy + Anatomy + Grammar)

4. **Four-component (1 config):**
   - `C2`: `"a single yellow cowpea flower with open petals"`

5. **Kitchen sink (1 config):**
   - `C1`: `"a single yellow cowpea flower with open petals, not a bud, not the green calyx, not a leaf"`

### Absorber Configs (Section 2.2)

- **H3b:** Bud + calyx absorbers
- **H2a:** Leaf + stem absorbers
- **H3c:** Bud + calyx + leaf absorbers
- **H5:** All absorbers (bud, calyx, leaf, stem, soil)

### Multi-Class Configs (Section 2.3)

- Single-class baseline: `"a single yellow cowpea flower with open petals"`
- Two-class: Flower + Bud
- Two-class: Yellow + White
- Three-class: Flower + Bud + Calyx

## Strategic Experiments (Based on mAP@0.5 Analysis) ✅

### Rationale
Based on Phase 2 mAP@0.5 results:
- **YOLO World:** C1 (with negation) = 0.3830 is best, but size component ("tiny") was best in Phase 1 and not tested in combinations
- **SAM3:** C2 (no negation) = 0.4925 is best, but negation hurts SAM3 (-0.1745), so optimize within negation-free space
- **Absorbers:** Current absorbers are poor (H3b = 0.1359 << C2 = 0.3327), test simpler base prompts

### Results Summary

**YOLO World - Key Findings:**
- ✅ **Partial negation wins:** `C2_not_bud_calyx` (0.4123) > C1 (0.3830) - **NEW BEST**
- ✅ Two negation clauses better than three: `C2_not_bud_calyx` (0.4123) > `C2_not_calyx` (0.3807) > C1 (0.3830)
- ❌ Size component doesn't help: `C2_tiny` (0.3053) < C2 (0.3327), `C1_tiny` (0.3481) < C1 (0.3830)
- ❌ Absorber optimization failed: All simpler bases = 0.0909 mAP (same as worst absorbers)

**SAM3 - Key Findings:**
- ✅ **Size component helps:** `C_color_taxonomy_anatomy_tiny` (0.5364) > C2 (0.4925) - **NEW BEST**
- ⚠️ Size with full C2 doesn't help: `C2_tiny` (0.4667) < C2 (0.4925)
- ✅ Best strategy: Add "tiny" to `C_color_taxonomy_anatomy` (simpler than C2, no grammar)

### YOLO World Strategic Experiments

1. **Size Component Integration (3 configs):**
   - `C2_tiny`: `"a single tiny yellow cowpea flower with open petals"`
   - `C1_tiny`: `"a single tiny yellow cowpea flower with open petals, not a bud, not the green calyx, not a leaf"`
   - `C_color_taxonomy_grammar_tiny`: `"a single tiny yellow cowpea flower"`

2. **Partial Negation Variants (4 configs):**
   - `C2_not_bud`: `"a single yellow cowpea flower with open petals, not a bud"`
   - `C2_not_calyx`: `"a single yellow cowpea flower with open petals, not the green calyx"`
   - `C2_not_leaf`: `"a single yellow cowpea flower with open petals, not a leaf"`
   - `C2_not_bud_calyx`: `"a single yellow cowpea flower with open petals, not a bud, not the green calyx"`

3. **Absorber Optimization (3 configs):**
   - `H3b_simple`: H3b absorbers with `"a yellow cowpea flower"` base (simplest)
   - `H3b_no_anatomy`: H3b absorbers with `"a single yellow cowpea flower"` base (no anatomy)
   - `H3b_no_grammar`: H3b absorbers with `"a yellow cowpea flower with open petals"` base (no grammar)

### SAM3 Strategic Experiments

1. **Negation-Free Optimization (2 configs):**
   - `C2_tiny`: `"a single tiny yellow cowpea flower with open petals"` (same as YOLO World)
   - `C_color_taxonomy_anatomy_tiny`: `"a tiny yellow cowpea flower with open petals"`

**Total strategic experiments:** 13 configs (8 YOLO World combinations + 3 YOLO World absorbers + 2 SAM3)

## Notes

- Running experiments with: `python experiments/scripts/experiments/phase2/run_combinations.py --model yolo_world --full-test-set`
- Can resume interrupted runs (checks existing results)
- Results saved incrementally after each config
- Batch size: 1 for SAM3 (sequential processing due to batching issues), 15 for YOLO World (except absorbers which run sequentially)
- Absorber architecture: YOLO World only (SAM3 doesn't support multi-class)
- Visualization: `python experiments/scripts/visualization/visualize_phase2_results.py` (generates plots for both YOLO World and SAM3)