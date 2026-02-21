# Phase 4: Unlabeled Threshold Selection

> **Status:** 🔄 In Progress (Step 1: ✅ Complete, Step 2: ⏳ Pending)  
> **Objective:** Develop and validate methods for selecting confidence thresholds without labeled data  
> **Details:** [approach_phase4.md](../approach/approach_phase4.md)

## Progress Tracking

### Step 1: Distribution Analysis ✅ COMPLETE

**Status:** ✅ Complete  
**Date:** 2025-02-19

#### Data Collection
- **YOLO World (C2_not_bud_calyx):** 8,956 detections collected at conf=0.01
- **SAM3 (C2_not_bud):** 25,692 detections collected at conf=0.01
- **Test set:** 158 images (full test set)
- **Collection script:** `experiments/scripts/experiments/phase4/collect_raw_confidence_scores.py`
- **Output:** `experiments/results/phase4_unlabeled_threshold_selection/`

#### Distribution Analysis Results

**YOLO World (C2_not_bud_calyx):**
- **Bimodality Coefficient:** BC = 0.8530 (high BC, but **NOT truly bimodal**)
- **Peak Detection (5% threshold):** 1 peak at 0.028
- **Status:** Unimodal but highly skewed (skewness=2.02, kurtosis=2.98)
- **Suggested Thresholds:**
  - **Elbow Point:** 0.9941 (too high, not useful)
- **Phase 3 Optimal F1 Threshold:** 0.4500 (F1 = 0.4920)
- **Finding:** High BC is due to high skewness, not true bimodality. No KDE valley method applicable.

**SAM3 (C2_not_bud):**
- **Bimodality Coefficient:** BC = 0.8094 (high BC, but **NOT truly bimodal**)
- **Peak Detection (5% threshold):** 1 peak at 0.023
- **Status:** Unimodal but highly skewed (skewness=3.13, kurtosis=10.35)
- **Suggested Thresholds:**
  - **Elbow Point:** 0.8887 (too high, not useful)
- **Phase 3 Optimal F1 Threshold:** 0.4500 (F1 = 0.5863)
- **Finding:** High BC is due to high skewness, not true bimodality. No KDE valley method applicable.

#### Key Findings
1. **Distributions are NOT truly bimodal** - they are unimodal but highly skewed ⚠️
2. **Bimodality coefficient can be misleading** - high BC (>0.555) can result from high skewness/kurtosis, not just true bimodality
3. **Peak detection issue fixed:** Initial 1% threshold picked up noise as "second peak" (only 1-4% of first peak's height)
4. **With robust 5% threshold:** Only 1 peak detected for both models
5. **KDE Valley method not applicable** - requires true bimodality (2+ peaks)
6. **Elbow method produces thresholds that are too high** - not useful for these distributions
7. **Distribution analysis needs alternative methods** - cannot rely on bimodality for threshold selection

#### Output Files
- **Analysis results:** `experiments/results/phase4_unlabeled_threshold_selection/distribution_analysis_results.json`
- **Summary report:** `experiments/results/phase4_unlabeled_threshold_selection/summary_report.txt`
- **Visualizations:** `experiments/results/phase4_unlabeled_threshold_selection/plots/`
  - `yolo_world_C2_not_bud_calyx_distribution.png`
  - `sam3_C2_not_bud_distribution.png`
- **Analysis script:** `experiments/scripts/experiments/phase4/analyze_distributions.py`

#### Technical Details
- **Peak detection threshold:** Increased from 1% to 5% to filter out noise
- **Bimodality assessment:** Requires both high BC (>0.555) AND multiple detected peaks
- **Issue identified:** Initial "second peak" was noise (density only 1-4% of first peak)
- **Root cause:** High skewness/kurtosis in unimodal distributions gives high BC values

### Step 2: Apply to Phase 1/2 Top Configs (Validation)

**Status:** ⏳ Pending  
**Objective:** Validate distribution analysis method on Phase 1/2 top configs

**Planned Tasks:**
1. Identify top configs from Phase 1/2
2. Collect raw confidence scores for these configs
3. Run distribution analysis
4. Compare suggested thresholds to Phase 3 optimal thresholds (if available)
5. Evaluate method generalizability

### Step 3: Embedding Space Analysis

**Status:** ⏳ Pending  
**Objective:** Develop embedding-based quality signals for threshold selection

## Quick Stats

- **Configs analyzed:** 2 (best YOLO World and SAM3 configs)
- **Total detections collected:** 34,648 (8,956 YOLO World + 25,692 SAM3)
- **Distribution methods tested:** 4 (bimodality, KDE peaks, elbow, gap)
- **Best method:** KDE valley (works well for YOLO World, needs refinement for SAM3)
