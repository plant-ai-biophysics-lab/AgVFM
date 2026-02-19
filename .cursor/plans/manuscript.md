# Manuscript: CVPR AgVision Workshop 2026

> **Target:** CVPR 2026 Workshop on Agricultural Vision  
> **Page Limit:** 8 pages (including references)  
> **Format:** CVPR style

## Paper Title (Draft)

**Zero-Shot Agricultural Object Detection: A Systematic Comparison of Vision Foundation Models**

*Alternative titles:*
- Comparing Vision Foundation Models for Zero-Shot Fine-Grained Agricultural Detection
- Prompt Engineering for Zero-Shot Agricultural Vision: YOLO World vs SAM3

## Structure (8 pages)

### 1. Abstract (~0.5 page)
- **Problem / deployment constraint:** Agricultural monitoring tasks (e.g., flower counting for phenotyping/yield proxies) often have **limited labels**, strong domain shift, and **small/cluttered targets**. “Zero-shot” models exist, but practitioners lack **reproducible guidance** on (i) which foundation model family to use and (ii) how sensitive performance is to prompt wording and confidence thresholds when labels are unavailable.
- **Approach:** A **clean-slate, model-agnostic evaluation protocol** that compares an open-vocabulary *detector* (YOLO World) to a promptable *segmenter* (SAM3) on a fine-grained flower dataset. We run a three-part experiment suite: **factor analysis → combination/negation/absorbers → confidence sweeps**, and report detection *and* counting metrics.
- **Key Findings (expected template):**
  - Prompt components have **architecture-specific effects**; prompt rules do not transfer reliably across models
  - Detector vs segmenter shows a **precision–recall tradeoff** and different failure modes (e.g., handling of negation, confusers)
  - Confidence threshold sensitivity is non-trivial; some configs are **more robust** (better suited for true zero-shot deployment)
- **Impact:** Practical, quantitative guidance for deploying VFMs in agricultural detection/counting without fine-tuning, plus a reproducible framework to extend to additional models and crops.

### 2. Introduction (~1 page)
- **Motivation:**
  - Agricultural monitoring needs (flower counting, phenotyping, crop development tracking) increasingly rely on vision in settings where **labeling is expensive**, conditions vary, and target objects are **small and confusable** (buds/calyx/leaves).
  - Vision foundation models promise rapid adaptation via text prompts, but real deployments face a key friction: **without labels, you cannot easily tune prompts or confidence thresholds**, and the “best” settings can be model-dependent.
  - Result: users can obtain impressive demos but struggle to achieve **reliable, reproducible performance** in domain settings like agriculture.
- **Vision Foundation Models:**
  - YOLO World: open-vocabulary detection
  - SAM3: promptable segmentation
  - These represent different architectural choices (detection vs segmentation). Agriculture needs clarity on **which family works better for fine-grained targets**, and under what prompting/thresholding regime.
- **Paper goal:** Provide a **systematic, quantitative comparison** and extract **actionable prompting rules** and **threshold-robust operating points** suitable for truly label-scarce deployment.
- **Contributions:**
  1. A **clean-slate evaluation suite** for zero-shot agricultural detection that is explicitly designed for the “no labels for tuning” constraint.
  2. A **structured prompt experiment suite**: (Phase 1) one-factor-at-a-time prompt component analysis and (Phase 2) combination tests that expose interaction effects and the precision–recall tradeoff.
  3. An evaluation of **negation strategies**, including absorber-style confuser handling for open-vocabulary detectors, and an analysis of when these strategies help/hurt.
  4. A **confidence threshold sensitivity study** (Phase 3) that measures robustness and recommends operating points for deployment.
  5. A unified report of **detection + counting metrics** across multiple IoU criteria, aligned with agricultural use (count accuracy matters).

### 3. Related Work (~1 page)
- **Agricultural Vision:**
  - Flower detection, counting, phenotyping
  - Domain-specific challenges
- **Zero-Shot Detection:**
  - Open-vocabulary object detection
  - CLIP-based models
  - Prompt engineering
- **Vision Foundation Models:**
  - YOLO World architecture
  - SAM3 architecture
  - Comparison studies (general vision, limited agriculture-specific guidance)
- **Confidence calibration / thresholding (brief):**
  - Why threshold sensitivity is a core issue in label-scarce deployment
  - Prior work on calibration/operating points (high level; keep short due to 8 pages)

### 4. Methods (~2 pages)

#### 4.1 Dataset (0.3 page)
- **Cowpea flower detection**
  - 138 test images, 839 ground truth flowers
  - Fine-grained challenge: small objects, cluttered scenes
  - YOLO format annotations
- **Evaluation splits:** Test set only (no training/validation splits needed for zero-shot)

#### 4.2 Models (0.5 page)
- **YOLO World:**
  - Architecture: CLIP text encoder + YOLO detection head
  - Open-vocabulary detection
  - Multi-class support (absorber architecture)
- **SAM3:**
  - Architecture: Promptable segmentation
  - Text-to-segmentation
  - Mask-to-box conversion for comparison

#### 4.3 Evaluation Metrics (0.4 page)
- **Detection metrics:**
  - mAP@0.3, mAP@0.5, mAP@0.5:0.95
  - Precision, Recall, F1
- **Counting metrics:**
  - R², RMSE, MAE, MAPE
  - Slope, intercept (linear regression)
- **Rationale:** Multiple IoUs for fine-grained objects; counting for agricultural applications

#### 4.4 Experimental Design (0.8 page)
- **Phase 1: Structured Factor Analysis**
  - One-factor-at-a-time (OFAT) across 7 axes
  - Axes: Color (yellow/purple/white/generic), Taxonomy (cowpea flower/flower), Anatomy (with open petals/generic), Negation (text negation/none), Grammar (a single/a), Phenology (in bloom/none), Size (small/tiny/none)
  - Baseline: `"a flower"` (all axes at baseline)
  - ~15-20 prompt configurations per model
  - Purpose: Isolate contribution of each component per model
- **Phase 2: Combination Tests & Absorber Architecture**
  - Systematic combinations of best components from Phase 1
  - Two-, three-, and four-component combinations
  - Kitchen sink: `"a single yellow cowpea flower with open petals, not a bud, not the green calyx, not a leaf"`
  - Negation strategy comparison: Text negation vs absorber architecture (YOLO World only)
  - Absorber configs: H3b (bud+calyx), H2a (leaf+stem), H3c (bud+calyx+leaf), H5 (all absorbers)
  - Multi-class assembly tests (single vs multi-class detection)
- **Phase 3: Confidence Threshold Analysis**
  - Sweep confidence thresholds: 0.05 to 0.95 in steps of 0.05 (19 values)
  - Test on best configs from Phase 2: C2, C1, H3b, H2a, C3
  - Analyze sensitivity: How performance changes with threshold
  - Critical for zero-shot deployment (no labeled data for tuning)

### 5. Experiments & Results (~2 pages)

#### 5.1 Factor Analysis Results (0.6 page)
- **Per-axis contributions** for each model (ΔmAP, ΔF1 from baseline)
- **Key findings:**
  - Which factors matter most for YOLO World vs SAM3
  - Model-specific sensitivities (e.g., does SAM3 respond differently to color/taxonomy?)
  - Dead ends identified (phenology, size modifiers expected to fail)
- **Quantitative results:**
  - Baseline performance: `"a flower"` for both models
  - Best single-factor improvements per axis
  - Model differences in factor sensitivity
- **Figure:** Side-by-side bar chart of factor contributions (YOLO World vs SAM3)

#### 5.2 Model Comparison (0.6 page)
- **Quantitative comparison table:**
  - Best configs per model (C2, C1, H3b for YOLO World; C2, C3 for SAM3)
  - Metrics: mAP@0.3, mAP@0.5, mAP@0.5:0.95, F1, Precision, Recall
  - Counting metrics: R², RMSE, MAE, MAPE, slope, intercept
- **Key findings:**
  - YOLO World: better recall, higher mAP, supports absorber architecture
  - SAM3: better precision, more conservative, cannot handle complex text negation
  - Trade-offs: Precision vs recall, architecture-specific capabilities
  - Absorber architecture: YOLO World only, significant precision improvement
- **Table:** Comprehensive comparison of best configs per model

#### 5.3 Prompt Engineering Rules (0.4 page)
- **Model-specific rules derived from Phase 1 & 2:**
  - **YOLO World:**
    - Works: Species names (cowpea), color (yellow), anatomy (open petals), grammar (a single), absorber architecture
    - Fails: Phenology ("in bloom"), size modifiers ("small", "tiny"), complex text negation (precision collapse)
  - **SAM3:**
    - Works: Simple prompts, color, taxonomy, anatomy
    - Fails: Complex text negation (complete failure), absorber architecture (not supported)
- **Component interaction effects:**
  - Super-additive interactions (sum of parts < whole)
  - Precision-recall tradeoff (mAP ≠ F1 optimum)
- **Table:** Prompt engineering rules per architecture (what works/fails)

#### 5.4 Confidence Threshold Analysis (0.4 page)
- **Sensitivity analysis:**
  - Performance curves: mAP, F1, Precision, Recall vs confidence threshold (0.05-0.95)
  - Optimal thresholds per model and config
  - Sensitivity windows: How much performance drops with ±0.1 threshold change
  - Robustness scores: Which configs are most robust to threshold choice
- **Key findings:**
  - Non-absorber configs: High sensitivity, threshold tuning critical
  - Absorber configs: Low sensitivity, already near-optimal
  - Model differences: YOLO World vs SAM3 threshold behavior
- **Implications for zero-shot deployment:**
  - Robust configs (low sensitivity) preferred for deployment
  - Threshold selection strategy per model
- **Figure:** Performance curves (mAP, F1) vs confidence threshold for best configs

### 6. Discussion (~0.5 page)
- **Why different architectures perform differently:**
  - **YOLO World:** Detection architecture, multi-class support enables absorber strategy, better at recall
  - **SAM3:** Segmentation architecture, more conservative (higher precision), cannot handle complex negation
  - Text prompt interpretation: SAM3 may interpret prompts differently (e.g., negation failure)
- **Implications for zero-shot agricultural vision:**
  - **When to use YOLO World:** High-recall applications, when absorber architecture is beneficial, complex prompt requirements
  - **When to use SAM3:** High-precision applications, simple prompts, when false positives are costly
  - **Prompt engineering:** Model-specific rules are critical; what works for one may fail for the other
  - **Confidence thresholds:** Absorber configs are more robust (lower sensitivity), preferred for deployment
- **Limitations:**
  - Single species (cowpea) - need multi-species validation
  - Single task (flower detection) - may not generalize to other agricultural objects
  - Limited to two models - future work should include more VFMs
- **Future work:**
  - Grounding gap analysis: Understand why models fail (structural vs linguistic)
  - Multi-species validation: Test prompt rules on additional species
  - Extension to other agricultural tasks: Fruits, leaves, pests, etc.
  - Additional models: CLIP-based detectors, other segmentation models

### 7. Conclusion (~0.5 page)
- **Key findings:**
  1. **Model-specific prompt rules are critical:** What works for YOLO World (absorber architecture, complex prompts) fails for SAM3 (simple prompts only)
  2. **Architecture choice matters:** YOLO World excels at recall and supports advanced strategies; SAM3 excels at precision but is more limited
  3. **Confidence threshold sensitivity varies:** Absorber configs are more robust, preferred for zero-shot deployment
  4. **Component interactions are super-additive:** Combining best components yields better-than-additive performance
- **Recommendations for practitioners:**
  - **Use YOLO World for:** High-recall applications, when absorber architecture is beneficial, complex prompt requirements
  - **Use SAM3 for:** High-precision applications, simple prompts, when false positives are costly
  - **Follow model-specific rules:** Don't assume prompts transfer between architectures
  - **Optimize for F1, not just mAP:** Precision-recall tradeoff is critical
  - **Use robust configs:** Absorber architecture provides lower threshold sensitivity
- **Impact:** Provides practical guidance for zero-shot agricultural vision, enabling effective deployment without labeled training data

## Key Figures/Tables (Target: 4-5 total)

1. **Figure 1:** Factor analysis results
   - Side-by-side bar chart of per-axis contributions (ΔmAP, ΔF1)
   - YOLO World vs SAM3 comparison
   - 7 axes: Color, Taxonomy, Anatomy, Negation, Grammar, Phenology, Size

2. **Table 1:** Model comparison
   - Best configs per model (C2, C1, H3b for YOLO World; C2, C3 for SAM3)
   - Metrics: mAP@0.3, mAP@0.5, mAP@0.5:0.95, F1, Precision, Recall
   - Counting: R², RMSE, MAE, slope

3. **Table 2:** Prompt engineering rules
   - What works for YOLO World (✓) vs SAM3 (✓)
   - What fails for each (✗)
   - Component interaction effects

4. **Figure 2:** Confidence threshold sensitivity
   - Performance curves: mAP and F1 vs confidence threshold (0.05-0.95)
   - Multiple configs overlaid (C2, C1, H3b)
   - YOLO World vs SAM3 comparison

5. **Figure 3:** Precision-Recall analysis
   - P-R scatter plot with F1 iso-lines
   - All configs from Phase 2
   - Shows tradeoff: text negation vs absorber architecture

6. **Figure 4 (optional):** Example predictions
   - Qualitative examples: TP, FP, FN
   - Best config per model side-by-side

## Writing Notes

- **Be concise:** 8 pages is tight, every word counts
- **Focus on comparison:** This is a comparison paper, emphasize differences
- **Practical emphasis:** Agricultural vision practitioners are the audience
- **Quantitative:** Numbers and tables are key
- **Clear structure:** Make it easy to scan and find key findings

## Status

📝 **Outlining** - Structure defined, awaiting experimental results
