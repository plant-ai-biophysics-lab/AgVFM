# Phase 4: Unlabeled Threshold Selection (Revised)

> **Objective:** Develop and validate methods for selecting confidence thresholds without labeled data, using principled approaches that don't rely on fixed counts or arbitrary percentiles.

## Motivation: The Threshold Selection Problem

### The Issue

Phase 1 and Phase 2 used a fixed confidence threshold of **0.1**, but Phase 3 revealed optimal F1 thresholds are **0.30-0.45** (3-4x higher). This raises concerns about the validity of Phase 1/2 results.

**Key Insight:** mAP (mean Average Precision) is threshold-agnostic - it's computed across all confidence thresholds and measures ranking quality. This means:
- **Phase 1/2 mAP rankings are valid** regardless of the threshold used for point metrics (P/R/F1)
- The primary findings (which components help, which configs are best) are based on mAP
- Point metrics (P/R/F1) at conf=0.1 are suboptimal, but they're not the primary ranking metric

**Zero-Shot Constraint:** We cannot use test labels to select thresholds. This means:
- ❌ Cannot re-run Phase 1/2 with "optimal" thresholds from Phase 3 (that's using test labels)
- ❌ Cannot tune thresholds per config using validation data (we have no validation data)
- ✅ Can use fixed thresholds that are pre-declared (not tuned on test data)
- ✅ Can develop unlabeled operating rules (distribution analysis, embedding space analysis)

**Solution:** Phase 4 develops methods for selecting thresholds without labels, addressing the core deployment challenge.

## Critical Issues with Initial Methods

### Problems Identified

1. **Top-K Detections:** ❌ Won't work - images have 1-50 flowers, no fixed K
2. **Percentile-Based:** ❌ Too arbitrary, needs dataset-specific tuning
3. **Detection Volume Matching:** ❌ No fixed expected count per image (varies 1-50)
4. **Distribution Analysis:** ✅ Promising - more principled

## Revised Methods

### Method 1: Confidence Distribution Analysis (Primary)

**Concept:** Use intrinsic structure in confidence distributions to identify natural thresholds.

**Rationale:**
- Confidence distributions often show bimodality (TP mode vs FP mode)
- Natural breaks, elbows, or gaps can suggest thresholds
- No domain knowledge needed - uses model's own signal

**Implementation Approaches:**

1. **Bimodality Detection:**
   - Compute bimodality coefficient (BC)
   - If BC > 0.555, distribution is bimodal
   - Find gap between modes → use as threshold
   - Validate: Does gap align with TP/FP separation?

2. **KDE Peak Detection:**
   - Fit kernel density estimation (KDE) to confidence scores
   - Identify peaks (modes) in distribution
   - Use valley between peaks as threshold
   - Works for multimodal distributions

3. **Elbow/Knee Detection:**
   - Plot cumulative distribution or sorted confidence scores
   - Find natural "elbow" where slope changes
   - Use elbow point as threshold
   - Common in clustering/optimization

4. **Gap Analysis:**
   - Sort confidence scores
   - Find largest gap between consecutive scores
   - Use gap location as threshold
   - Assumes TP/FP modes are separated

**Advantages:**
- Fully unlabeled (no domain knowledge)
- Uses intrinsic model structure
- Principled (based on distribution properties)
- Can work per-image or globally

**Disadvantages:**
- May not work if distribution is unimodal
- Requires sufficient detections for stable distribution
- May need different methods for different configs

**Validation:**
- Test on Phase 3 configs (known optimal thresholds)
- Compare distribution-suggested thresholds to optimal
- Analyze when method works vs fails

**Results from Step 1 (2025-02-20):**
- **Finding:** Distributions are NOT truly bimodal - they are unimodal but highly skewed
- **Issue:** Bimodality coefficient (BC) can be misleading - high BC (>0.555) can result from high skewness/kurtosis, not just true bimodality
- **Peak Detection:** With robust 5% threshold, only 1 peak detected for both YOLO World and SAM3
- **Initial Problem:** 1% threshold picked up noise as "second peak" (density only 1-4% of first peak's height)
- **Conclusion:** KDE Valley method not applicable - requires true bimodality (2+ peaks)
- **Next Steps:** Need alternative methods that don't rely on bimodality (e.g., embedding space analysis, percentile-based methods with better justification)

### Method 2: Embedding Space Analysis (New - High Potential) ⭐

**Concept:** Use CLIP embedding space to identify high-quality detections without labels.

**Rationale:**
- CLIP embeddings encode semantic similarity
- High-quality detections should have embeddings close to text prompt
- Low-quality detections (FPs) should be farther from prompt
- Can use embedding distance as quality signal
- **Existing infrastructure:** AgVFM codebase has tools for this (see below)

**Implementation Approaches:**

1. **Text-Image Embedding Distance (Primary):**
   - For each detection, extract image patch embedding using CLIP vision encoder
   - Compute cosine distance to text prompt embedding
   - Use distance distribution to identify threshold
   - Low distance = high quality (likely TP)
   - High distance = low quality (likely FP)
   - **Existing code:** `AgVFM/vlme/evaluation/backward_interrogation.py` has `encode_flower_crops()` and CLIP vision model access
   - **Existing code:** `AgVFM/vlme/models/embeddings.py` has `encode_texts()` for text embeddings

2. **Embedding Clustering:**
   - Cluster detection embeddings (all detections across images)
   - Identify clusters that align with text prompt
   - Use cluster membership as quality signal
   - Detections in "prompt-aligned" cluster = high confidence

3. **Multi-Prompt Consensus:**
   - Generate detections with multiple related prompts
   - Detections that appear across prompts = high confidence
   - Use consensus as quality signal
   - Example: "cowpea flower" + "yellow flower" + "legume flower"

4. **Embedding Density:**
   - Compute embedding density around each detection
   - High-density regions = common patterns (likely TPs)
   - Low-density regions = outliers (likely FPs)
   - Use density as quality signal

**Advantages:**
- Uses semantic understanding (CLIP's strength)
- Directly measures alignment with prompt
- Can work per-detection (not just per-image)
- Connects to grounding gap analysis

**Disadvantages:**
- Requires access to CLIP embeddings (may need model modifications)
- Computational cost (embedding extraction)
- May need calibration (distance threshold selection)

**Connection to Grounding Gap:**
- This method directly addresses "grounding" - how well vision aligns with language
- Can identify structural failures (wrong object) vs linguistic failures (prompt mismatch)
- Natural bridge between Phase 4 (deployment) and Phase 5 (understanding)
- **Existing findings:** AgVFM grounding gap analysis found:
  - Text-image cosine similarity = 0.270 (max)
  - Image-image similarity = 0.932 (CLIP knows what flowers look like)
  - **Gap = 0.662** (the distance text must bridge)
  - Text-image similarity correlates with mAP (ρ=0.785) - label-free proxy!

**Validation:**
- Extract embeddings for Phase 3 detections
- Compute distance to prompt embeddings
- Compare distance-based ranking to confidence-based ranking
- Validate that low-distance detections are TPs
- **Leverage existing:** `AgVFM/vlme.backup/probes/confidence_analysis.py` has `clip_rescore_detections()` function

### Method 3: Self-Consistency (New)

**Concept:** Use model's own consistency as quality signal.

**Implementation Approaches:**

1. **Multi-Scale Consistency:**
   - Run detection at multiple image scales
   - Detections that appear at multiple scales = high confidence
   - Use consistency as quality signal
   - Requires multi-scale inference

2. **Spatial Consistency:**
   - Use spatial relationships (flowers shouldn't overlap too much)
   - Detections that violate spatial constraints = low confidence
   - Use NMS-like logic but with quality scoring
   - Example: If two detections overlap >0.9 IoU, lower confidence of both

3. **Temporal Consistency (if available):**
   - If video data, use temporal consistency
   - Detections that persist across frames = high confidence
   - Not applicable to static images

**Advantages:**
- Uses model's own predictions
- No external knowledge needed
- Can work with any model

**Disadvantages:**
- May not be sufficient alone (needs combination with other methods)
- Spatial consistency requires domain knowledge (overlap rules)

### Method 4: Hybrid: Distribution + Embedding (Recommended)

**Concept:** Combine distribution analysis with embedding space analysis.

**Rationale:**
- Distribution analysis identifies natural breaks
- Embedding analysis validates quality of detections
- Combination is more robust than either alone

**Implementation:**
1. Use distribution analysis to identify candidate thresholds
2. For each candidate, compute embedding distances
3. Select threshold where:
   - Distribution suggests natural break AND
   - Embedding distances show clear separation
4. Validate on Phase 3 data

**Advantages:**
- Combines multiple signals
- More robust than single method
- Uses both statistical and semantic information

## Experimental Plan (Revised)

### Step 1: Distribution Analysis (Primary Method) ✅ COMPLETE

**Objective:** Develop and validate confidence distribution analysis.

**Status:** ✅ Complete - Results below

**Tasks:**
1. **Collect confidence distributions:** ✅
   - For each Phase 3 config, collect all confidence scores
   - Per-image and global distributions
   - Include TP/FP labels (for validation only)

2. **Apply distribution methods:**
   - Bimodality detection
   - KDE peak detection
   - Elbow detection
   - Gap analysis

3. **Compare to optimal thresholds:**
   - Do distribution-suggested thresholds match optimal?
   - When does each method work?
   - Which method is most reliable?

**Deliverables:**
- ✅ Distribution analysis tools (`agvfm/analysis/confidence_distribution.py`)
- ✅ Validation results (see Results section below)
- ✅ Best method(s) identified (see Results - KDE valley not applicable)

**Results (2025-02-20):**

#### Data Collection
- **YOLO World (C2_not_bud_calyx):** 8,956 detections collected at conf=0.01
- **SAM3 (C2_not_bud):** 25,692 detections collected at conf=0.01
- **Test set:** 158 images (full test set)

#### Distribution Analysis Results

**YOLO World (C2_not_bud_calyx):**
- **Bimodality Coefficient:** BC = 0.8530 (high BC, but **NOT truly bimodal**)
- **Peak Detection (5% threshold):** 1 peak at 0.028
- **Status:** Unimodal but highly skewed (skewness=2.02, kurtosis=2.98)
- **Suggested Thresholds:**
  - **Elbow Point:** 0.9941 (too high, not useful)
- **Phase 3 Optimal F1 Threshold:** 0.4500 (F1 = 0.4920)
- **Finding:** High BC is due to high skewness, not true bimodality. KDE Valley method not applicable.

**SAM3 (C2_not_bud):**
- **Bimodality Coefficient:** BC = 0.8094 (high BC, but **NOT truly bimodal**)
- **Peak Detection (5% threshold):** 1 peak at 0.023
- **Status:** Unimodal but highly skewed (skewness=3.13, kurtosis=10.35)
- **Suggested Thresholds:**
  - **Elbow Point:** 0.8887 (too high, not useful)
- **Phase 3 Optimal F1 Threshold:** 0.4500 (F1 = 0.5863)
- **Finding:** High BC is due to high skewness, not true bimodality. KDE Valley method not applicable.

**Key Findings:**
1. **Distributions are NOT truly bimodal** - they are unimodal but highly skewed ⚠️
2. **Bimodality coefficient can be misleading** - high BC (>0.555) can result from high skewness/kurtosis, not just true bimodality
3. **Peak detection issue fixed:** Initial 1% threshold picked up noise as "second peak" (only 1-4% of first peak's height)
4. **With robust 5% threshold:** Only 1 peak detected for both models
5. **KDE Valley method not applicable** - requires true bimodality (2+ peaks)
6. **Elbow method produces thresholds that are too high** - not useful for these distributions
7. **Distribution analysis needs alternative methods** - cannot rely on bimodality for threshold selection

**Output Files:**
- Analysis results: `experiments/results/phase4_unlabeled_threshold_selection/distribution_analysis_results.json`
- Summary report: `experiments/results/phase4_unlabeled_threshold_selection/summary_report.txt`
- Visualizations: `experiments/results/phase4_unlabeled_threshold_selection/plots/`

### Step 2: Embedding Space Analysis (High Potential)

**Objective:** Develop embedding-based quality signals.

**Tasks:**
1. **Extract embeddings:**
   - For Phase 3 detections, extract CLIP image embeddings
   - Extract text prompt embeddings
   - Compute distances

2. **Develop quality signals:**
   - Text-image embedding distance
   - Embedding clustering
   - Multi-prompt consensus (if feasible)

3. **Validate quality signals:**
   - Do low-distance detections = TPs?
   - Can distance replace/improve confidence?
   - How does distance-based ranking compare to confidence?

**Deliverables:**
- Embedding analysis tools
- Quality signal validation
- Comparison to confidence-based methods

### Step 3: Hybrid Method

**Objective:** Combine distribution and embedding analysis.

**Tasks:**
1. **Develop hybrid approach:**
   - Use distribution to suggest thresholds
   - Use embeddings to validate/refine
   - Combine signals

2. **Validate on Phase 3:**
   - Does hybrid outperform single methods?
   - Is it more robust across configs?

**Deliverables:**
- Hybrid method implementation
- Validation results

### Step 4: Apply to Phase 1/2 (Selective)

**Objective:** Validate Phase 1/2 rankings using best method(s).

**Approach:** Same as before - top configs only

**Configs:** Top 5 Phase 1 + Top 5 Phase 2 per model = 20 configs

**Methods:**
- Distribution analysis (primary)
- Embedding analysis (if feasible)
- Hybrid (if developed)

**Metrics:**
- Compare rankings at different thresholds
- Validate mAP rankings are stable
- Document F1 ranking changes

## Connection to Grounding Gap Analysis

**Strong Connection:**

1. **Embedding Space Analysis:**
   - Directly measures "grounding" (vision-language alignment)
   - Can identify structural failures (wrong object) vs linguistic failures (prompt mismatch)
   - Natural bridge between Phase 4 and Phase 5

2. **Distribution Analysis:**
   - May reveal calibration issues (confidence doesn't match quality)
   - Can identify when model is overconfident/underconfident
   - Informs understanding of failure modes

**Recommendation:**
- **Phase 4:** Develop embedding-based quality signals (deployment focus)
- **Phase 5:** Deep dive into embedding space to understand failures (analytical focus)
- **Synergy:** Phase 4 tools can inform Phase 5 analysis

## Implementation Priorities

### Priority 1: Distribution Analysis
- **Why:** Most feasible, no model modifications needed
- **Effort:** Low-medium
- **Potential:** Medium (may work for some configs)

### Priority 2: Embedding Space Analysis
- **Why:** High potential, connects to grounding gap
- **Effort:** Medium (need embedding extraction)
- **Potential:** High (uses CLIP's semantic understanding)

### Priority 3: Hybrid Method
- **Why:** Most robust, combines signals
- **Effort:** Medium-high
- **Potential:** High (best of both worlds)

### Priority 4: Self-Consistency
- **Why:** Lower priority, may not be sufficient alone
- **Effort:** Medium
- **Potential:** Medium (useful as supplement)

## Questions to Resolve

1. **Embedding extraction:** ✅ **RESOLVED**
   - ✅ YOLO World: Can access CLIP via `_get_clip_model()` (text) and `_get_clip_vision_model()` (image)
   - ✅ Text embeddings: `encode_texts(yolo_model, texts)` returns (N, 512) tensor
   - ✅ Image embeddings: Can extract from CLIP vision encoder (see `backward_interrogation.py`)
   - ⚠️ SAM3: Need to check if SAM3 exposes CLIP embeddings (may need Hugging Face API)
   - Computational cost: Moderate (need to extract embeddings for all detections)

2. **Distribution analysis:**
   - Per-image vs global distributions?
   - How many detections needed for stable distribution?
   - Which method (bimodality, KDE, elbow) is most reliable?

3. **Validation:**
   - How to validate without labels?
   - Use Phase 3 "optimal" thresholds for validation?
   - Cross-validation approach?

## Next Steps

1. **✅ Investigate embedding access:** **DONE**
   - ✅ YOLO World: CLIP access confirmed (text + vision)
   - ⚠️ SAM3: Check Hugging Face API for CLIP access
   - ✅ Existing code in AgVFM can be adapted

2. **Implement distribution analysis:**
   - Start with Phase 3 data
   - Test all distribution methods (bimodality, KDE, elbow, gap)
   - Compare to optimal thresholds
   - **Leverage:** `AgVFM/vlme.backup/probes/confidence_analysis.py` has `analyze_confidence_distributions()`

3. **Develop embedding analysis:**
   - Extract embeddings for Phase 3 detections
   - Compute distances to prompts
   - Validate quality signals
   - **Leverage:** Existing AgVFM code for text/image embedding extraction
   - **Key insight:** Text-image similarity already shown to correlate with mAP (ρ=0.785)

4. **Combine methods:**
   - Develop hybrid approach (distribution + embedding)
   - Validate on Phase 3
   - Apply to Phase 1/2 subset

5. **Port/adapt existing code:**
   - Review `AgVFM/vlme/models/embeddings.py` for text embedding extraction
   - Review `AgVFM/vlme/evaluation/backward_interrogation.py` for image embedding extraction
   - Review `AgVFM/vlme.backup/probes/confidence_analysis.py` for distribution analysis
   - Adapt to AgVFM2 codebase structure
