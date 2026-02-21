# Phase 5: Embedding Space Analysis for Threshold Selection

> **Status:** 🧪 Planning  
> **Objective:** Use vision-text embedding space to identify confidence thresholds without labeled data  
> **Context:** Phase 4 distribution analysis revealed that confidence distributions are unimodal (not bimodal), making distribution-based methods inapplicable. Phase 5 explores embedding space as an alternative quality signal.

## Problem Framing

In zero-shot object detection, we need to select confidence thresholds without access to labeled validation data. Phase 4 demonstrated that confidence score distributions are unimodal and highly skewed, making traditional distribution-based threshold selection methods (e.g., KDE valley detection) inapplicable. However, both YOLO World and SAM3 use CLIP embeddings to align vision and text representations. We hypothesize that the embedding space contains a quality signal: detections that are semantically similar to the text prompt (low text-image embedding distance) should be higher quality (true positives), while detections with high embedding distance are more likely to be false positives. This suggests we can use embedding distance as a proxy for detection quality and identify thresholds where embedding distance distributions show natural separation between high-quality and low-quality detections. The key challenge is determining how to translate embedding distance distributions into actionable confidence thresholds, and whether this approach generalizes across different models (YOLO World vs SAM3) and prompt configurations.

## Core Hypothesis

**Hypothesis:** Text-image embedding distance (cosine distance between CLIP text prompt embedding and CLIP image patch embedding) correlates with detection quality. Low-distance detections are more likely to be true positives, high-distance detections are more likely to be false positives. We can use embedding distance distributions to identify natural thresholds that separate high-quality from low-quality detections, then map these embedding distance thresholds back to confidence score thresholds.

## Key Questions

1. **Does embedding distance correlate with detection quality?**
   - Do true positives have lower embedding distances than false positives?
   - What is the distribution of embedding distances for TPs vs FPs?
   - Is the correlation strong enough to be useful?

2. **Can we identify thresholds in embedding space?**
   - Do embedding distance distributions show natural separation (bimodality, gaps, elbows)?
   - Can we find embedding distance thresholds that separate TPs from FPs?
   - How do these thresholds compare to optimal confidence thresholds?

3. **How do we map embedding thresholds to confidence thresholds?**
   - Given an embedding distance threshold, what confidence threshold yields similar filtering?
   - Is the mapping consistent across models and configurations?
   - Can we use embedding distance to directly rescore detections?

4. **Does this generalize?**
   - Does the approach work for both YOLO World and SAM3?
   - Does it work across different prompt configurations?
   - Is it more robust than confidence-based methods?

## Initial Experiments (Brainstorming)

### Experiment 1: Embedding Distance vs Detection Quality (Validation)

**Objective:** Validate that embedding distance correlates with detection quality.

**Approach:**
1. For Phase 3 best configs (YOLO World C2_not_bud_calyx, SAM3 C2_not_bud):
   - Run inference at very low confidence threshold (0.01) to get all detections
   - Extract CLIP image embeddings for each detection crop
   - Extract CLIP text embeddings for the prompt
   - Compute cosine distance between text and image embeddings
   - Match detections to ground truth (using IoU threshold 0.5)
   - Label each detection as TP or FP

2. Analyze:
   - Distribution of embedding distances for TPs vs FPs
   - Statistical tests: Mann-Whitney U test, effect size (Cohen's d)
   - Visualization: Overlapping histograms, ROC curve (using embedding distance as score)
   - Correlation: Spearman correlation between embedding distance and TP/FP label

**Expected Outcomes:**
- If hypothesis is correct: TPs should have significantly lower embedding distances than FPs
- Effect size should be large (d > 0.8) for the method to be useful
- ROC AUC should be > 0.7 (ideally > 0.8) for embedding distance as a quality signal

**Why This First:**
- Validates the core hypothesis before building threshold selection methods
- Uses existing Phase 3 data (no new inference needed)
- Provides baseline metrics for comparison

### Experiment 2: Embedding Distance Distribution Analysis

**Objective:** Identify natural thresholds in embedding distance space.

**Approach:**
1. For all detections from Experiment 1:
   - Plot embedding distance distribution (histogram + KDE)
   - Apply distribution analysis methods (bimodality, KDE peaks, elbow, gaps)
   - Check if embedding distance distributions show better separation than confidence distributions

2. Compare to confidence distributions:
   - Do embedding distance distributions show clearer bimodality?
   - Are the peaks/valleys more distinct?
   - Does the separation align with TP/FP separation?

**Expected Outcomes:**
- Embedding distance distributions may show clearer bimodality than confidence distributions
- If bimodal, KDE valley method could work for embedding distance thresholds
- Could identify embedding distance thresholds that separate TPs from FPs

**Why This Second:**
- Builds on Experiment 1 validation
- Tests if embedding space has better structure than confidence space
- Could recover the KDE valley method that failed for confidence distributions

### Experiment 3: Embedding Distance Threshold to Confidence Threshold Mapping

**Objective:** Map embedding distance thresholds to confidence score thresholds.

**Approach:**
1. For each embedding distance threshold identified in Experiment 2:
   - Filter detections by embedding distance threshold
   - Find the confidence threshold that yields the same number of detections
   - Compare this confidence threshold to Phase 3 optimal F1 threshold

2. Alternative approach:
   - For each confidence threshold in Phase 3 sweep:
     - Compute mean embedding distance of detections above threshold
     - Plot confidence threshold vs mean embedding distance
   - Find confidence threshold where mean embedding distance matches embedding distance threshold

**Expected Outcomes:**
- Mapping between embedding distance and confidence thresholds
- Validation: Do mapped confidence thresholds match Phase 3 optimal thresholds?
- Understanding of the relationship between embedding space and confidence space

**Why This Third:**
- Connects embedding space analysis to actionable confidence thresholds
- Validates that embedding-based thresholds are useful for deployment
- Provides method for zero-shot threshold selection

### Experiment 4: Embedding-Based Rescoring

**Objective:** Use embedding distance to directly rescore detections (alternative to confidence).

**Approach:**
1. For each detection:
   - Compute embedding distance score
   - Convert to similarity score (1 - distance, or normalized)
   - Use embedding similarity as new confidence score

2. Evaluate:
   - Run evaluation with embedding similarity as confidence
   - Compare mAP, F1, Precision, Recall to confidence-based evaluation
   - Find optimal embedding similarity threshold
   - Compare to confidence-based optimal threshold

**Expected Outcomes:**
- Embedding similarity may be a better quality signal than raw confidence
- Could outperform confidence-based methods
- Provides alternative deployment strategy

**Why This Fourth:**
- Tests if embedding space is not just a threshold selection tool, but a better quality signal
- Could lead to improved detection performance
- More direct application than threshold mapping

### Experiment 5: Cross-Model and Cross-Config Generalization

**Objective:** Test if embedding-based methods generalize across models and configurations.

**Approach:**
1. Run Experiments 1-3 for:
   - All Phase 3 configs (YOLO World: C1, C2_not_bud_calyx, C2_not_calyx; SAM3: C2_not_bud, C_color_taxonomy_anatomy_tiny, C2)
   - Compare embedding distance distributions across configs
   - Check if embedding distance thresholds are consistent

2. Analyze:
   - Do embedding distance distributions vary by prompt?
   - Are embedding distance thresholds model-specific or general?
   - Can we find universal embedding distance thresholds?

**Expected Outcomes:**
- Understanding of how prompt affects embedding space
- Identification of model-specific vs universal thresholds
- Validation of method robustness

**Why This Fifth:**
- Tests generalizability (critical for deployment)
- Identifies limitations and edge cases
- Provides guidance on when method works vs fails

### Experiment 6: Per-Image vs Global Embedding Analysis

**Objective:** Compare per-image embedding analysis to global analysis.

**Approach:**
1. For each image:
   - Compute embedding distances for all detections in that image
   - Identify embedding distance threshold per image (using distribution methods)
   - Map to confidence threshold per image

2. Compare:
   - Per-image thresholds vs global thresholds
   - Which approach yields better performance?
   - Is per-image adaptation beneficial?

**Expected Outcomes:**
- Understanding of whether embedding space structure varies per image
- Validation of global vs adaptive threshold selection
- Potential for improved performance with per-image thresholds

**Why This Sixth:**
- Tests if adaptation improves performance
- Explores whether embedding space structure is image-dependent
- Could lead to more sophisticated threshold selection

## Implementation Considerations

### Data Requirements
- Phase 3 inference results (already have)
- Ground truth labels (for validation only)
- CLIP model access (text and vision encoders)

### Computational Requirements
- CLIP embedding extraction for all detections
- Distribution analysis (similar to Phase 4)
- Statistical analysis and visualization

### Validation Strategy
- Use Phase 3 optimal thresholds as ground truth (for validation only)
- Compare embedding-based thresholds to optimal thresholds
- Measure correlation and error rates

### Success Criteria
- Embedding distance correlates strongly with detection quality (d > 0.8, AUC > 0.7)
- Embedding-based thresholds are within 10% of optimal thresholds
- Method works for both YOLO World and SAM3
- Provides actionable threshold selection method

## Analysis of ChatGPT5.2's Approach

### Strengths

1. **Semantic Margin (Target vs Hard Negatives)**
   - Addresses "generic flower" problem (CLIP scores plant-ish backgrounds highly)
   - Forces discrimination: "flower-ness" must beat "plant-ness"
   - Aligns with Phase 2 findings (negation helps, confusers known)

2. **Per-Image Adaptation**
   - Handles variable object counts (0-50 flowers per image) naturally
   - Addresses context leakage (small flowers in cluttered scenes)
   - No brittle global threshold

3. **FDR Control (Benjamini-Hochberg)**
   - Principled statistical method (not arbitrary)
   - Naturally handles multiple testing per image
   - Adapts to image difficulty (0 objects → keeps none, many objects → keeps many)

4. **Background Sampling for P-Values**
   - Normalizes per-image (handles different image contexts)
   - Provides statistical significance measure
   - Handles 0-object images naturally

### Critical Considerations

1. **SAM3 Embedding Projection** ⚠️ **CRITICAL CHECKPOINT**
   - **Issue:** SAM3 vision encoder outputs patch embeddings (1024-d), text encoder outputs (512-d)
   - **Need:** Shared embedding space for meaningful similarity computation
   - **Action Required:** 
     - Check for built-in projection: `model.vision_proj` / `model.visual_projection` / `image_embeds` in forward pass
     - If exists: use it
     - If not: need pooling + projection strategy (Route B in ChatGPT5.2 feedback)
   - **Status:** ✅ Embeddings accessible, ❓ Projection to be confirmed

2. **Computational Cost**
   - Background sampling: 128-256 samples/image × 158 images = 20,224-40,448 embeddings (reduced from 200-1000)
   - Multi-scale crops: 2 scales × all candidates = 2× embeddings
   - Total: ~144k region embeddings (manageable with batching on GPU)
   - **Mitigation:** Start with K=128, two crop scales, batch processing, cache text embeddings

3. **Cowpea-Specific Hard Negatives**
   - ChatGPT5.2 examples are grape-specific; we need cowpea-specific prompts
   - **Based on Phase 2 findings:**
     - Targets: "cowpea flower", "yellow cowpea flower", "cowpea flower with open petals", "a single yellow cowpea flower", "vigna unguiculata flower"
     - Hard Negatives: "cowpea bud", "cowpea calyx", "cowpea leaf", "cowpea stem", "cowpea pod", "soil", "shadow", "sun glare"

4. **Progressive Implementation Strategy**
   - Start with simpler validation, add complexity progressively
   - Validate each component before adding the next
   - Provides fallback if full pipeline is too complex

### Label-Free Evaluation (Recommended by ChatGPT5.2)

Instead of proving "distance correlates with quality" (requires GT), use:
1. **Empty-image suppression test:** % images with 0 kept detections (should be high for negative-only sets)
2. **Prompt invariance:** True detections persist across prompt paraphrases; junk churns
3. **Augmentation stability:** Kept detections stable under brightness/contrast jitter
4. **Cross-model agreement:** YOLO World vs SAM3 agreement increases after semantic filtering

## Implementation Plan

### Phase 5A: Critical Checkpoints & Setup

1. **Confirm SAM3 Image Projection** ⚠️ **BLOCKER**
   - Check for `model.vision_proj` / `model.visual_projection` / `image_embeds` in SAM3 forward pass
   - If found: implement `encode_image_crops()` returning (B, 512) normalized embeddings
   - If not: implement Route B (pooling + projection strategy)

2. **Build Cowpea Prompt Banks**
   - Targets: 5-15 prompts (mix plain and botanical phrasing)
   - Hard Negatives: 10-30 prompts (based on Phase 2 confusers)
   - Cache text embeddings for efficiency

3. **Verify YOLO World Region Embeddings**
   - Confirm `encode_flower_crops()` works for region-level embeddings
   - Test multi-scale crop extraction

### Phase 5B: Core Pipeline Implementation

1. **Candidate Generation**
   - Run detectors at very low conf floor (0.001-0.01)
   - Keep permissive NMS (we'll clean later with semantic filtering)

2. **Semantic Margin Computation**
   - Multi-scale crops: tight + medium (add wide if needed)
   - For each candidate: `margin = max_sim(e_img, e_txt_target) - max_sim(e_img, e_txt_neg)`
   - Take max margin across crop scales/variants

3. **Per-Image P-Values (Background Sampling)**
   - Sample K=128-256 background regions per image
   - Match region sizes/aspect ratios to candidates
   - Reject samples with IoU > 0.1-0.2 with candidates
   - Compute empirical p-value: `p = (1 + count(m_bg >= m)) / (1 + K)`

4. **BH-FDR Selection**
   - Apply Benjamini-Hochberg per image with q=0.10 (test {0.05, 0.10, 0.20})
   - Behavior: empty images → keep none, dense images → keep many

5. **Post-Processing**
   - Dedup/NMS after semantic filtering
   - Spatial consistency cleanup

### Phase 5C: Evaluation & Validation

1. **Label-Free Sanity Checks**
   - Empty-image suppression test
   - Prompt invariance test
   - Augmentation stability test
   - Cross-model agreement test

2. **Optional: Semantic-Purity Confidence Threshold**
   - If needed for downstream: sweep confidence threshold
   - Choose smallest c that yields stable purity proxy
   - Often can keep c very low (semantic gate does filtering)

## Next Steps (Prioritized)

1. ⚠️ **CRITICAL:** Confirm SAM3 image projection to 512-d space
2. Build cowpea prompt banks (targets + hard negatives)
3. Implement semantic margin computation utilities
4. Implement per-image p-value computation (K=128, two scales)
5. Implement BH-FDR selection
6. Run label-free sanity suite
7. Compare YOLO-only vs SAM3-only vs intersection (agreement)

## ChatGPT5.2 Thinking Proposed Approach
Here are the thoughts from another llm:

# Zero-Shot Thresholding for Fine-Grained Agricultural Object Detection
_A practical, label-free plan using (internal) vision–text embedding signals to choose thresholds and suppress false positives, designed for images with 0–50 objects._

## 0) One last critical pass (what I’d change / emphasize)
Your core hypothesis (“embedding space contains a quality signal”) is directionally right, but **raw target similarity / distance alone is fragile** in fine-grained ag scenes because:

- **Context leakage dominates**: small flowers/pods/fruit occupy few pixels; crops/masks include lots of canopy/background.
- **“Flower” is too generic**: CLIP-like spaces often score “plant-ish” background highly.
- **Internal embedding scales are not comparable across models** (YOLO World vs SAM3), and may drift with prompt phrasing.

**Key updates to make it robust and actionable without labels:**
1. Use a **semantic margin** (target vs hard negatives), not just target distance/similarity.
2. Normalize semantic scores **per image** using a background distribution → convert to **p-values**.
3. Replace “pick a global confidence threshold” with **per-image FDR control** (Benjamini–Hochberg), which naturally handles 0–50 objects per image and often removes the need for a brittle conf cutoff.
4. If you still need a confidence threshold (runtime/UI), derive it **from semantic purity**, not from confidence histograms.
5. Compare YOLO World vs SAM3 using **rank/p-value behavior**, not absolute embedding distances (unless you later add an external CLIP alignment).

Everything below is built around those updates.

---

## 1) Goals & constraints
### Goals
- Choose thresholds / filtering rules **without labeled validation**.
- Maintain recall for small, subtle targets (flowers/pods/fruit/leaves).
- Strongly suppress false positives on **0-object images**.
- Provide a method that is **consistent across images** and reasonably comparable across YOLO World vs SAM3.

### Constraints
- Currently relying on **internal embeddings** from each model (not a shared external CLIP).
- Images may contain **0 to ~50 instances**.
- Need an exportable, implementable plan for an LLM agent (Cursor).

---

## 2) Core idea
Each candidate detection has:
- a detector confidence `conf`
- a region-level image embedding `e_img` (from the model’s vision–text embedding space)
- a set of target text embeddings `e_txt_target`
- a set of hard-negative text embeddings `e_txt_neg`

We compute a **semantic margin** per candidate:
- `sT = max_sim(e_img, e_txt_target)`
- `sN = max_sim(e_img, e_txt_neg)`
- `margin = sT - sN`

Then we turn margin into a **per-image p-value** by comparing to a background margin distribution sampled from the same image.

Finally we use **Benjamini–Hochberg (BH) FDR** per image to decide which detections to keep.

This yields a label-free filter that adapts to images with 0 objects vs 50 objects.

---

## 3) End-to-end pipeline (per model)
Run this separately for YOLO World and SAM3 first.

### Step A — Candidate generation (high recall)
- Run the detector with a **very low conf floor** to gather candidates:
  - YOLO World: `conf_floor = 0.001–0.01` (and keep NMS fairly permissive; we’ll clean later)
  - SAM3: generate proposals/masks as you already do; keep generous candidates

**Rationale:** confidence is not calibrated; we want to avoid dropping true positives early.

### Step B — Prompt bank (targets + hard negatives)
Create a small bank per task (flowers/pods/fruit/etc).

**Targets (T):** 5–15 prompts mixing plain and botanical phrasing. Examples for grape flower:
- “grape flower”
- “grapevine flower”
- “grape inflorescence”
- “tiny grape flowers on rachis”
- “grape bloom”
- “capfall on grape flowers” (if relevant)

**Hard negatives (N):** 10–30 prompts for common confusions in your domain:
- “grape leaf”, “leaf”, “leaf vein”
- “stem”, “shoot”, “cane”, “cordon”
- “tendril”
- “bud”
- “cluster”, “berry” (if detecting flowers; include stage-adjacent negatives!)
- “wire”, “trellis”, “trellis clip”, “post”
- “soil”, “shadow”, “specular highlight”, “sun glare”

> Use **margin** specifically to force “flower-ness” to beat “plant-ness”.

### Step C — Region embedding extraction (avoid context traps)
For each candidate region, build region crops for embedding:
- Use **multi-scale crops** to reduce sensitivity:
  - tight crop (box/mask bounds)
  - medium crop (pad by 10–20%)
  - wide crop (pad by 30–50%)
- Compute margin for each scale and take:
  - `margin = max(margin_tight, margin_med, margin_wide)`

For SAM3 masks:
- embed the **masked crop** (background zeroed / blurred) AND a lightly dilated version.
- take best margin across variants.

### Step D — Background sampling and per-image p-values
Per image, build a background margin distribution `M_bg`:

**Background sampling:**
- Sample `K = 200–1000` random regions per image.
- Match sampled region sizes/aspect ratios to candidate boxes (important).
- Reject samples with high overlap with any candidate detection (IoU > 0.1–0.2).
- Compute margin for each background region the same way (same multi-scale rules).

**Convert candidate margin to p-value:**
For candidate i with margin `m_i`:
- `p_i = (1 + count(M_bg >= m_i)) / (1 + K)`  (a smoothed empirical p-value)

Interpretation: “How surprising is this region’s semantic margin compared to background in this image?”

This step is the **workhorse** for handling 0-object images.

### Step E — Per-image selection via BH-FDR
Let an image have `M` candidates, with p-values `p_1..p_M`.

Run Benjamini–Hochberg (BH) at level `q`:
1. Sort p-values: `p_(1) <= ... <= p_(M)`
2. Find largest `k` such that: `p_(k) <= (k/M) * q`
3. Keep detections with `p_i <= p_(k)`

**Default:** start with `q = 0.10` (tune later without labels; see Section 6).

Behavior you want:
- If image has no flowers → p-values ~ uniform → BH keeps none.
- If image has many true flowers → many low p-values → BH keeps many.

### Step F — Dedup / spatial consistency cleanup (post semantic gate)
After BH selection:
- Run NMS / merging to remove duplicates (especially in dense scenes):
  - If boxes: Soft-NMS or standard NMS with IoU 0.3–0.5 (task-dependent)
  - If masks: merge by mask IoU / containment rules
- Optionally apply a “one per small neighborhood” rule if you see bursty duplicates.

### Step G — Optional: derive a confidence threshold (if needed)
If you must output a single `conf_thresh` for downstream:

**Semantic-purity thresholding:**
- Sweep candidate confidence threshold `c` over a grid (e.g., 0.0→0.5).
- For each `c`, run Steps C–F on candidates with `conf >= c`.
- Compute:
  - `N_kept(c)` per image (and summary stats)
  - `PurityProxy(c)`: e.g., median margin of kept set, or fraction with `p <= 0.01`
- Choose the **smallest** `c` that yields stable `PurityProxy` and doesn’t explode `N_kept`.

In many cases you’ll find you can keep `c` very low because the semantic gate is doing the real filtering.

---

## 4) Model-specific notes
### YOLO World
- Candidate generation: set low `conf_floor`, permissive NMS.
- Embeddings: if YOLO World exposes region-level embeddings tied to detections, use those. If it exposes only global image embeddings, you’ll need to embed crops through the internal image encoder (or fall back to an external CLIP later).

### SAM3
- Candidate generation: you may have fewer but higher-quality proposals; still keep generously.
- Masks reduce background leakage, but small masks can become too “textureless”; multi-scale + masked/dilated variants help.

### Cross-model comparison (with internal embeddings)
Avoid comparing raw margins across models. Compare:
- fraction of images with 0 kept detections (should be high in negative-only sets)
- distribution of kept counts on dense scenes
- stability under augmentations (Section 6)
- agreement rates between models (overlap in kept detections)

---

## 5) Implementation pseudocode (per model)
```python
def zero_shot_filter(
    image,
    candidates,          # list of {region, conf, (optional) e_img_internal}
    embed_region_fn,      # returns e_img for a region crop
    text_embed_fn,        # returns e_txt for prompts
    target_prompts,
    neg_prompts,
    K_bg=500,
    q_fdr=0.10,
):
    # Precompute text embeddings
    E_T = [text_embed_fn(p) for p in target_prompts]
    E_N = [text_embed_fn(p) for p in neg_prompts]

    def margin_for_region(region):
        # Multi-scale crops / variants
        margins = []
        for variant in make_region_variants(region):  # tight/med/wide (+ masked/dilated for masks)
            e = embed_region_fn(image, variant)
            sT = max(sim(e, t) for t in E_T)
            sN = max(sim(e, n) for n in E_N)
            margins.append(sT - sN)
        return max(margins)

    # Candidate margins
    cand_m = [margin_for_region(c.region) for c in candidates]

    # Background distribution
    bg_regions = sample_background_regions(image, candidates, K=K_bg)
    bg_m = [margin_for_region(r) for r in bg_regions]
    bg_m_sorted = sorted(bg_m)

    # p-values (empirical tail probability)
    def p_value(m):
        # count(bg_m >= m) via binary search
        idx = lower_bound(bg_m_sorted, m)
        ge = len(bg_m_sorted) - idx
        return (1 + ge) / (1 + len(bg_m_sorted))

    pvals = [p_value(m) for m in cand_m]

    # BH-FDR
    keep_idx = benjamini_hochberg(pvals, q=q_fdr)

    kept = [candidates[i] for i in keep_idx]

    # Dedup / NMS after semantic filtering
    kept = spatial_dedup(kept)

    return kept, {"pvals": pvals, "margins": cand_m}
```

## Updated response from GPT5.2

Updated Response to Share With Agent (Phase 5 Plan Revision)
1) Key update: SAM3 embedding access is sufficient for the full pipeline

Your test confirms:

Text embeddings are straightforward: model.text_encoder(**tokens).text_embeds → shape (B, 512)

Vision embeddings are patch tokens: model.vision_encoder(**image_inputs).last_hidden_state → shape (B, P, 1024)

This is enough to:

embed region crops (boxes or masks)

compute target vs hard-negative semantic margin

compute per-image null distributions via background crops

implement FDR / BH selection for per-image adaptive filtering

So: no blocker. Proceed.

2) Critical correction: SAM3 “vision embeddings” need a pooling/projection step

SAM3’s vision encoder gives patch embeddings in a 1024-d space; text embeddings are 512-d in a projection space.

To compute meaningful similarity between region crops and text, you need a shared embedding space. There are two viable routes:

Route A (preferred if model provides it): use SAM3’s built-in projection / similarity head

Check whether SAM3 has a projection layer that maps vision tokens → same 512-d space as text_embeds, e.g. something like:

model.vision_proj / model.visual_projection

or a CLIP-like “image_embeds” output somewhere in the model forward pass

If it exists: use it.

Route B (if no projection exists): build a region embedding by pooling + learned-free mapping

If you can’t find a built-in projection, do:

pool patch embeddings into a single vector:

mean pool: v = mean(last_hidden_state, dim=1) → (B, 1024)

optionally exclude CLS-like tokens if present

or attention pool if model provides attention maps (optional)

normalize for cosine similarity:

v = v / ||v||

For text, you already have t = text_embeds (B, 512) normalized.

But note: 1024 vs 512 dims means you still can’t cosine compare directly. So you either:

locate the correct vision projection, or

temporarily use vision–vision comparisons only (not what we want), or

introduce an external CLIP scorer (not necessary if SAM3 has projection, which many models do).

Action item: agent must confirm the presence of SAM3 image projection into 512-d CLIP space. The fact that text encoder is CLIPTextModelWithProjection strongly suggests a corresponding image projection exists somewhere.

3) Updated pipeline: implement the full method per model (YOLO World and SAM3)

This is the pipeline I recommend implementing now that SAM3 embeddings are accessible.

Step A — Candidate generation (high recall)

Run each detector at a low conf floor (keep many candidates).

Keep post-NMS candidates but permissive.

Step B — Cowpea prompt banks

Build these immediately (based on Phase 2 confusers).

Targets (T):

“cowpea flower”

“yellow cowpea flower”

“cowpea flower with open petals”

“a single yellow cowpea flower”

“vigna unguiculata flower” (test; may help)

Hard negatives (N):

“cowpea bud”, “flower bud”

“cowpea calyx”, “green calyx”

“cowpea leaf”, “leaf”

“cowpea stem”, “stem”, “shoot”

“cowpea pod”, “young pod”

“soil”, “ground”

“shadow on leaf”, “shadow”

“sun glare”, “specular highlight”, “bright reflection”

(optional) “yellow leaf spot” if you see that confound

Step C — Region crop embedding extraction

For each candidate region (box or mask), compute region embeddings:

Use two crop scales first:

tight crop

medium crop (+15–25% padding)

Add wide crop (+40–50%) only if small flowers are too textureless.

For SAM3 masks:

embed masked crop (background removed/blurred) AND slightly dilated mask variant.

Step D — Semantic margin per candidate

Compute for each candidate region r:

sT(r) = max_{t in T} cos(e_img(r), e_txt(t))

sN(r) = max_{n in N} cos(e_img(r), e_txt(n))

margin: m(r) = sT(r) - sN(r)

Take m(r) as the max across crop scales/variants.

Step E — Per-image null distribution + p-values (background sampling)

Per image, sample K background regions matched to candidate sizes/aspects.

Compute background margins m_bg the same way.

Empirical p-value:

p(r) = (1 + count(m_bg >= m(r))) / (1 + K)

Compute control defaults:

start with K=128–256 (not 500–1000)

two crop scales only

cache text embeddings; batch region embedding calls

skip bg sampling if image has < ~10 candidates (optional)

Step F — Per-image selection using BH-FDR

Apply Benjamini–Hochberg per image with q:

start with q=0.10

test {0.05, 0.10, 0.20}

This gives the desired behavior:

empty images → keep none

dense images → keep many (adaptive)

Step G — Dedup / merging

After semantic selection, dedup:

boxes: NMS / Soft-NMS

masks: mask IoU merge rules

4) How to evaluate without labels (better than “distance vs quality” first)

I recommend not spending time proving “distance correlates with quality” (hard without GT). Instead run these label-free checks:

Empty-image suppression test
On a subset of images believed to have 0 flowers: measure % images with any kept detections (should be low).

Prompt invariance
Use 5–10 paraphrases of target prompts. True detections should persist; junk should churn.

Augmentation stability
Brightness/contrast jitter: kept detections should be stable.

Cross-model agreement
Compare YOLO World vs SAM3 after semantic filtering: agreement should increase vs confidence-only filtering.

These are fast and highly diagnostic.

5) Model-specific implementation notes
YOLO World

You already have utilities like encode_flower_crops() and encode_texts() — likely enough to build region embeddings and margins.

SAM3

You can embed crops by running vision encoder on region crops; but ensure:

you use the correct image projection to 512-d (see Section 2)

you normalize embeddings before cosine similarity

you batch crops for speed

6) Practical guidance on compute

A reasonable first run:

~158 images

assume 200 candidates/image @ low conf floor (worst-case), but often fewer after NMS

candidates: 158×200×2 scales = 63k region embeddings

background: 158×256×2 scales = 81k region embeddings
Total ~144k region embeddings

That’s high but manageable if batched and run on GPU. Reduce cost by:

K=128 initially

only compute background on images with many candidates

reuse crop tensors for both models if feasible

7) Immediate “next steps” for agent

Confirm SAM3 image projection to 512 exists and expose encode_image_crops() returning (B,512) normalized.

Build cowpea prompt banks (targets + hard negatives).

Implement semantic margin + per-image p-values (with K=128) + BH-FDR.

Run label-free sanity suite (empty suppression, invariance, stability).

Compare YOLO-only vs SAM3-only vs intersection (agreement).

Short conclusion

Given SAM3 embeddings are accessible, you can implement the full semantic-margin + per-image null + BH-FDR pipeline now. The only critical engineering checkpoint is ensuring SAM3 crop embeddings live in the same projected space as text embeddings (likely a built-in projection). Once that’s confirmed, proceed with K=128–256 and two crop scales, and validate with invariance/stability + empty-image suppression rather than trying to prove “distance correlates with quality” first.

## GPT5.2 Analysis of phase_5_results_summary.pdf

I read through the Phase 5 results summary, and I think the pipeline is behaving “as designed” statistically, but in a way that’s guaranteed to be too conservative given your current p-value scale and the per-image BH setup.

Below are (A) likely “mistakes / mismatches” in how the method was instantiated vs the intent, (B) what the numbers are telling you diagnostically, and (C) concrete next steps that should quickly recover recall without giving back all the FP reduction.

A) Places where the current setup is likely “wrong” (or at least mis-specified)
1) Per-image BH-FDR at q=0.10 is implicitly much stricter than “keep p ≤ 0.10”

Your doc reports that 61.8% of candidates have p-values ≤ 0.10. 

phase5_results_summary


But BH-FDR “q=0.10” does not mean “keep p ≤ 0.10.” With ~19 candidates/image on average 

phase5_results_summary

, the first BH cutoff is roughly:

to keep even 1 detection, you typically need p ≤ q/M ≈ 0.10/19 ≈ 0.005

to keep a few, you still need p-values in the low hundredths

So if your p-values aren’t routinely in the 0.001–0.01 regime, BH will often keep none—explaining why 87.3% of images end up empty. 

phase5_results_summary

This is the biggest “spec mismatch”: you wanted adaptive thresholding, but the chosen statistical control (per-image BH at q=0.10) effectively demands very strong evidence per image.

2) Your semantic margin signal is weak, so “very strong evidence” almost never appears

Margins are tightly centered near zero (mean -0.010, std 0.024), and kept vs rejected overlap heavily. 

phase5_results_summary


That’s consistent with the observed outcome: huge FP drop, huge TP drop. 

phase5_results_summary

This suggests at least one of these is true:

crops are too contaminated by background (tiny flowers)

prompts (targets/negatives) aren’t separating well in the embedding space

“max(target) - max(negative)” is too brittle (one strong negative prompt dominates)

the null/background distribution is too “similar” to candidates (or sampling is biased), so candidates don’t look significant relative to background

The doc itself flags background sampling bias as a concern. 

phase5_results_summary

3) Candidate generation at conf=0.01 may not be the right operating point for semantic filtering

You used conf=0.01 to generate 3,005 candidates (19/image). 

phase5_results_summary


That set has recall 0.521 but precision 0.172 (2489 FP). 

phase5_results_summary

Semantic gating works best when it’s deciding among “plausible” candidates. At very low conf, you’re giving it a lot of junk where the embedding score has little reason to be well-behaved.

This doesn’t mean “raise conf and be done,” but it strongly motivates a hybrid gate (confidence first, then semantic refinement for the gray zone).

B) What the results imply (diagnosis)
1) The method is currently acting like a precision-only filter

You cut FP by 96.7% (2489 → 82), but also cut TP by 95% (516 → 26). 

phase5_results_summary


That’s why recall collapses to 0.026 and F1 collapses. 

phase5_results_summary

If your product requirement is “don’t hallucinate flowers when none exist,” this is a useful failure mode—it proves you can get extremely low FP. But you need a control knob that restores recall.

2) The p-values are not “tiny enough” for per-image BH

Median p-value is 0.0465 overall, and median kept p-value is 0.0155. 

phase5_results_summary


Those aren’t extreme. They’ll rarely survive BH when M≈10–20 unless q is high.

So the fix is not mysterious: either (a) change the selection rule, or (b) strengthen the semantic signal so p-values become far smaller.

C) Suggested next steps (highest ROI first)
Step 1 — Change selection rule before changing everything else

Do these three ablations immediately (they’re quick and will tell you what’s salvageable):

Raise q a lot
Try q ∈ {0.2, 0.3, 0.5}. The report itself suggests this direction. 

phase5_results_summary


Expectation: recall should come back fast, precision will fall somewhat, but you’ll likely land at a better F1.

Replace per-image BH with a simpler per-image rule
Two strong options:

Fixed p-value cutoff per image: keep p ≤ 0.05 (or 0.1) + dedup
(This aligns with the intuitive reading of p-values and avoids the q/M harshness.)

Top-k per image by margin (or by p) with a cap (e.g., k=3–10), optionally conditioned on p ≤ 0.2
(This is extremely effective in 0–50 object regimes.)

Try global BH across all candidates
Your own doc notes per-image BH may be too conservative and suggests global. 

phase5_results_summary


Global BH avoids the “q/M per image” penalty and tends to behave closer to what people expect from FDR.

If any of these recover recall meaningfully, you’ve learned the main issue is the selection policy, not necessarily the embedding signal.

Step 2 — Add a hybrid gate (confidence + semantic)

The doc already lists this as a medium-term idea; I’d promote it to immediate. 

phase5_results_summary

A very practical hybrid:

If conf ≥ c_hi (e.g., 0.3), keep (or lightly semantic-check)

Else if conf in [c_lo, c_hi) (e.g., 0.01–0.3), apply semantic filter

Else discard

This makes semantic filtering focus on the ambiguous region where it can help, while preserving high-confidence true positives that your current filter is discarding.

Step 3 — Strengthen the semantic signal (don’t jump to this until Step 1–2)

Your margins being near-zero is a clear warning. 

phase5_results_summary


Here are the highest-leverage fixes:

Prompt bank upgrades

Add more visual descriptors (“small yellow cowpea flower”, “open petals”, “flower cluster on plant”)

Add “photometric confusers” explicitly (shadow, glare, reflection)

Consider adding stage-adjacent negatives (pods, buds, calyx) in multiple phrasings

Aggregation change
Instead of max(target) - max(neg) (very brittle), try:

logsumexp(target) - logsumexp(neg) (softmaxed max)

or average of top-2 target sims minus top-2 negative sims

This often increases separation without labels.

Crop strategy
You only used padding 0.0 and 0.2. 

phase5_results_summary


For tiny flowers, you often need either:

a wider context crop (0.4–0.6) or

a masked/attenuated background crop (if you can approximate a mask)
Otherwise, embeddings largely see “leaf canopy” and your signal stays near 0.

Step 4 — Fix the background null (if needed)

The report correctly flags that random non-overlapping regions may not represent the “hard background” distribution. 

phase5_results_summary

Two better nulls:

near-candidate background: sample regions near detections but offset (captures leaf edges, clutter, flower-like textures)

candidate-derived null: treat the lower 80–90% of candidate margins as “background-like” and compute p-values vs that distribution (much cheaper and often more realistic)

One “potential mistake” to check in code (fast sanity check)

Because the p-values are central to everything:

Verify you’re computing cosine similarity with L2-normalized embeddings for both image and text. If either side isn’t normalized, margin magnitudes compress toward 0 and become hard to separate (which matches your observed “near zero” margins). The report doesn’t mention normalization, so it’s worth a quick check.

What I’d do next if I were driving the repo (ordered)

Run q sweep {0.1,0.2,0.3,0.5} with the existing setup.

Implement fixed p cutoff (0.05/0.1) and top-k per image; compare.

Implement global BH across all 3005 candidates.

Add confidence+semantic hybrid gate.

Only then: prompt bank + logsumexp aggregation + crop scale 0.5.

Update background sampling to near-candidate null if still needed.

If you want, paste the snippet of your BH implementation and p-value computation loop (just those functions) and I’ll look for any subtle off-by-one / sorting / per-image grouping mistakes that could be making it harsher than intended.