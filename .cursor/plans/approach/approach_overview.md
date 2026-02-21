# Research Approach: AgVFM2

> **Purpose:** Document the systematic approach for comparing vision foundation models on zero-shot agricultural detection

## Core Strategy

Create a clean, reproducible framework for:
1. **Systematic model comparison** (YOLO World vs SAM3, extensible to more)
2. **Quantitative understanding** of how different VFMs link language and vision for botany/agriculture
3. **Prompt engineering rules** specific to each model architecture
4. **Grounding gap analysis** (next phase) to understand failure modes

## Experimental Framework

### Phase 1: Structured Factor Analysis ✅
- One-factor-at-a-time (OFAT) across 7 axes:
  - Color, Taxonomy, Anatomy, Negation, Grammar, Phenology, Size
- Run on full test set (158 images)
- Evaluate at multiple IoUs: 0.3, 0.5, 0.5:0.95
- **Status:** Complete for both YOLO World and SAM3

### Phase 2: Combination Tests ✅
- Systematic prompt combinations
- Negation tests (text vs absorber architecture)
- Multi-class assembly
- Absorber classes (detect confusers, filter to target)
- **Status:** Complete (YOLO World: combinations + absorbers; SAM3: combinations)

### Phase 3: Confidence Threshold Analysis ✅
- Sweep confidence thresholds to understand model sensitivity
- Critical for zero-shot generalizability assessment
- **Status:** Complete for both YOLO World and SAM3

### Phase 4: Unlabeled Threshold Selection (In Progress)
- Develop methods for selecting confidence thresholds without labeled data
- Addresses limitation that Phase 1/2 used arbitrary threshold (0.1)
- Methods: Top-K, percentiles, detection volume matching, distribution analysis
- **Status:** Planning

### Phase 5: Grounding Gap Analysis (Future)
- Embedding space analysis
- Understand why models succeed/fail
- Motivate future model improvements

## Key Principles

1. **Zero-shot only** - No model weight changes
2. **Full test set** - No sampling, evaluate on all 158 images
3. **Multiple metrics** - mAP, F1, P, R, counting (R², RMSE, MAE, slope)
4. **Model-agnostic** - Same evaluation pipeline for all models
5. **Reproducible** - Configuration-driven, results saved in structured format

## Expected Outcomes

1. **Quantitative comparison** of YOLO World vs SAM3 across all metrics
2. **Model-specific prompt rules** - What works for each architecture
3. **Failure mode analysis** - Understanding where each model struggles
4. **Foundation for grounding gap work** - Structured analysis framework
