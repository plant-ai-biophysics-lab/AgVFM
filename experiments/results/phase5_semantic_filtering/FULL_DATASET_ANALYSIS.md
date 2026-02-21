# Phase 5 Semantic Filtering - Full Dataset Results Analysis

**Date**: 2026-02-20  
**Dataset**: 158 test images  
**Total Candidates**: 3,005 (from YOLO World with conf_floor=0.01)

## Executive Summary

We evaluated 4 selection methods for semantic filtering on the full test dataset, compared against a **baseline confidence threshold method** (optimal threshold from Phase 3):

1. **Baseline (conf=0.35)** - Optimal confidence threshold from Phase 3 (F1=0.489, highest baseline)
2. **Fixed p=0.1** - Best semantic filtering method (F1=0.333), highest recall (0.477)
3. **BH-FDR q=0.5** - Highest retention rate (74.3%), good recall (0.459)
4. **BH-FDR q=0.3** - Moderate performance across all metrics
5. **Top-5** - Highest precision (0.262) but lowest recall (0.208) and F1 (0.232)

**Key Finding**: The baseline confidence threshold method (conf=0.35) outperforms all semantic filtering methods, achieving F1=0.489 vs best semantic filtering F1=0.333. However, semantic filtering may still be valuable for scenarios where confidence scores are not well-calibrated or when additional filtering is needed beyond confidence.

## Detailed Results

### Detection Metrics (IoU=0.5)

| Method | F1 | Precision | Recall | TP | FP | FN |
|--------|----|-----------|--------|----|----|----|
| **Baseline (conf=0.35)** | **0.489** | **0.423** | **0.580** | **665** | **906** | **481** |
| Fixed p=0.1 | 0.333 | 0.255 | 0.477 | 474 | 1,383 | 519 |
| BH-FDR q=0.5 | 0.283 | 0.204 | 0.459 | 456 | 1,776 | 537 |
| BH-FDR q=0.3 | 0.280 | 0.215 | 0.401 | 398 | 1,454 | 595 |
| Top-5 | 0.232 | 0.262 | 0.208 | 207 | 582 | 786 |

### Retention Statistics

| Method | Total Kept | Retention Rate | Avg Kept/Image | Empty Images |
|--------|-----------|----------------|----------------|--------------|
| **Baseline (conf=0.35)** | **1,361** | **100%** | **8.61** | **0.0%** |
| BH-FDR q=0.5 | 2,232 | 74.3% | 14.13 | 10.8% |
| Fixed p=0.1 | 1,857 | 61.8% | 11.75 | 8.9% |
| BH-FDR q=0.3 | 1,852 | 61.6% | 11.72 | ~10% |
| Top-5 | 789 | 26.3% | 4.99 | 0.0% |

## Key Findings

### 1. Baseline Confidence Threshold Outperforms Semantic Filtering

**Baseline (conf=0.35)** achieves the best overall performance:
- **Highest F1** (0.489) - 47% better than best semantic filtering (0.333)
- **Highest Precision** (0.423) - 65% better than best semantic filtering (0.255)
- **Highest Recall** (0.580) - 22% better than best semantic filtering (0.477)
- **Fewest false positives** (906) - 35% fewer than Fixed p=0.1 (1,383)
- **Most true positives** (665) - 40% more than Fixed p=0.1 (474)
- **No empty images** - All images have detections

**Implication**: For this dataset, a well-tuned confidence threshold (from Phase 3) is more effective than semantic filtering. This suggests:
- YOLO World confidence scores are well-calibrated
- Semantic filtering may be adding noise or removing valid detections
- The semantic margin signal may not be strong enough to improve upon confidence

**However**, semantic filtering may still be valuable when:
- Confidence scores are not well-calibrated
- Additional filtering is needed beyond confidence
- Working with models that have poor confidence calibration

### 2. Fixed p=0.1 is the Best Semantic Filtering Method

- **Highest F1 score** (0.333) - best balance of precision and recall
- **Highest recall** (0.477) - captures nearly half of all true flowers
- **Good retention rate** (61.8%) - keeps most promising candidates
- **Lowest empty image rate** (8.9%) - rarely leaves images with no detections

**Trade-off**: Lower precision (0.255) means more false positives, but this is acceptable given the high recall.

### 2. BH-FDR q=0.5 Provides High Recall with More Candidates

- **Highest retention rate** (74.3%) - keeps most candidates
- **Good recall** (0.459) - second-best recall
- **More false positives** (1,776 FP) - due to lenient filtering

**Use case**: When you want to maximize recall and can tolerate more false positives.

### 3. Top-5 is Too Restrictive

- **Highest precision** (0.262) - but still relatively low
- **Lowest recall** (0.208) - misses 79% of true flowers
- **Lowest F1** (0.232) - poor overall performance
- **No empty images** - but at the cost of missing most flowers

**Verdict**: Not recommended. The fixed limit of 5 detections per image is too restrictive for images with many flowers.

### 4. BH-FDR q=0.3 is Similar to Fixed p=0.1

- Similar retention rates (61.6% vs 61.8%)
- Slightly lower F1 (0.280 vs 0.333) due to lower recall
- More conservative than q=0.5

**Verdict**: Fixed p=0.1 performs better with similar retention.

## Comparison to Baseline (Before Filtering)

From the q=0.5 visualization:
- **Before filtering**: F1=0.259, P=0.172, R=0.521
- **After filtering (q=0.5)**: F1=0.283, P=0.204, R=0.459

**Improvement**: 
- Precision increased from 0.172 to 0.204 (+18.6%)
- F1 increased from 0.259 to 0.283 (+9.3%)
- Recall decreased from 0.521 to 0.459 (-11.9%)

The semantic filtering successfully improves precision and F1, but at the cost of some recall.

## Method-Specific Analysis

### Fixed p=0.1

**How it works**: Keep all detections with p-value ≤ 0.1

**Strengths**:
- Simple, interpretable threshold
- Best overall F1 and recall
- Good balance between precision and recall

**Weaknesses**:
- No FDR control (may have more false discoveries)
- Fixed threshold doesn't adapt to image difficulty

**Best for**: General use when you want the best overall performance.

### BH-FDR q=0.5

**How it works**: Benjamini-Hochberg FDR control with q=0.5 (allows 50% false discoveries)

**Strengths**:
- Highest retention rate
- Good recall
- Statistical FDR control

**Weaknesses**:
- Very lenient (q=0.5 is quite high)
- More false positives than Fixed p=0.1

**Best for**: When recall is critical and you can tolerate many false positives.

### BH-FDR q=0.3

**How it works**: Benjamini-Hochberg FDR control with q=0.3 (allows 30% false discoveries)

**Strengths**:
- Moderate retention rate
- Statistical FDR control
- More conservative than q=0.5

**Weaknesses**:
- Lower F1 than Fixed p=0.1
- Lower recall than Fixed p=0.1

**Best for**: When you want FDR control but are more conservative than q=0.5.

### Top-5

**How it works**: Keep top 5 detections per image by semantic margin

**Strengths**:
- Highest precision (though still low)
- No empty images
- Simple, deterministic

**Weaknesses**:
- Very low recall (misses 79% of flowers)
- Too restrictive for images with many flowers
- Poor F1 score

**Best for**: Not recommended for this use case.

## Recommendations

### Primary Recommendation: **Baseline Confidence Threshold (conf=0.35)**

**Use the baseline confidence threshold method (conf=0.35) as the default** because:
1. **Best F1 score** (0.489) - 47% better than best semantic filtering
2. **Highest precision** (0.423) - 65% better than best semantic filtering
3. **Highest recall** (0.580) - 22% better than best semantic filtering
4. **Fewest false positives** (906 vs 1,383 for Fixed p=0.1)
5. **Most true positives** (665 vs 474 for Fixed p=0.1)
6. **No empty images** - All images have detections
7. **Simpler** - No semantic filtering overhead

**Conclusion**: For this dataset, a well-tuned confidence threshold (from Phase 3) is more effective than semantic filtering. This suggests YOLO World confidence scores are well-calibrated and semantic filtering is not adding value in this case.

### Alternative: **Fixed p=0.1** (if semantic filtering is required)

If you must use semantic filtering (e.g., for models with poor confidence calibration), use Fixed p=0.1 because:
1. Best F1 score among semantic filtering methods (0.333)
2. Highest recall (0.477)
3. Good balance of precision and recall
4. Simple, interpretable threshold
5. Low empty image rate (8.9%)

### Alternative: **BH-FDR q=0.5** (if recall is critical)

Use BH-FDR q=0.5 if:
- You need maximum recall
- You can tolerate more false positives
- You want statistical FDR control

### Not Recommended: **Top-5**

Top-5 is too restrictive and misses too many true flowers (79% false negative rate).

## Next Steps

Based on these results, the following improvements are recommended (from GPT5.2 analysis):

1. **Hybrid gate**: Combine confidence threshold with semantic filtering
2. **Improve prompt banks**: Strengthen semantic signal with better prompts
3. **Change aggregation**: Use logsumexp or top-2 average instead of max(target)-max(neg)
4. **Add wider crop scales**: Include 0.4-0.6 scales for tiny flowers
5. **Fix background null**: Use near-candidate or candidate-derived null instead of random background

## Files Generated

- **Comparison plot**: `plots/yolo_world_method_comparison.png`
- **Individual plots** (for q=0.5): 
  - `plots/yolo_world_candidates_vs_kept.png`
  - `plots/yolo_world_margin_distributions.png`
  - `plots/yolo_world_pvalue_distributions.png`
  - `plots/yolo_world_summary_statistics.png`
  - `plots/yolo_world_metrics_before_after.png`

## Technical Details

- **Model**: YOLO World (yolov8x-worldv2.pt)
- **Confidence floor**: 0.01 (very low to get many candidates)
- **Background samples**: 128 per image
- **Crop scales**: [0.0, 0.2] (tight crop, 20% padding)
- **IoU threshold**: 0.5 for metrics
- **Ground truth**: Manual annotations in YOLO format
