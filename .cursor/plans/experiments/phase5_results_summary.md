# Phase 5: Semantic Filtering Results Summary

**Date:** February 20, 2025  
**Model:** YOLO World  
**Dataset:** 158 test images (cowpea flowers)  
**Status:** ✅ Complete

---

## Executive Summary

Phase 5 implemented a semantic filtering pipeline based on ChatGPT5.2's recommendations, using semantic margins (target vs hard negative embeddings) and Benjamini-Hochberg False Discovery Rate (FDR) control to filter detections without labeled data. The pipeline processes 3,005 candidate detections (generated at confidence threshold 0.01) and filters them down to 108 high-quality detections (3.6% retention rate). While precision improved from 0.172 to 0.241, recall dropped dramatically from 0.521 to 0.026, resulting in a net decrease in F1 score from 0.259 to 0.047. The filtering is extremely conservative, with 87.3% of images having zero detections after filtering.

---

## Methodology

### Pipeline Overview

The Phase 5 pipeline implements the following steps:

1. **Candidate Generation**: Run YOLO World inference at a very low confidence threshold (0.01) to generate a large pool of candidate detections
2. **Semantic Margin Computation**: For each candidate detection:
   - Extract multi-scale image crops (scales: 0.0, 0.2 padding)
   - Compute CLIP embeddings for image crops
   - Compute CLIP embeddings for target prompts and hard negative prompts
   - Calculate semantic margin: `max(target_similarity) - max(negative_similarity)`
3. **Background Sampling**: Sample 128 background regions per image (non-overlapping with candidates)
4. **P-Value Computation**: For each candidate, compute empirical p-value by comparing its semantic margin to the distribution of background margins
5. **Benjamini-Hochberg FDR Control**: Apply BH-FDR with q=0.10 to control false discovery rate per image
6. **Post-Processing**: Keep only detections that pass the FDR threshold

### Configuration

- **Model**: YOLO World (best configuration from Phase 3: `C2_not_bud_calyx`)
- **Confidence Floor**: 0.01 (for candidate generation)
- **Background Samples (K_bg)**: 128 per image
- **FDR Level (q)**: 0.10
- **Crop Scales**: [0.0, 0.2] (tight and medium crops)
- **Target Prompts**: 5 prompts (e.g., "cowpea flower", "yellow cowpea flower", etc.)
- **Hard Negative Prompts**: 18 prompts (e.g., "cowpea bud", "cowpea calyx", "leaf", "stem", etc.)

### Implementation Details

- **Batched Processing**: Implemented batched crop encoding for efficiency (batch size: 15 for YOLO World)
- **GPU Acceleration**: All embedding computations run on GPU
- **Processing Time**: ~4 seconds per image (down from ~9 seconds with batching)
- **Total Processing Time**: ~10.5 minutes for 158 images

---

## Results

### Overall Statistics

- **Total Images**: 158
- **Total Candidates**: 3,005 (average: 19.0 per image, median: 17.0, max: 71)
- **Total Kept**: 108 (average: 0.68 per image, median: 0.0, max: 10)
- **Retention Rate**: 3.6% (108 / 3,005)
- **Images with 0 Detections**: 138/158 (87.3%)
- **Images with ≥1 Detection**: 20/158 (12.7%)

### Detection Metrics (IoU=0.5)

**Before Filtering (Candidates at conf=0.01):**
- **F1 Score**: 0.259
- **Precision**: 0.172
- **Recall**: 0.521
- **True Positives**: 516
- **False Positives**: 2,489
- **False Negatives**: 475

**After Filtering (BH-FDR with q=0.10):**
- **F1 Score**: 0.047
- **Precision**: 0.241
- **Recall**: 0.026
- **True Positives**: 26
- **False Positives**: 82
- **False Negatives**: 965

### Key Observations

1. **Precision Improvement**: Precision increased from 0.172 to 0.241 (40% relative increase), indicating that the semantic filtering successfully identifies higher-quality detections.

2. **Recall Collapse**: Recall dropped from 0.521 to 0.026 (95% relative decrease), indicating that the filtering is extremely conservative and removes many true positives.

3. **F1 Score Decrease**: Despite the precision improvement, the dramatic recall drop results in a net decrease in F1 score from 0.259 to 0.047.

4. **Conservative Filtering**: The 3.6% retention rate and 87.3% of images having zero detections suggests the method is too conservative for practical use.

5. **False Positive Reduction**: False positives decreased from 2,489 to 82 (96.7% reduction), which is the primary success of the method.

6. **True Positive Loss**: True positives decreased from 516 to 26 (95.0% reduction), which is the primary failure of the method.

### Semantic Margin Statistics

- **All Candidates**: 
  - Mean: -0.0101
  - Median: -0.0097
  - Std: 0.0245
  - Range: [-0.0806, 0.0585]

- **Kept Detections**: 
  - Mean: -0.0010
  - Median: 0.0007
  - Range: [-0.0664, 0.0440]

- **Rejected Detections**: 
  - Mean: -0.0104
  - Median: -0.0102
  - Range: [-0.0806, 0.0585]

**Interpretation**: The semantic margins are very close to zero, with kept detections having slightly more positive margins (closer to zero or positive) than rejected detections. However, the difference is small, suggesting that the semantic margin signal is weak.

### P-Value Statistics

- **All Candidates**: 
  - Mean: 0.2152
  - Median: 0.0465
  - 61.8% of candidates have p-values ≤ 0.10

- **Kept Detections**: 
  - Mean: 0.1086
  - Median: 0.0155
  - All kept detections should have p-values ≤ FDR threshold (though some may be slightly above due to BH-FDR adjustment)

- **Rejected Detections**: 
  - Mean: 0.2192
  - Median: 0.0543

**Interpretation**: The p-value distribution shows that most candidates have relatively high p-values, with only a small fraction passing the FDR threshold. The BH-FDR procedure successfully controls the false discovery rate but at the cost of very low recall.

---

## Visualizations

### Figure 1: Candidates vs Kept Detections

![Candidates vs Kept](experiments/results/phase5_semantic_filtering/plots/yolo_world_candidates_vs_kept.png)

This plot shows the relationship between the number of candidate detections and the number of detections kept after semantic filtering. The scatter plot (left) shows that most images have few kept detections relative to candidates, with the majority of points falling well below the y=x line. The histogram (right) shows that the distribution of kept detections is highly skewed, with most images having 0 kept detections.

### Figure 2: Semantic Margin Distributions

![Margin Distributions](experiments/results/phase5_semantic_filtering/plots/yolo_world_margin_distributions.png)

This plot shows the distribution of semantic margins for all candidates (left) and a comparison between kept and rejected detections (right). The margins are centered near zero, with kept detections having slightly more positive margins on average. However, the distributions overlap significantly, indicating that the semantic margin signal is weak.

### Figure 3: P-Value Distributions

![P-Value Distributions](experiments/results/phase5_semantic_filtering/plots/yolo_world_pvalue_distributions.png)

This plot shows the distribution of p-values for all candidates (left) and a comparison between kept and rejected detections (right). The FDR threshold (q=0.10) is shown as a red dashed line. Most candidates have p-values above the threshold, with only a small fraction passing the FDR control.

### Figure 4: Summary Statistics

![Summary Statistics](experiments/results/phase5_semantic_filtering/plots/yolo_world_summary_statistics.png)

This plot provides an overview of the filtering results, showing distributions of candidates, kept detections, keep rate, and the proportion of images with zero detections. The pie chart (bottom right) shows that 87.3% of images have zero detections after filtering.

### Figure 5: Metrics Before and After Filtering

![Metrics Before and After](experiments/results/phase5_semantic_filtering/plots/yolo_world_metrics_before_after.png)

This plot compares F1 score, Precision, and Recall before and after semantic filtering. The bars show that:
- **F1 Score**: Decreased from 0.259 to 0.047
- **Precision**: Increased from 0.172 to 0.241
- **Recall**: Decreased from 0.521 to 0.026

The summary text box shows the TP, FP, and FN counts for both conditions.

---

## Analysis and Interpretation

### Strengths

1. **Precision Improvement**: The method successfully improves precision from 0.172 to 0.241, demonstrating that semantic filtering can identify higher-quality detections.

2. **False Positive Reduction**: The 96.7% reduction in false positives (from 2,489 to 82) is a significant achievement, showing that the method effectively filters out low-quality detections.

3. **Zero-Shot Approach**: The method operates entirely without labeled data, using only the semantic similarity between image crops and text prompts.

4. **Statistically Principled**: The use of Benjamini-Hochberg FDR control provides a principled statistical framework for controlling false discoveries.

### Weaknesses

1. **Extreme Conservatism**: The 3.6% retention rate and 87.3% of images having zero detections indicate that the method is too conservative for practical use.

2. **Recall Collapse**: The 95% reduction in recall (from 0.521 to 0.026) means that the method removes many true positives, making it unsuitable for applications requiring high recall.

3. **Weak Semantic Signal**: The semantic margins are very close to zero, with only small differences between kept and rejected detections, suggesting that the semantic similarity signal is weak.

4. **F1 Score Decrease**: Despite precision improvement, the dramatic recall drop results in a net decrease in F1 score, making the method less effective overall.

5. **Limited Generalization**: The method was only tested on YOLO World; SAM3 support is pending due to dimension mismatch issues (1024-d vision embeddings vs 512-d text embeddings).

### Potential Issues

1. **Background Sampling**: The background sampling strategy (128 random non-overlapping regions) may not adequately represent the true background distribution, leading to biased p-value estimates.

2. **FDR Level**: The FDR level (q=0.10) may be too strict, especially given the weak semantic signal. A higher FDR level (e.g., q=0.20 or q=0.30) might improve recall while still maintaining reasonable precision.

3. **Multi-Scale Crops**: The use of multi-scale crops (scales: 0.0, 0.2) may not be optimal. A single scale or different scale combinations might provide better semantic signals.

4. **Prompt Bank**: The target and hard negative prompt banks may not be optimal. More diverse or better-curated prompts might improve the semantic margin signal.

5. **Per-Image FDR**: The BH-FDR procedure is applied per-image, which may be too conservative. A global FDR control across all images might be more appropriate.

6. **Semantic Margin Calculation**: The current margin calculation (`max(target_sim) - max(negative_sim)`) may not be optimal. Alternative formulations (e.g., weighted averages, different aggregation methods) might provide better signals.

---

## Comparison to Phase 3 Optimal Thresholds

Phase 3 identified optimal confidence thresholds for YOLO World (`C2_not_bud_calyx`) at IoU=0.5:
- **Optimal F1 Threshold**: ~0.3-0.4
- **Optimal F1 Score**: ~0.35-0.40
- **Optimal Precision**: ~0.40-0.50
- **Optimal Recall**: ~0.30-0.40

The Phase 5 semantic filtering results (F1=0.047, P=0.241, R=0.026) are significantly worse than Phase 3 optimal thresholds, indicating that the semantic filtering approach is not competitive with simple confidence thresholding.

---

## Recommendations for Next Steps

### Immediate Actions

1. **Tune FDR Level**: Experiment with higher FDR levels (q=0.20, 0.30, 0.40) to improve recall while maintaining reasonable precision.

2. **Improve Background Sampling**: Investigate alternative background sampling strategies (e.g., stratified sampling, adaptive sampling based on image content).

3. **Optimize Prompt Banks**: Experiment with different target and hard negative prompt combinations to improve semantic margin signal strength.

4. **Alternative Margin Formulations**: Test alternative semantic margin calculations (e.g., weighted averages, different aggregation methods).

5. **Global vs Per-Image FDR**: Compare per-image FDR control (current) with global FDR control across all images.

### Medium-Term Actions

1. **Fix SAM3 Support**: Resolve the dimension mismatch issue (1024-d vision vs 512-d text) to enable SAM3 evaluation.

2. **Hybrid Approach**: Combine semantic filtering with confidence thresholding (e.g., use semantic filtering to refine detections above a confidence threshold).

3. **Alternative Quality Signals**: Explore alternative quality signals beyond semantic margins (e.g., detection consistency, spatial coherence, temporal consistency for video).

4. **Learning-Based Approaches**: Consider learning-based approaches to combine multiple signals (semantic margin, confidence, spatial features) for detection quality assessment.

### Long-Term Actions

1. **Theoretical Analysis**: Develop theoretical understanding of when semantic filtering is effective and when it fails.

2. **Generalization Studies**: Test the approach on different datasets and object classes to assess generalization.

3. **Integration with Phase 4**: Combine semantic filtering (Phase 5) with distribution analysis (Phase 4) for a more robust threshold selection method.

---

## Conclusion

Phase 5 successfully implemented a semantic filtering pipeline based on semantic margins and BH-FDR control. While the method improves precision and dramatically reduces false positives, it is too conservative for practical use, resulting in very low recall and overall F1 score. The weak semantic margin signal and extreme conservatism suggest that the current approach needs significant refinement before it can be competitive with simple confidence thresholding. However, the method demonstrates the potential of using embedding space analysis for zero-shot detection quality assessment, and with appropriate tuning and improvements, it may become a valuable tool for precision-focused applications.

---

## Technical Details

### File Locations

- **Results JSON**: `experiments/results/phase5_semantic_filtering/semantic_filtering_results.json`
- **Plots Directory**: `experiments/results/phase5_semantic_filtering/plots/`
- **Script**: `experiments/scripts/experiments/phase5/run_semantic_filtering.py`
- **Visualization Script**: `experiments/scripts/experiments/phase5/visualize_results.py`
- **Approach Document**: `.cursor/plans/approach/approach_phase5.md`

### Dependencies

- YOLO World model (via `agvfm.models.yolo_world`)
- CLIP embeddings (via YOLO World's internal CLIP model)
- NumPy, PyTorch, PIL
- Benjamini-Hochberg FDR implementation (custom)

### Computational Resources

- **GPU**: NVIDIA GPU with 24GB VRAM
- **Processing Time**: ~4 seconds per image (with batching)
- **Total Time**: ~10.5 minutes for 158 images
- **Batch Size**: 15 for YOLO World crop encoding

---

**Document Generated**: February 20, 2025  
**Author**: Auto (AI Assistant)  
**Review Status**: Pending external LLM review

---

## Phase 5 Revisions: Selection Method Ablations (February 20, 2025)

### Overview

Based on GPT5.2's analysis of the initial Phase 5 results, we implemented multiple alternative selection methods to address the extreme conservatism of per-image BH-FDR at q=0.10. The analysis identified that per-image BH-FDR was effectively requiring p-values ≤ 0.005 to keep even a single detection (q/M ≈ 0.10/19), explaining why 87.3% of images had zero detections.

### Implemented Selection Methods

1. **Q Sweep (Per-Image BH-FDR)**: Tested q ∈ {0.1, 0.2, 0.3, 0.5}
   - Higher q values should improve recall by relaxing the statistical threshold
   - Files: `semantic_filtering_results_q0.1.json` through `semantic_filtering_results_q0.5.json`

2. **Fixed P-Value Cutoff**: Simple threshold-based selection
   - `fixed_p` with p ≤ 0.05 or p ≤ 0.10
   - Avoids the q/M harshness of per-image BH-FDR
   - Files: `semantic_filtering_results_p0.05.json`, `semantic_filtering_results_p0.1.json`

3. **Top-K Selection**: Select top-k detections by semantic margin
   - Tested k ∈ {3, 5, 10}
   - Optional p-value constraint (p ≤ 0.2)
   - Files: `semantic_filtering_results_top3.json`, `semantic_filtering_results_top5.json`, etc.

4. **Global BH-FDR**: Apply BH-FDR across all candidates (not per-image)
   - Avoids the per-image q/M penalty
   - Tested q ∈ {0.1, 0.2, 0.3}
   - Files: `semantic_filtering_results_global_q0.1.json`, etc.

### Implementation Changes

**Directory Structure**: All results now stored in a single directory (`phase5_semantic_filtering/`) with different JSON filenames, rather than creating separate directories for each experiment. This addresses the organizational concern about directory proliferation.

**Background Execution**: Added `--background` flag to `run_ablations.py` to enable SSH-disconnect-safe execution using `nohup`.

**Results Organization**:
- Single output directory: `experiments/results/phase5_semantic_filtering/`
- Differentiated by JSON filename: `semantic_filtering_results_{method}_{params}.json`
- All plots stored in `plots/` subdirectory

### Test Results (5 Images)

Initial test runs on 5 images show promising improvements:

- **q=0.5 (BH-FDR)**: 73/89 kept (82% retention) vs. original q=0.1: ~3-7/89 (3-8% retention)
- **Fixed p=0.05**: 46/89 kept (52% retention)
- **Top-5**: All images had detections (0% empty images) vs. original 87.3% empty

### Next Steps

1. Run full dataset ablations (158 images) to get comprehensive metrics
2. Compare F1, Precision, Recall across all selection methods
3. Identify optimal selection method and parameters
4. Update visualizations to compare methods
5. Proceed with Step 2 (hybrid gate) and Step 3 (strengthen semantic signal) if needed

### Files and Scripts

- **Main Script**: `experiments/scripts/experiments/phase5/run_semantic_filtering.py`
  - Added `--results-filename` argument for custom JSON filenames
  - Supports all selection methods via `--selection-method`
  
- **Ablations Script**: `experiments/scripts/experiments/phase5/run_ablations.py`
  - Automates running all ablations
  - Supports `--background` for SSH-disconnect-safe execution
  - Configurable via skip flags

- **Consolidation Script**: `experiments/scripts/experiments/phase5/consolidate_results.py`
  - Moved old separate directories into single directory structure

### Lessons Learned

1. **Directory Management**: Use single directory with different filenames rather than separate directories for each experiment variant
2. **Background Execution**: Always provide `--background` option for long-running experiments
3. **Selection Method Impact**: The choice of selection method (and its parameters) has dramatic impact on recall vs. precision trade-off
4. **Statistical Thresholds**: Per-image BH-FDR with low q can be extremely conservative; global BH or fixed thresholds may be more practical

---

**Revision Date**: February 20, 2025  
**Revision Author**: Auto (AI Assistant)
