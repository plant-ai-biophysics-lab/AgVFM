# Phase 5: Semantic Filtering (Embedding Space Thresholding)

## Overview

Phase 5 implements the ChatGPT5.2-proposed pipeline for zero-shot threshold selection using semantic margin and per-image p-values with Benjamini-Hochberg FDR control.

## Pipeline Steps

1. **Candidate Generation**: Run detection at low confidence threshold (default: 0.01) to get high-recall candidate set
2. **Semantic Margin Computation**: For each candidate, compute margin = max(target similarity) - max(negative similarity)
   - Uses multi-scale crop encoding (tight + medium padding)
   - Target prompts: cowpea flower variants
   - Hard negatives: buds, calyx, leaves, stems, pods, soil, shadows, glare
3. **Background Sampling**: Sample K background regions (default: 128) per image, compute their margins
4. **P-Value Computation**: Empirical p-value = (1 + count(bg_margins >= candidate_margin)) / (1 + K)
5. **BH-FDR Selection**: Apply Benjamini-Hochberg FDR control (default: q=0.10) to select detections
6. **Post-Processing**: (Optional) Deduplication/NMS can be added

## Scripts

### `run_semantic_filtering.py`
Main experiment script implementing the full pipeline.

**Usage:**
```bash
# Test mode (5 images)
python run_semantic_filtering.py --test-mode

# Process subset of images
python run_semantic_filtering.py --num-images 20

# Process all images
python run_semantic_filtering.py

# Custom parameters
python run_semantic_filtering.py \
    --conf-floor 0.01 \
    --K-bg 128 \
    --q-fdr 0.10 \
    --output-dir experiments/results/phase5_semantic_filtering
```

**Parameters:**
- `--conf-floor`: Low confidence threshold for candidate generation (default: 0.01)
- `--K-bg`: Number of background samples per image (default: 128)
- `--q-fdr`: FDR level for Benjamini-Hochberg (default: 0.10)
- `--selection-method`: Selection method - `bh_fdr`, `fixed_p`, `top_k`, or `global_bh` (default: `bh_fdr`)
- `--p-cutoff`: P-value cutoff for `fixed_p` method (default: 0.05)
- `--top-k`: Number of top detections for `top_k` method (default: 5)
- `--top-k-p-max`: Optional max p-value for `top_k` method (e.g., 0.2)
- `--results-filename`: Custom JSON filename (default: `semantic_filtering_results.json`)
- `--results-subdir`: Subdirectory for JSON files (default: `results`)
- `--test-mode`: Process only 5 images for quick testing
- `--num-images`: Number of images to process (None = all)

**Output:**
- JSON file with filtered detections and metadata (saved to `results/` subdirectory)
- Summary statistics printed to console

### `run_ablations.py`
Run all selection method ablations automatically.

**Usage:**
```bash
# Test mode (5 images)
python run_ablations.py --test-mode

# Full dataset
python run_ablations.py

# Background mode (allows SSH disconnect)
python run_ablations.py --background

# Skip certain ablations
python run_ablations.py --skip-top-k --skip-global-bh
```

**Output:**
- Multiple JSON files in `results/` subdirectory, one per ablation
- All results use same base directory with different filenames

### `visualize_results.py`
Generate visualizations from results JSON files.

**Usage:**
```bash
python visualize_results.py \
    --results-file experiments/results/phase5_semantic_filtering/results/semantic_filtering_results.json
```

## Current Status

✅ **YOLO World**: Fully implemented and tested
❌ **SAM3**: Pending dimension mismatch fix (1024-d vision vs 512-d text embeddings)

## Known Issues

1. **SAM3 Dimension Mismatch**: SAM3 vision embeddings are 1024-d but text embeddings are 512-d. Need to find projection layer or implement alternative similarity metric.
2. **Conservative Filtering**: FDR control (q=0.10) is very conservative, may result in low recall. Consider tuning q or using alternative selection criteria.

## Timing Estimates

From test runs:
- **Per image**: ~9-20 seconds (depends on number of candidates)
- **Full dataset (158 images)**: ~25-50 minutes (estimated)
- **Bottleneck**: Background sampling (K=128 per image) is the main compute cost

## Next Steps

1. Fix SAM3 dimension mismatch (find projection or alternative)
2. Evaluate results on full dataset
3. Compare to Phase 3 optimal thresholds
4. Tune FDR level (q) for better recall/precision tradeoff
5. Add post-processing (dedup/NMS) if needed
