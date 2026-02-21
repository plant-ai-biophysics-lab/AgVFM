# Phase 5 Ablation Results Summary

**Date:** February 20, 2025  
**Test Mode:** 5 images (quick validation)  
**Status:** ✅ Plots Generated

---

## Test Results Summary

### Retention Rates (5 Images, 89 Total Candidates)

| Method | Kept/Candidates | Retention Rate | Empty Images |
|--------|----------------|----------------|--------------|
| **baseline (q=0.1)** | 108/3005* | 0.036 | 87.3%* |
| **Global BH q=0.1** | 0/89 | 0.000 | 100.0% |
| **Global BH q=0.2** | 0/89 | 0.000 | 100.0% |
| **Global BH q=0.3** | 0/89 | 0.000 | 100.0% |
| **Fixed p=0.05** | 46/89 | 0.517 | 0.0% |
| **Fixed p=0.1** | 59/89 | 0.663 | 0.0% |
| **BH-FDR q=0.1** | 3/89 | 0.034 | 80.0% |
| **BH-FDR q=0.2** | 64/89 | 0.719 | 0.0% |
| **BH-FDR q=0.3** | 70/89 | 0.787 | 0.0% |
| **BH-FDR q=0.5** | 73/89 | 0.820 | 0.0% |
| **Top-10** | 50/89 | 0.562 | 0.0% |
| **Top-10 (p≤0.2)** | 46/89 | 0.517 | 0.0% |
| **Top-3** | 15/89 | 0.169 | 0.0% |
| **Top-3 (p≤0.2)** | 15/89 | 0.169 | 0.0% |
| **Top-5** | 25/89 | 0.281 | 0.0% |
| **Top-5 (p≤0.2)** | 25/89 | 0.281 | 0.0% |

*Note: Baseline results are from full dataset (158 images), others are from 5-image test*

### Key Observations

1. **Global BH Methods**: All show 0% retention in test mode. This is likely because with only 5 images and 89 total candidates, the global BH threshold is still extremely strict. May perform better on full dataset.

2. **Fixed P-Value Methods**: Show good retention (52-66%) with 0% empty images. These are promising for practical use.

3. **Higher Q Values**: BH-FDR with q=0.2-0.5 shows excellent retention (72-82%) with 0% empty images. This confirms GPT5.2's recommendation to raise q.

4. **Top-K Methods**: Show moderate retention (17-56%) depending on k. The p≤0.2 constraint doesn't seem to help much in test mode.

5. **Best Performers (Test Mode)**:
   - **BH-FDR q=0.5**: 82% retention, 0% empty
   - **BH-FDR q=0.3**: 79% retention, 0% empty
   - **Fixed p=0.1**: 66% retention, 0% empty

---

## Generated Plots

### Comparison Plot
- **File**: `plots/yolo_world_method_comparison.png`
- **Contents**: 
  - Metrics comparison (F1, Precision, Recall) - if computed
  - Retention rate comparison
  - Empty image percentage
  - Summary table

### Individual Plots (per result file)
- Candidates vs Kept
- Margin distributions
- P-value distributions
- Summary statistics
- Metrics before/after (if computed)

---

## Next Steps: Full Dataset Run

### Option 1: Run All Ablations Sequentially
```bash
# Run all ablations in background
python experiments/scripts/experiments/phase5/run_ablations.py --background

# Monitor progress
tail -f experiments/results/phase5_semantic_filtering/ablations.log
```

**Pros:**
- Complete comparison across all methods
- Can identify best method definitively

**Cons:**
- Long runtime (~2-3 hours for all ablations)
- May want to prioritize based on test results

### Option 2: Run Selected Methods Only
Based on test results, prioritize:
- **BH-FDR q=0.3** (best retention in test)
- **BH-FDR q=0.5** (highest retention)
- **Fixed p=0.1** (good balance)
- **Top-5** (moderate retention, simple)

```bash
# Run selected methods
python experiments/scripts/experiments/phase5/run_semantic_filtering.py \
    --selection-method bh_fdr --q-fdr 0.3 \
    --results-filename semantic_filtering_results_q0.3_full.json

python experiments/scripts/experiments/phase5/run_semantic_filtering.py \
    --selection-method fixed_p --p-cutoff 0.1 \
    --results-filename semantic_filtering_results_p0.1_full.json
```

**Pros:**
- Faster (focus on promising methods)
- Can run multiple in parallel if desired

**Cons:**
- May miss optimal method

### Option 3: Staged Approach
1. **Stage 1**: Run top 3-4 methods on full dataset
2. **Stage 2**: Compute metrics and compare
3. **Stage 3**: Run additional methods if needed

**Pros:**
- Balanced approach
- Can iterate based on results

**Cons:**
- Requires multiple iterations

---

## Recommendations

Based on test results, I recommend **Option 2** with these methods:
1. **BH-FDR q=0.3** - Best balance in test (79% retention, 0% empty)
2. **BH-FDR q=0.5** - Highest retention (82% retention, 0% empty)
3. **Fixed p=0.1** - Simple and effective (66% retention, 0% empty)
4. **Top-5** - Simple fallback (28% retention, 0% empty)

These can be run in parallel or sequentially in background mode.

---

## Files Generated

- `plots/yolo_world_method_comparison.png` - Comparison across all methods
- Individual plots for each result file (if generated)
- All result JSON files in `results/` subdirectory

---

**Generated**: February 20, 2025
