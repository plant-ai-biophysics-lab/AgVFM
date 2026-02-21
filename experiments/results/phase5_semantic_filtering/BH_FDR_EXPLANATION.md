# Understanding BH-FDR q=0.5

## What is BH-FDR?

**Benjamini-Hochberg False Discovery Rate (BH-FDR)** is a statistical method for controlling the expected proportion of false positives when testing multiple hypotheses simultaneously. In our case, each detection candidate is a "hypothesis" (is this a real flower?).

## The Problem It Solves

When you have many detections and filter them using p-values, you need to account for **multiple testing**. If you use a fixed p-value threshold (like p ≤ 0.05) for 100 detections, you'd expect ~5 false positives by chance alone. BH-FDR adjusts the threshold based on how many tests you're running.

## What Does q=0.5 Mean?

**q = 0.5** means: "I'm willing to accept that up to 50% of the detections I keep might be false positives."

This is quite **lenient** compared to:
- **q = 0.1** (10% false discovery rate) - more conservative, keeps fewer detections
- **q = 0.05** (5% false discovery rate) - very conservative, keeps even fewer

## How BH-FDR Works (Step-by-Step)

### Step 1: Compute P-Values
For each candidate detection, we compute a p-value:
- **P-value** = probability that a random background region would have a semantic margin ≥ this candidate's margin
- Lower p-value = more likely to be a real flower (less likely to occur by chance)

### Step 2: Sort P-Values
Sort all p-values from smallest to largest:
```
p₁ ≤ p₂ ≤ p₃ ≤ ... ≤ pₘ
```

### Step 3: Find the Largest k
Find the largest k such that:
```
pₖ ≤ (k / M) × q
```

Where:
- **k** = rank (1, 2, 3, ..., M)
- **M** = total number of candidates
- **q** = FDR level (0.5 in our case)

### Step 4: Keep All Detections Up to k
Keep all detections with p-values ≤ pₖ

## Example: BH-FDR q=0.5

Let's say you have **10 candidates** with these p-values:

| Rank (k) | P-value | Threshold (k/10 × 0.5) | Keep? |
|----------|---------|------------------------|-------|
| 1        | 0.01    | 0.05                   | ✅ Yes (0.01 ≤ 0.05) |
| 2        | 0.08    | 0.10                   | ✅ Yes (0.08 ≤ 0.10) |
| 3        | 0.12    | 0.15                   | ✅ Yes (0.12 ≤ 0.15) |
| 4        | 0.20    | 0.20                   | ✅ Yes (0.20 ≤ 0.20) |
| 5        | 0.25    | 0.25                   | ✅ Yes (0.25 ≤ 0.25) |
| 6        | 0.30    | 0.30                   | ✅ Yes (0.30 ≤ 0.30) |
| 7        | 0.35    | 0.35                   | ✅ Yes (0.35 ≤ 0.35) |
| 8        | 0.40    | 0.40                   | ✅ Yes (0.40 ≤ 0.40) |
| 9        | 0.45    | 0.45                   | ✅ Yes (0.45 ≤ 0.45) |
| 10       | 0.50    | 0.50                   | ✅ Yes (0.50 ≤ 0.50) |

**Result:** Keep all 10 detections!

Now compare to **q=0.1** (more conservative):

| Rank (k) | P-value | Threshold (k/10 × 0.1) | Keep? |
|----------|---------|------------------------|-------|
| 1        | 0.01    | 0.01                   | ✅ Yes (0.01 ≤ 0.01) |
| 2        | 0.08    | 0.02                   | ❌ No (0.08 > 0.02) |
| 3        | 0.12    | 0.03                   | ❌ No (0.12 > 0.03) |
| ...      | ...     | ...                    | ...   |

**Result:** Keep only 1 detection!

## Why q=0.5 is More Lenient

The threshold formula is: `threshold = (k / M) × q`

- **q=0.1**: threshold grows slowly (0.01, 0.02, 0.03, ...)
- **q=0.5**: threshold grows faster (0.05, 0.10, 0.15, ...)

With q=0.5, the threshold is **5x larger** at each rank, so more detections pass the test.

## Applied Per-Image in Phase 5

In our implementation, BH-FDR is applied **per-image**:

1. For each image:
   - Generate candidate detections (low confidence threshold: 0.01)
   - Compute semantic margins for each candidate
   - Sample 128 background regions
   - Compute p-values for each candidate
   - Apply BH-FDR q=0.5 to select which candidates to keep

2. The procedure runs independently for each image (not globally across all images)

## Results from Test Mode

From `ABLATION_RESULTS.md`:
- **BH-FDR q=0.1**: 3/89 kept (3.4% retention) - very conservative
- **BH-FDR q=0.2**: 64/89 kept (71.9% retention)
- **BH-FDR q=0.3**: 70/89 kept (78.7% retention)
- **BH-FDR q=0.5**: 73/89 kept (82.0% retention) - most lenient

## Trade-offs

**q=0.5 (lenient):**
- ✅ Keeps more detections (higher recall)
- ✅ Fewer empty images
- ❌ More false positives (up to 50% of kept detections could be wrong)

**q=0.1 (conservative):**
- ✅ Fewer false positives (only 10% expected)
- ❌ Keeps fewer detections (lower recall)
- ❌ More empty images

## Code Implementation

```python
def benjamini_hochberg_fdr(p_values: List[float], q: float = 0.10) -> List[int]:
    M = len(p_values)
    
    # Sort p-values with indices
    indexed_pvals = [(i, p) for i, p in enumerate(p_values)]
    indexed_pvals.sort(key=lambda x: x[1])
    
    # Find largest k such that p_(k) <= (k/M) * q
    keep_indices = []
    for k in range(1, M + 1):
        p_k = indexed_pvals[k - 1][1]
        threshold = (k / M) * q
        
        if p_k <= threshold:
            keep_indices.append(indexed_pvals[k - 1][0])
        else:
            break
    
    return sorted(keep_indices)
```

## Summary

**BH-FDR q=0.5** means:
- "I'll keep detections where the expected false discovery rate is ≤ 50%"
- More lenient than q=0.1, q=0.2, q=0.3
- Applied per-image to filter candidate detections
- Balances recall (keeping more detections) vs precision (fewer false positives)

In practice, q=0.5 gives you **82% retention** in test mode with **0% empty images**, making it a good choice for agricultural applications where missing flowers is costly.
