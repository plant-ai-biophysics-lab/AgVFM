# Exact P-Value Calculation

## Overview

The p-value is computed using an **empirical (non-parametric) approach** that compares a candidate detection's semantic margin against a distribution of background region margins.

## Step-by-Step Process

### Step 1: Compute Semantic Margin for Candidate

For each candidate detection box, compute the **semantic margin**:

```python
semantic_margin = max(target_similarities) - max(negative_similarities)
```

Where:
- **target_similarities**: Cosine similarities between the candidate crop embedding and all target text embeddings (e.g., "cowpea flower", "yellow cowpea flower", etc.)
- **negative_similarities**: Cosine similarities between the candidate crop embedding and all negative text embeddings (e.g., "cowpea bud", "leaf", "stem", etc.)

**Intuition**: High margin = candidate is more similar to flowers than to non-flowers.

### Step 2: Sample Background Regions

Sample **K = 128** random background regions from the same image:

```python
bg_regions = sample_background_regions(image, candidate_boxes, K=128)
```

**Background sampling criteria:**
- Random boxes of size 20-200 pixels (or up to 1/4 of image size)
- Must have IoU < 0.1 with any candidate detection (non-overlapping)
- At least 20×20 pixels minimum size

**Purpose**: These represent "null" regions that are NOT flowers.

### Step 3: Compute Semantic Margins for Background Regions

For each of the 128 background regions, compute the same semantic margin:

```python
bg_margins = compute_multi_scale_margins_batch(
    model, image, bg_regions, 
    target_text_embeddings, negative_text_embeddings
)
```

This gives us a **distribution of margins for background (non-flower) regions**.

### Step 4: Calculate P-Value

For each candidate detection with margin `m`, compute:

```python
p_value = (1 + count(bg_margins >= m)) / (1 + K)
```

Where:
- **K** = number of background samples (128)
- **count(bg_margins >= m)** = how many background regions have margin ≥ candidate margin

## Exact Code Implementation

```python
def compute_p_value(margin: float, bg_margins: List[float]) -> float:
    """
    Compute empirical p-value: (1 + count(bg_margins >= margin)) / (1 + K)
    
    Args:
        margin: Candidate margin
        bg_margins: Sorted list of background margins
        
    Returns:
        P-value (float)
    """
    K = len(bg_margins)  # Typically 128
    if K == 0:
        return 1.0
    
    # Count how many background margins are >= candidate margin
    count_ge = sum(1 for bg_m in bg_margins if bg_m >= margin)
    
    # Smoothed empirical p-value
    p_value = (1 + count_ge) / (1 + K)
    
    return p_value
```

## Why This Formula?

### The "+1" Smoothing

The formula uses `(1 + count) / (1 + K)` instead of `count / K`:

1. **Prevents p=0**: If no background margins ≥ candidate, p = 1/(1+K) instead of 0
2. **Prevents p=1**: If all background margins ≥ candidate, p = (1+K)/(1+K) = 1
3. **Standard empirical p-value formula**: Accounts for the observed value itself

### Interpretation

**P-value = probability that a random background region would have a semantic margin ≥ this candidate's margin**

- **Low p-value (e.g., 0.01)**: Very few background regions have such high margins → candidate is likely a real flower
- **High p-value (e.g., 0.8)**: Many background regions have similar or higher margins → candidate is likely NOT a flower

## Example Calculation

Let's say:
- **Candidate margin**: 0.15
- **Background margins** (128 samples): [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, ...] (sorted descending)

Count how many bg_margins ≥ 0.15:
- bg_margins ≥ 0.15: [0.15, 0.18, 0.20, 0.22, 0.25] = 5 values

P-value = (1 + 5) / (1 + 128) = 6 / 129 = **0.0465**

This means: "4.65% of background regions have margins ≥ 0.15, so this candidate is relatively unlikely to be background."

## Multi-Scale Margin

In practice, we compute margins at **multiple scales** (crop sizes):

```python
scales = [0.0, 0.2]  # Tight crop, 20% padding
```

For each candidate:
1. Extract crop at scale 0.0 (tight)
2. Extract crop at scale 0.2 (20% padding)
3. Compute margin for each scale
4. Take the **maximum margin** across scales

This handles cases where:
- Tight crop misses context → wider crop helps
- Wider crop includes too much background → tight crop helps

## Background Sampling Details

```python
def sample_background_regions(image, candidate_boxes, K=128, min_iou=0.1):
    """
    Sample K random boxes that don't overlap with candidates.
    
    - Random size: 20-200 pixels (or 1/4 image size)
    - Random position: anywhere in image
    - Reject if IoU > 0.1 with any candidate
    """
```

**Why non-overlapping?**
- We want "pure background" regions
- Overlapping regions might contain partial flowers
- IoU < 0.1 ensures clean background samples

## Complete Pipeline

For each image:

1. **Generate candidates**: Low confidence threshold (0.01) → many candidate boxes
2. **Compute candidate margins**: For each candidate, compute max(target_sim) - max(neg_sim) across scales
3. **Sample background**: 128 random non-overlapping regions
4. **Compute background margins**: Same margin calculation for all 128 regions
5. **Calculate p-values**: For each candidate, p = (1 + count(bg ≥ candidate)) / (1 + 128)
6. **Apply BH-FDR**: Filter candidates using p-values with FDR control

## Key Properties

1. **Per-image null distribution**: Each image gets its own background distribution (accounts for image-specific context)
2. **Non-parametric**: No assumptions about margin distribution shape
3. **Empirical**: Based on actual observed background regions, not theoretical distribution
4. **Smoothed**: The "+1" prevents extreme p-values (0 or 1)

## Edge Cases

### No Background Samples (K=0)
```python
if K == 0:
    return 1.0  # Conservative: assume worst case
```

### Few Candidates (< 10)
```python
if len(candidate_boxes) < min_candidates_for_bg:
    p_values = [1.0] * len(candidate_margins)  # Skip filtering
```

### All Background Margins < Candidate Margin
```python
count_ge = 0
p_value = (1 + 0) / (1 + 128) = 1/129 ≈ 0.0078  # Very low p-value!
```

This means the candidate is **very unlikely** to be background.

## Summary

**P-value formula:**
```
p = (1 + count(background_margins ≥ candidate_margin)) / (1 + 128)
```

**Interpretation:**
- Low p-value → candidate is unlikely to be background → likely a real flower
- High p-value → candidate is similar to background → likely NOT a flower

**Key insight:** We're comparing each candidate against a **per-image null distribution** of background region margins, making the test adaptive to each image's characteristics.
