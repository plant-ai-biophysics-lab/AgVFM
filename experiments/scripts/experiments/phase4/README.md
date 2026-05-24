# Phase 4: Unlabeled Threshold Selection

## Overview

Phase 4 develops methods for selecting confidence thresholds without labeled data, addressing the limitation that Phase 1/2 used an arbitrary threshold (0.1).

## Tools Created

### 1. Distribution Analysis Module
**Location:** `agvfm/analysis/confidence_distribution.py`

**Functions:**
- `compute_bimodality_coefficient()` - Compute BC to detect bimodality
- `find_kde_peaks()` - Find peaks in KDE distribution
- `find_elbow_point()` - Find elbow/knee point in sorted distribution
- `find_largest_gap()` - Find largest gap between consecutive scores
- `analyze_confidence_distribution()` - Comprehensive analysis combining all methods
- `plot_confidence_distribution()` - Visualization of distribution with annotations

### 2. Raw Confidence Score Collector
**Location:** `experiments/scripts/experiments/phase4/collect_raw_confidence_scores.py`

**Purpose:** Collect all raw confidence scores by running inference at very low threshold (0.01)

**Usage:**
```bash
# Collect for all models
python experiments/scripts/experiments/phase4/collect_raw_confidence_scores.py --model all

# Collect for specific model
python experiments/scripts/experiments/phase4/collect_raw_confidence_scores.py --model yolo_world

# Specify data root
python experiments/scripts/experiments/phase4/collect_raw_confidence_scores.py --model all --data-root /path/to/data
```

**Output:** JSON files with raw confidence scores for each Phase 3 config

## Workflow

### Step 1: Collect Raw Confidence Scores
```bash
python experiments/scripts/experiments/phase4/collect_raw_confidence_scores.py --model all
```

This will:
- Run inference at conf=0.01 for all Phase 3 configs
- Collect all confidence scores (not filtered)
- Save to `experiments/results/phase4_unlabeled_threshold_selection/`

### Step 2: Analyze Distributions
```python
from agvfm.analysis import analyze_confidence_distribution
import json
import numpy as np

# Load raw confidence scores
with open("experiments/results/phase4_unlabeled_threshold_selection/yolo_world_C2_not_bud_calyx_raw_confidence.json") as f:
    data = json.load(f)

confidences = np.array(data["confidences"])

# Analyze distribution
analysis = analyze_confidence_distribution(confidences, config_name="C2_not_bud_calyx")

# Print suggested thresholds
for method, threshold in analysis["suggested_thresholds"]:
    print(f"{method}: {threshold:.4f}")
```

### Step 3: Compare to Optimal Thresholds
Compare distribution-suggested thresholds to Phase 3 optimal thresholds to validate the method.

## Next Steps

1. **Run collection script** to gather raw confidence scores
2. **Analyze distributions** for all Phase 3 configs
3. **Compare suggested thresholds** to optimal thresholds from Phase 3
4. **Validate method** - does it work without labels?
5. **Apply to Phase 1/2 subset** to validate rankings

## Methods Implemented

1. **Bimodality Detection:** BC > 0.555 indicates bimodal distribution
2. **KDE Peak Detection:** Find peaks in kernel density estimate
3. **Elbow Detection:** Find natural break point in sorted distribution
4. **Gap Analysis:** Find largest gap between consecutive scores

## Notes

- Distribution analysis requires raw confidence scores (not just aggregated metrics)
- Collection script runs at very low threshold (0.01) to capture all detections
- Analysis is fully unlabeled - no GT needed
- Can be applied retrospectively to Phase 1/2 configs
