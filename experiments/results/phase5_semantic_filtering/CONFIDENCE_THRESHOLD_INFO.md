# Confidence Threshold Information

## Default Confidence Threshold

**Default in our YOLO World wrapper: `conf_threshold = 0.1`**

This is the default value we set in our wrapper code:
- `YOLOWorldModel.predict()` method
- `YOLOWorldModel.predict_multi_class()` method  
- `YOLOWorldModel.predict_batch()` method

Defined in: `agvfm/models/yolo_world.py`

**Note**: This is **our custom default**, not the Ultralytics library default. The Ultralytics YOLOWorld library appears to use `conf=0.25` as its default (found in source code inspection). We chose `0.1` as our default to be more permissive (higher recall) for our use case.

## Baseline Comparison Threshold

**Baseline threshold used in Phase 5 comparison: `conf_threshold = 0.35`**

This was **not** the default, but rather the **optimal threshold** found through systematic confidence sweeps in Phase 3.

### How the Optimal Threshold Was Found

1. **Phase 3 Confidence Sweeps**: Systematic evaluation across confidence thresholds from 0.05 to 0.90 (step 0.05)
2. **Evaluation**: Tested on 178 images (158 test + 20 dev) with multiple prompt configurations
3. **Selection Criteria**: Chose threshold with **highest F1 score** across all configurations
4. **Result**: `conf=0.35` achieved the best F1 score (0.492) for the top-performing configuration (C1)

### Performance Comparison

| Threshold | F1 | Precision | Recall | Source |
|-----------|----|-----------|--------|--------|
| **Default (0.1)** | ~0.365 | ~0.254 | ~0.681 | From Phase 3 results |
| **Optimal (0.35)** | **0.489** | **0.423** | **0.580** | Baseline used in Phase 5 |

**Improvement from default to optimal:**
- F1: +34% (0.365 → 0.489)
- Precision: +67% (0.254 → 0.423)
- Recall: -15% (0.681 → 0.580)

### Why Use Optimal Instead of Default?

The default threshold (0.1) is:
- **Too permissive** - Accepts many low-confidence detections
- **High recall but low precision** - Many false positives
- **Not tuned for this specific task** - Generic default for general use

The optimal threshold (0.35) is:
- **Task-specific** - Tuned for cowpea flower detection
- **Better balance** - Optimizes F1 score (harmonic mean of precision and recall)
- **Higher precision** - Fewer false positives while maintaining good recall

### Phase 3 Findings

From Phase 3 confidence threshold analysis:
- **Optimal F1 thresholds**: 0.30-0.45 (varies by prompt configuration)
- **Low sensitivity**: YOLO World is robust to threshold choice (<0.02 F1 change per ±0.05 threshold)
- **Default (0.1) performance**: F1 ≈ 0.365, Precision ≈ 0.254, Recall ≈ 0.698
- **Optimal (0.35) performance**: F1 ≈ 0.492, Precision ≈ 0.436, Recall ≈ 0.566

### Recommendation

For **zero-shot deployment** without labeled data:
- **Use default (0.1)** if you need maximum recall and can tolerate many false positives
- **Use optimal (0.35)** if you want best overall F1 performance (recommended)
- **Use conf=0.50** as a conservative default that matches ground truth density (~6.4 predictions/image)

For **Phase 5 comparison**, we used the optimal threshold (0.35) as the baseline to show:
1. How well a tuned confidence threshold performs
2. Whether semantic filtering can improve upon this baseline
3. The value (or lack thereof) of semantic filtering for this task

## Summary

- **Code default**: `conf_threshold = 0.1` (generic, high recall)
- **Baseline used**: `conf_threshold = 0.35` (optimal from Phase 3, best F1)
- **Performance**: Optimal threshold significantly outperforms default (+34% F1)
- **Finding**: Optimal confidence threshold (0.35) outperforms all semantic filtering methods
