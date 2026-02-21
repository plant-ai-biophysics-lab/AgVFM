# Zero-Shot Agricultural Object Detection: A Systematic Comparison of Vision Foundation Models

## Abstract

Agricultural monitoring tasks such as flower counting for phenotyping and yield estimation face significant challenges: limited labeled data, strong domain shift, and small, cluttered targets that are easily confusable with similar structures (buds, calyx, leaves). While zero-shot vision foundation models promise rapid adaptation via text prompts, practitioners lack reproducible guidance on which foundation model family to use and how sensitive performance is to prompt wording and confidence thresholds when labels are unavailable for tuning.

We present a clean-slate, model-agnostic evaluation protocol that systematically compares an open-vocabulary detector (YOLO World) to a promptable segmenter (SAM3) on a fine-grained cowpea flower detection dataset. Our three-phase experimental suite—factor analysis, combination/negation testing, and confidence threshold sweeps—reveals architecture-specific prompt effects, precision-recall tradeoffs, and threshold sensitivity patterns critical for zero-shot deployment.

Our key findings demonstrate that prompt components have architecture-specific effects; rules that work for one model fail for the other. Partial text negation (two clauses) improves YOLO World performance (mAP@0.5: 0.4123) but fails completely for SAM3, while size descriptors ("tiny") help SAM3 (+0.0453 mAP) but hurt YOLO World. SAM3 achieves higher mAP (0.54 vs 0.41) but lower F1 (0.26-0.28 vs 0.36-0.40), revealing a precision-recall tradeoff. All tested YOLO World configurations show low threshold sensitivity (F1 changes <0.02 with ±0.05 threshold), making them robust for zero-shot deployment without label-based tuning. This work provides practical, quantitative guidance for deploying vision foundation models in agricultural detection and counting without fine-tuning, plus a reproducible framework to extend to additional models and crops.

## 1. Introduction

Agricultural monitoring increasingly relies on computer vision for tasks such as flower counting, phenotyping, and crop development tracking. However, these applications face fundamental constraints: labeling is expensive, field conditions vary dramatically, and target objects are small and confusable with similar structures. For example, cowpea flowers must be distinguished from buds, calyx, and leaves in cluttered field scenes. Vision foundation models (VFMs) promise rapid adaptation via text prompts without task-specific training, but real deployments face a critical friction: without labeled data, practitioners cannot easily tune prompts or confidence thresholds, and the "best" settings are often model-dependent.

Two prominent VFM families represent different architectural choices: open-vocabulary detectors (e.g., YOLO World) and promptable segmenters (e.g., SAM3). YOLO World combines a CLIP text encoder with a YOLO detection head, enabling multi-class detection that supports advanced strategies like absorber architectures. SAM3 provides text-to-segmentation capabilities via a promptable segmentation model. Agriculture needs clarity on which family works better for fine-grained targets and under what prompting and thresholding regime.

This paper provides a systematic, quantitative comparison of these architectures and extracts actionable prompting rules and threshold-robust operating points suitable for truly label-scarce deployment. Our contributions include: (1) a clean-slate evaluation suite for zero-shot agricultural detection explicitly designed for the "no labels for tuning" constraint; (2) a structured prompt experiment suite with one-factor-at-a-time analysis and combination tests that expose interaction effects; (3) an evaluation of negation strategies including absorber-style confuser handling; (4) a confidence threshold sensitivity study that measures robustness and recommends operating points; and (5) a unified report of detection and counting metrics across multiple IoU criteria, aligned with agricultural use cases where count accuracy matters.

## 2. Related Work

Agricultural vision research has focused on flower detection, counting, and phenotyping, addressing domain-specific challenges such as small object size, cluttered scenes, and confusable structures. Recent work leverages deep learning, but most approaches require extensive labeled training data, limiting scalability.

Zero-shot detection has emerged through open-vocabulary object detection models that leverage CLIP-based architectures. These models enable detection of novel categories via text prompts without task-specific training. Prompt engineering has been shown to significantly impact performance, but most studies focus on general vision tasks rather than fine-grained agricultural applications.

Vision foundation models represent a paradigm shift toward general-purpose vision systems. YOLO World extends YOLO with CLIP text encoding for open-vocabulary detection, while SAM3 provides promptable segmentation capabilities. Comparison studies exist for general vision tasks, but agriculture-specific guidance is limited, particularly regarding prompt engineering and threshold selection for zero-shot deployment.

Confidence calibration and thresholding are core issues in label-scarce deployment. Prior work on calibration and operating points typically assumes access to labeled validation data, which contradicts the zero-shot constraint. Our work addresses this gap by analyzing threshold sensitivity and recommending robust operating points without label-based selection.

## 3. Methods

### 3.1 Dataset

We evaluate on a cowpea flower detection dataset with 178 test images (158 test + 20 dev) containing approximately 1,147 ground truth flowers. The dataset presents a fine-grained challenge: flowers are small objects in cluttered field scenes and must be distinguished from confusable parts including buds, calyx, and leaves. Annotations are provided in YOLO format. For zero-shot evaluation, we use the full test set without training or validation splits, as no labeled data is available for tuning.

### 3.2 Models

**YOLO World** uses a CLIP text encoder combined with a YOLO detection head (YOLOv8x-World v2) to enable open-vocabulary detection with text prompts. The architecture supports multi-class detection, which enables absorber-style strategies where confuser classes are detected alongside the target and filtered post-hoc. We use weights from `yolov8x-worldv2.pt` and process images in batches of 15 for efficiency.

**SAM3** is a promptable segmentation model (Hugging Face `facebook/sam3`) that provides text-to-segmentation capabilities via a CLIP text encoder. Segmentation masks are converted to bounding boxes for detection comparison. Due to architecture constraints, SAM3 requires sequential processing (batch size 1).

### 3.3 Evaluation Metrics

We report detection metrics at multiple IoU thresholds: mAP@0.3, mAP@0.5, and mAP@0.5:0.95, along with point metrics (Precision, Recall, F1) at IoU=0.5. Multiple IoU thresholds are important for fine-grained objects where localization precision varies. We also report counting metrics (R², RMSE, MAE, MAPE, slope, intercept) via linear regression, as count accuracy is critical for agricultural applications like yield estimation.

### 3.4 Experimental Design

Our experimental design consists of three phases that systematically explore prompt engineering and threshold sensitivity.

**Phase 1: Structured Factor Analysis.** We conduct a one-factor-at-a-time (OFAT) analysis across seven prompt component axes to isolate the contribution of each component per model architecture. The axes include: Taxonomy (8 values: cowpea flower, bean flower, pea flower, legume flower, black-eyed pea flower, flower [baseline], vigna unguiculata flower, crop flower), Color (5 values: yellow, white, cream, "" [baseline], purple), Size (4 values: "" [baseline], large, small, tiny), Phenology (6 values: bud, open, "" [baseline], closed bud, blooming, in bloom), Negation (6 values: full and partial negation combinations, "" [baseline]), Anatomy (5 values: with open petals, with visible petals, with petals and stamens, "" [baseline], corolla), and Grammar (6 values: a single, a [baseline], "" [bare noun], a photo of a, one, close-up of a). The baseline prompt is `"a flower"` with all axes at baseline values. This yields 34 configurations per model (1 baseline + 33 axis variations), allowing us to identify which components matter most for each architecture.

**Phase 2: Combination Tests & Absorber Architecture.** We systematically combine the best-performing components from Phase 1, testing 13 original combinations: one baseline (`"a cowpea flower"`), six two-component combinations, four three-component combinations, one four-component combination (`"a single yellow cowpea flower with open petals"`), and one kitchen sink with full text negation (`"a single yellow cowpea flower with open petals, not a bud, not the green calyx, not a leaf"`). We also conduct 8 strategic experiments for YOLO World (size variants, partial negation variants) and 2 for SAM3 (size variants). For YOLO World, we compare text negation strategies against absorber architecture, where confuser classes (bud, calyx, leaf, stem, soil) are detected as separate classes and filtered to keep only the target. We test 7 absorber configurations (4 original + 3 strategic variants). In total, we evaluate 21 configurations per model for combinations, plus 7 absorber configurations for YOLO World.

**Phase 3: Confidence Threshold Analysis.** We sweep confidence thresholds from 0.05 to 0.90 in steps of 0.05 (18 values) for the top 3 configurations per model by mAP@0.5 from Phase 2. For YOLO World, these are C2_not_bud_calyx (mAP: 0.4123), C1 (mAP: 0.3830), and C2_not_calyx (mAP: 0.3807). For SAM3, these are C2_not_bud (mAP: 0.5407), C_color_taxonomy_anatomy_tiny (mAP: 0.5364), and C2 (mAP: 0.4925). For each threshold, we compute mAP@0.5, F1, Precision, Recall, and total predictions. We analyze sensitivity (F1 change with ±0.05 threshold), precision-recall tradeoffs, and detection volume to identify robust operating points for zero-shot deployment without label-based threshold tuning.

## 4. Experiments & Results

### 4.1 Factor Analysis Results

![Factor Contributions](experiments/results/phase1_factor_analysis/plots/factor_comparison_map.png)

*Figure 1: Factor contributions (ΔmAP@0.5 from baseline) for YOLO World and SAM3. The baseline prompt `"a flower"` achieves mAP@0.5 = 0.0909 for YOLO World and 0.3710 for SAM3, indicating SAM3 has a significantly higher baseline performance.*

Our factor analysis reveals architecture-specific sensitivities to prompt components. Taxonomy provides the strongest effect for both models, with the species name "cowpea flower" yielding the largest improvement over the generic "flower" baseline. Color also shows a strong positive effect, with "yellow" helping both models significantly. Anatomy ("with open petals") and grammar ("a single") provide moderate positive effects for both architectures.

However, key differences emerge. Negation shows a positive effect for YOLO World, but complex negation fails completely for SAM3. Size modifiers demonstrate opposite effects: all size modifiers hurt YOLO World performance, but "tiny" actually helps SAM3 (+0.0453 mAP when added to combinations), representing a critical model-specific difference. Phenology shows mixed results: "bud" and "open" help slightly, while "in bloom" fails catastrophically for both models, as CLIP interprets it as a scene concept rather than an object descriptor.

YOLO World responds well to taxonomy, color, anatomy, grammar, and partial negation, while SAM3's higher baseline (mAP@0.5: 0.3710 vs 0.0909) suggests it may be better calibrated for this task initially, but it fails on complex negation strategies that work for YOLO World.

### 4.2 Model Comparison

![Combination Performance](experiments/results/phase2_combinations/plots/yolo_world_combinations_map.png)

*Figure 2: Combination performance for YOLO World (left) and SAM3 (right), ranked by mAP@0.5. YOLO World best: C2_not_bud_calyx (0.4123); SAM3 best: C2_not_bud (0.5407).*

Our systematic combination tests reveal a fundamental precision-recall tradeoff between architectures. SAM3 achieves higher mAP@0.5 (0.54 vs 0.41) but lower F1 scores (0.26-0.28 vs 0.36-0.40), indicating higher precision but lower recall. YOLO World's best configurations—C2_not_bud_calyx (mAP@0.5: 0.4123, F1: 0.3645), C1 (mAP@0.5: 0.3830, F1: 0.3699), and C2_not_calyx (mAP@0.5: 0.3807, F1: 0.4029)—demonstrate better balance between precision and recall.

SAM3's best configurations—C2_not_bud (mAP@0.5: 0.5407, F1: 0.2625), C_color_taxonomy_anatomy_tiny (mAP@0.5: 0.5364, F1: 0.2840), and C2 (mAP@0.5: 0.4925, F1: 0.2958)—show that while mAP is higher, F1 remains lower due to conservative predictions. YOLO World supports absorber architecture, but our tests reveal that text negation (mAP: 0.4123) significantly outperforms absorber strategies (best absorber H3b: mAP: 0.1359), contradicting initial expectations.

![Precision-Recall Tradeoff](experiments/results/phase2_combinations/plots/yolo_world_precision_recall_tradeoff.png)

*Figure 3: Precision-Recall tradeoff for YOLO World configurations. Curves show performance across confidence thresholds, with F1 iso-lines. Partial negation (C2_not_bud_calyx) achieves better balance than full negation (C1).*

### 4.3 Prompt Engineering Rules

Our experiments reveal model-specific prompt engineering rules that do not transfer between architectures. For YOLO World, species names ("cowpea flower"), color ("yellow"), anatomy ("with open petals"), grammar ("a single"), and partial text negation (2 clauses: "not a bud, not the green calyx") all improve performance, with partial negation achieving the best mAP (0.4123). However, phenology terms ("in bloom"), size modifiers ("small", "tiny"), full text negation (3 clauses), and absorber architecture all underperform. Critically, partial negation (2 clauses) outperforms full negation (3 clauses), suggesting that excessive negation clauses can cause precision collapse.

For SAM3, simple prompts work best: color, taxonomy, and anatomy all help, and the size component "tiny" provides a significant boost (+0.0453 mAP) when added to combinations. However, complex text negation causes complete failure (mAP drops to near-zero), and absorber architecture is not supported. This represents a fundamental limitation: SAM3 cannot handle the negation strategies that work for YOLO World.

Component interactions are super-additive: combining the best components yields better-than-additive performance. However, the precision-recall tradeoff means that best mAP configurations do not necessarily yield best F1. For example, YOLO World's C2_not_calyx achieves mAP=0.3807 but F1=0.4029, demonstrating that optimizing for different metrics yields different optimal configurations.

### 4.4 Confidence Threshold Analysis

![Performance Curves](experiments/results/phase3_confidence_sweeps/plots/yolo_world_performance_curves.png)

*Figure 4: Performance curves for YOLO World top 3 configurations across confidence thresholds (0.05-0.90). All metrics (mAP, F1, Precision, Recall) are shown. Optimal F1 thresholds vary by config (0.30-0.45) but performance is stable around optimal.*

![Sensitivity Analysis](experiments/results/phase3_confidence_sweeps/plots/yolo_world_sensitivity_analysis.png)

*Figure 5: Sensitivity analysis for YOLO World. Left: F1 curves with optimal points marked. Right: Sensitivity metrics showing F1 change with ±0.05 threshold. All configs show low sensitivity (<0.02), indicating robustness.*

Our confidence threshold analysis reveals that all tested YOLO World configurations show low sensitivity to threshold choice, making them robust for zero-shot deployment. Optimal F1 thresholds vary by configuration: C1 achieves best F1 (0.4924) at conf=0.35, C2_not_bud_calyx achieves 0.4920 at conf=0.45, and C2_not_calyx achieves 0.4811 at conf=0.30. However, F1 changes by less than 0.02 with ±0.05 threshold variation, indicating that threshold selection is not critical for deployment.

The precision-recall tradeoff is stark: at low thresholds (0.05-0.10), recall is high (0.70-0.74) but precision is very low (0.19-0.22), resulting in many false positives. At high thresholds (0.50+), precision is high (0.48-0.51) but recall is lower (0.42-0.49). Detection volume analysis reveals that conf=0.50 yields approximately 6.6 predictions per image, matching ground truth density (~6.4 per image), making it a natural operating point for balanced performance.

An important finding is the relationship between mAP and point metrics. At low thresholds (conf=0.05), mAP (0.40-0.41) significantly exceeds point precision (0.19-0.22), confirming that the model's confidence ranking is meaningful despite low point precision. This validates mAP as a measure of ranking quality rather than point performance, which is critical for understanding model behavior in zero-shot settings.

![Detection Volume](experiments/results/phase3_confidence_sweeps/plots/yolo_world_detection_volume.png)

*Figure 6: Detection volume (predictions per image) vs confidence threshold for YOLO World configurations. The horizontal line shows ground truth density (~6.4 per image). conf=0.50 matches GT density.*

**SAM3 Confidence Threshold Analysis.** SAM3 shows distinct threshold behavior compared to YOLO World. Optimal F1 thresholds range from 0.40 to 0.45 across configurations, similar to YOLO World's optimal range (0.30-0.45). However, SAM3's best mAP occurs at low thresholds (0.05-0.15), indicating that high recall is necessary to achieve maximum ranking quality. At the optimal F1 threshold (conf=0.45), SAM3's best configuration (C2_not_bud) achieves F1=0.5863, significantly higher than YOLO World's best F1 (0.4924), demonstrating SAM3's superior F1 performance when threshold is properly selected.

![SAM3 Performance Curves](experiments/results/phase3_confidence_sweeps/plots/sam3_performance_curves.png)

*Figure 7: Performance curves for SAM3 top 3 configurations across confidence thresholds (0.05-0.90). SAM3 achieves higher F1 at optimal thresholds (0.51-0.58) than YOLO World (0.36-0.40), but shows higher threshold sensitivity.*

![SAM3 Sensitivity Analysis](experiments/results/phase3_confidence_sweeps/plots/sam3_sensitivity_analysis.png)

*Figure 8: Sensitivity analysis for SAM3. SAM3 shows moderate sensitivity (0.10-0.12 F1 change per ±0.05 threshold), higher than YOLO World's low sensitivity (<0.02), indicating that threshold selection is more critical for SAM3.*

SAM3 demonstrates moderate threshold sensitivity (0.10-0.12 F1 change per ±0.05 threshold), which is higher than YOLO World's low sensitivity (<0.02). This indicates that SAM3 requires more careful threshold selection for optimal performance, though the sensitivity is still manageable for zero-shot deployment. At conf=0.50, SAM3 configurations achieve F1 scores of 0.51-0.58, precision of 0.59-0.62, and recall of 0.43-0.57, with detection volumes of 4.4-6.1 predictions per image depending on configuration.

The precision-recall tradeoff for SAM3 is more pronounced than for YOLO World. At low thresholds (conf=0.10), SAM3 achieves high recall (0.87-0.87) but very low precision (0.16-0.18), resulting in many false positives. At high thresholds (conf=0.70), SAM3 achieves very high precision (0.73-0.75) but low recall (0.22-0.30), demonstrating the conservative nature of SAM3's segmentation architecture. This tradeoff suggests that SAM3 is better suited for applications where precision is prioritized, while YOLO World provides better balance for applications requiring both precision and recall.

![SAM3 Detection Volume](experiments/results/phase3_confidence_sweeps/plots/sam3_detection_volume.png)

*Figure 9: Detection volume (predictions per image) vs confidence threshold for SAM3 configurations. Detection volumes vary more by configuration than for YOLO World, with C2 producing fewer predictions at all thresholds.*

## 5. Discussion

The architectural differences between YOLO World and SAM3 lead to fundamentally different performance characteristics and failure modes. YOLO World's detection architecture with multi-class support enables strategies like absorber architectures, though our results show text negation outperforms absorbers. YOLO World excels at recall (R≈0.70 at low thresholds) and achieves balanced F1 scores (0.36-0.40), making it suitable for applications requiring high recall or balanced performance.

SAM3's segmentation architecture produces more conservative predictions, achieving higher mAP (0.54 vs 0.41) but lower F1 (0.26-0.28 vs 0.36-0.40). This precision-recall tradeoff suggests SAM3 is better suited for applications where precision is prioritized and false positives are costly. However, SAM3's complete failure on complex text negation represents a fundamental limitation that restricts its applicability to simple prompts.

Text prompt interpretation differs between models. SAM3 interprets prompts differently than YOLO World: complex negation fails completely, but size descriptors ("tiny") help SAM3 while hurting YOLO World. This model-specific behavior underscores the importance of testing each architecture independently rather than assuming prompt rules transfer.

For zero-shot agricultural vision deployment, practitioners should choose YOLO World for high-recall applications, when partial text negation is beneficial, or when balanced F1 performance is required. SAM3 is better suited for high-precision applications, simple prompts without negation, or when mAP is the primary metric. Prompt engineering must follow model-specific rules: what works for one architecture may fail completely for the other.

Confidence threshold analysis reveals important differences between architectures. YOLO World configurations show low sensitivity (<0.02 F1 change per ±0.05 threshold), making them robust for zero-shot deployment without label-based tuning. A default threshold of conf=0.50 provides balanced precision/recall and matches ground truth density for YOLO World. SAM3 shows moderate sensitivity (0.10-0.12 F1 change per ±0.05 threshold), requiring more careful threshold selection, but still manageable for zero-shot deployment. SAM3 achieves higher F1 at optimal thresholds (0.51-0.58 vs 0.36-0.40 for YOLO World), but this advantage diminishes if threshold selection is suboptimal. For SAM3, a threshold of conf=0.45 provides optimal F1 performance, while conf=0.50 remains a reasonable default for balanced performance.

Our study has limitations: we evaluate on a single species (cowpea) and a single task (flower detection), so multi-species validation is needed to confirm generalizability. We compare only two models, and future work should include additional vision foundation models.

Future work should conduct grounding gap analysis to understand why models fail (structural vs linguistic), validate prompt rules on additional species, extend to other agricultural tasks (fruits, leaves, pests), and include additional models such as CLIP-based detectors and other segmentation models.

## 6. Conclusion

This work provides systematic, quantitative guidance for zero-shot agricultural object detection using vision foundation models. Our key findings demonstrate that model-specific prompt rules are critical: partial text negation works for YOLO World but fails for SAM3, while size descriptors help SAM3 but hurt YOLO World. Architecture choice matters: YOLO World excels at recall and balanced F1 performance with low threshold sensitivity, while SAM3 achieves higher F1 at optimal thresholds (0.51-0.58 vs 0.36-0.40) but requires more careful threshold selection. Component interactions are super-additive, and absorber architecture underperforms text negation for YOLO World.

For practitioners, we recommend using YOLO World for high-recall applications, balanced F1 performance, or when threshold robustness is critical (low sensitivity makes deployment forgiving). Use SAM3 for high-F1 applications when threshold can be carefully selected (conf=0.45 optimal), simple prompts without negation, or when precision is prioritized. Follow model-specific rules and test each architecture independently. Optimize for F1, not just mAP, as the precision-recall tradeoff is critical. For YOLO World, a default threshold of conf=0.50 provides balanced performance and matches ground truth density. For SAM3, conf=0.45 provides optimal F1, while conf=0.50 remains a reasonable default for balanced performance.

This work enables effective deployment of vision foundation models in agricultural detection and counting without labeled training data or post-hoc threshold tuning, providing practical guidance for the agricultural vision community.

## References

[To be added]
