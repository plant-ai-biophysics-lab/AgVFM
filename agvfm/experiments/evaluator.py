"""Evaluation runner for comparing models on test datasets."""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from agvfm.data import get_image_paths, load_ground_truth
from agvfm.evaluation import compute_counting_metrics, compute_f1_max, compute_map_coco, compute_metrics_at_iou
from agvfm.models.base import BaseModel


class Evaluator:
    """Evaluator for running comprehensive model evaluations."""

    def __init__(
        self,
        model: BaseModel,
        images_dir: Path,
        labels_dir: Path,
        conf_threshold: float = 0.1,
        image_paths: List[Path] = None,
        batch_size: int = 6,
    ):
        """
        Initialize evaluator.

        Args:
            model: Model instance (YOLO World, SAM3, etc.)
            images_dir: Directory containing test images
            labels_dir: Directory containing YOLO format labels
            conf_threshold: Confidence threshold for predictions
            image_paths: Optional list of specific image paths to use (if None, loads all from images_dir)
            batch_size: Batch size for inference (default: 6)
        """
        self.model = model
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.conf_threshold = conf_threshold
        self.batch_size = batch_size

        # Load image paths
        if image_paths is not None:
            self.image_paths = image_paths
        else:
            self.image_paths = get_image_paths(images_dir)
        print(f"Loaded {len(self.image_paths)} images from {images_dir}")

    def evaluate_prompt(
        self,
        prompt: str,
        absorber_classes: List[str] = None,
        target_indices: List[int] = None,
        iou_thresholds: List[float] = None,
        map_coco_style: bool = False,
    ) -> Dict:
        """
        Evaluate a single prompt configuration.

        Args:
            prompt: Text prompt for detection
            absorber_classes: Optional list of classes for absorber architecture
            target_indices: Optional indices of target classes to keep
            iou_thresholds: List of IoU thresholds for evaluation (default: [0.5])
            map_coco_style: If True, also compute mAP@0.5:0.95 and include
                            it as ``map_coco`` in the returned dict.
                            Disabled by default because it requires 10× the
                            matching work and is slow over large datasets.

        Returns:
            Dictionary with comprehensive evaluation results
        """
        if iou_thresholds is None:
            iou_thresholds = [0.5]

        # Collect predictions and ground truth
        list_pred_xyxy = []
        list_pred_conf = []
        list_gt_xyxy = []

        print(f"\nEvaluating prompt: {prompt}")
        if absorber_classes:
            print(f"  Absorber classes: {absorber_classes}")
            print(f"  Target indices: {target_indices}")
        print(f"  Using batch size: {self.batch_size}")

        # Process in batches if batch_size > 1 and model supports it
        use_batching = (
            self.batch_size > 1
            and not absorber_classes  # Batching not supported for absorber architecture yet
            and hasattr(self.model, "predict_batch")
        )

        if use_batching:
            # Batch processing
            for batch_start in range(0, len(self.image_paths), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(self.image_paths))
                batch_paths = self.image_paths[batch_start:batch_end]
                
                if (batch_start // self.batch_size + 1) % 10 == 0 or batch_end == len(self.image_paths):
                    print(f"  Processing batch {batch_start // self.batch_size + 1} ({batch_end}/{len(self.image_paths)})...")

                # Load ground truth for batch
                for img_path in batch_paths:
                    gt_boxes = load_ground_truth(img_path, self.labels_dir)
                    list_gt_xyxy.append(gt_boxes)

                # Run batch prediction
                batch_results = self.model.predict_batch(
                    batch_paths,
                    prompt,
                    conf_threshold=self.conf_threshold,
                )

                # Unpack batch results
                for pred_boxes, pred_confs in batch_results:
                    list_pred_xyxy.append(pred_boxes)
                    list_pred_conf.append(pred_confs)
        else:
            # Sequential processing (original behavior)
            for idx, img_path in enumerate(self.image_paths, 1):
                if idx % 20 == 0:
                    print(f"  Processing {idx}/{len(self.image_paths)}...")

                # Load ground truth
                gt_boxes = load_ground_truth(img_path, self.labels_dir)
                list_gt_xyxy.append(gt_boxes)

                # Run prediction
                if absorber_classes and hasattr(self.model, "predict_multi_class"):
                    # Use absorber architecture (YOLO World only)
                    pred_boxes, pred_confs = self.model.predict_multi_class(
                        img_path,
                        class_names=absorber_classes,
                        target_indices=target_indices,
                        conf_threshold=self.conf_threshold,
                    )
                else:
                    # Single class prediction
                    pred_boxes, pred_confs = self.model.predict(
                        img_path,
                        prompt,
                        conf_threshold=self.conf_threshold,
                    )

                list_pred_xyxy.append(pred_boxes)
                list_pred_conf.append(pred_confs)

        # Compute metrics at each IoU threshold.
        # All scalar metrics (F1, precision, recall, TP/FP/FN) are reported at
        # the F1-maximising confidence threshold (threshold-free).
        # map uses 101-point interpolated AP from the full sorted P-R curve.
        metrics_by_iou = {}
        for iou_thresh in iou_thresholds:
            m_fixed = compute_metrics_at_iou(
                list_gt_xyxy, list_pred_xyxy, list_pred_conf, iou_threshold=iou_thresh
            )
            m_fmax = compute_f1_max(
                list_gt_xyxy, list_pred_xyxy, list_pred_conf, iou_threshold=iou_thresh
            )
            metrics_by_iou[f"iou_{iou_thresh}"] = {
                "precision":    m_fmax["precision"],
                "recall":       m_fmax["recall"],
                "map":          m_fixed["map"],     # 101-pt AP from full curve
                "f1":           m_fmax["f1_max"],
                "total_tp":     m_fmax["total_tp"],
                "total_fp":     m_fmax["total_fp"],
                "total_fn":     m_fmax["total_fn"],
                "n_images":     m_fixed["n_images"],
                "n_gt_total":   m_fixed["n_gt_total"],
                "n_pred_total": m_fixed["n_pred_total"],
                "best_conf":    m_fmax["best_conf"],
            }

        # Optionally compute mAP@0.5:0.95 (COCO-style) — expensive, off by default
        map_coco = compute_map_coco(list_gt_xyxy, list_pred_xyxy, list_pred_conf) if map_coco_style else None

        # Compute counting metrics
        counting_metrics = compute_counting_metrics(list_gt_xyxy, list_pred_xyxy)

        # Aggregate results
        total_predictions = sum(len(boxes) for boxes in list_pred_xyxy)
        total_ground_truth = sum(len(boxes) for boxes in list_gt_xyxy)

        result = {
            "prompt": prompt,
            "model": self.model.model_name,
            "absorber_classes": absorber_classes,
            "target_indices": target_indices,
            "conf_threshold": self.conf_threshold,
            "n_images": len(self.image_paths),
            "total_predictions": total_predictions,
            "total_ground_truth": total_ground_truth,
            "metrics_by_iou": metrics_by_iou,
            "counting": counting_metrics,
        }
        if map_coco is not None:
            result["map_coco"] = map_coco
        return result

    def evaluate_confidence_sweep(
        self,
        prompt: str,
        conf_thresholds: List[float],
        iou_threshold: float = 0.5,
        absorber_classes: List[str] = None,
        target_indices: List[int] = None,
    ) -> Dict:
        """
        Evaluate prompt across multiple confidence thresholds.

        Args:
            prompt: Text prompt
            conf_thresholds: List of confidence thresholds to test
            iou_threshold: IoU threshold for evaluation
            absorber_classes: Optional absorber classes
            target_indices: Optional target indices

        Returns:
            Dictionary with results for each confidence threshold
        """
        results = {}

        for conf_thresh in conf_thresholds:
            print(f"\nEvaluating at conf_threshold={conf_thresh}...")
            self.conf_threshold = conf_thresh

            eval_result = self.evaluate_prompt(
                prompt=prompt,
                absorber_classes=absorber_classes,
                target_indices=target_indices,
                iou_thresholds=[iou_threshold],
            )

            results[conf_thresh] = {
                "metrics": eval_result["metrics_by_iou"][f"iou_{iou_threshold}"],
                "counting": eval_result["counting"],
                "total_predictions": eval_result["total_predictions"],
            }

        return {
            "prompt": prompt,
            "model": self.model.model_name,
            "iou_threshold": iou_threshold,
            "conf_thresholds": conf_thresholds,
            "results_by_conf": results,
        }
