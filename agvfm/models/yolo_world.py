"""YOLO World model implementation."""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from ultralytics import YOLOWorld

from agvfm.models.base import BaseModel


class YOLOWorldModel(BaseModel):
    """YOLO World model wrapper for zero-shot object detection."""

    def __init__(
        self,
        weights_path: str = "model_weights/yolo_world/yolov8x-worldv2.pt",
        device: Optional[str] = None,
    ):
        """
        Initialize YOLO World model.

        Args:
            weights_path: Path to YOLO World weights file
            device: Device to run on (default: auto-detect)
        """
        import torch
        
        self.weights_path = weights_path
        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Initialize model - YOLOWorld automatically detects and uses CUDA if available
        # Match previous working code exactly - no device manipulation
        self.model = YOLOWorld(str(weights_path))
        self._class_names: Optional[List[str]] = None

    def set_classes(self, class_names: List[str]) -> None:
        """
        Set class names for open-vocabulary detection.

        Args:
            class_names: List of class name strings
        """
        self.model.set_classes(class_names)
        self._class_names = class_names

    def predict(
        self,
        image_path: Path,
        prompt: str,
        conf_threshold: float = 0.1,
        iou_threshold: float = 0.5,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run inference on a single image.

        Args:
            image_path: Path to image file
            prompt: Text prompt describing objects to detect
            conf_threshold: Confidence threshold for filtering
            iou_threshold: IoU threshold for NMS
            **kwargs: Additional arguments passed to model.predict()

        Returns:
            Tuple of (boxes, confidences) in xyxy format
        """
        # Set single class if not already set or if prompt changed
        # CRITICAL: Always include "" as background class for proper confidence calibration
        # Previous code always used [prompt, ""] even for single-class detection
        target_classes = [prompt, ""]
        if self._class_names != target_classes:
            # Reinitialize model when changing classes to avoid device errors
            # Calling set_classes repeatedly on the same model instance causes device mismatches
            self.model = YOLOWorld(str(self.weights_path))
            self.set_classes(target_classes)

        # Run inference
        # Use imgsz=1280 and iou=0.3 to match previous experiments
        # Previous code used: imgsz=1280, conf=0.1, iou=0.3 (for NMS, not eval IoU)
        # Note: iou_threshold here is for NMS, not evaluation IoU
        # Also include device if CUDA is available (matches DEFAULT_PREDICT_KWARGS)
        import torch
        predict_kwargs = {
            "imgsz": 1280,
            "conf": conf_threshold,
            "iou": 0.3,  # NMS IoU - previous code used 0.3, not 0.5
            "verbose": False,
        }
        if torch.cuda.is_available():
            predict_kwargs["device"] = "cuda"
        predict_kwargs.update(kwargs)  # User kwargs override defaults
        
        results = self.model.predict(str(image_path), **predict_kwargs)

        r = results[0]
        if r.boxes.xyxy.numel() == 0:
            return np.zeros((0, 4), dtype=np.float64), np.zeros(0, dtype=np.float64)

        boxes = r.boxes.xyxy.cpu().numpy()
        confidences = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)

        # Filter out background class (index 1 when we have [prompt, ""])
        # Only keep predictions for the target class (index 0)
        target_mask = classes == 0
        boxes = boxes[target_mask]
        confidences = confidences[target_mask]

        return boxes, confidences

    def predict_multi_class(
        self,
        image_path: Path,
        class_names: List[str],
        target_indices: Optional[List[int]] = None,
        conf_threshold: float = 0.1,
        iou_threshold: float = 0.5,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run multi-class inference and optionally filter to target classes.

        Args:
            image_path: Path to image file
            class_names: List of class names (target + absorbers)
            target_indices: Indices of target classes to keep (None = keep all)
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            **kwargs: Additional arguments

        Returns:
            Tuple of (boxes, confidences) for target classes only
        """
        # Set classes
        self.set_classes(class_names)

        # Run inference
        # Use imgsz=1280 and iou=0.3 to match previous experiments
        # Previous code used: imgsz=1280, conf=0.1, iou=0.3 (for NMS)
        import torch
        predict_kwargs = {
            "imgsz": 1280,
            "conf": conf_threshold,
            "iou": 0.3,  # NMS IoU - previous code used 0.3
            "verbose": False,
        }
        if torch.cuda.is_available():
            predict_kwargs["device"] = "cuda"
        predict_kwargs.update(kwargs)  # User kwargs override defaults
        
        results = self.model.predict(str(image_path), **predict_kwargs)

        r = results[0]
        if r.boxes.xyxy.numel() == 0:
            return np.zeros((0, 4), dtype=np.float64), np.zeros(0, dtype=np.float64)

        boxes = r.boxes.xyxy.cpu().numpy()
        confidences = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)

        # Filter to target classes if specified
        if target_indices is not None:
            target_set = set(target_indices)
            mask = np.array([c in target_set for c in classes])
            boxes = boxes[mask]
            confidences = confidences[mask]

        return boxes, confidences

    def predict_batch(
        self,
        image_paths: List[Path],
        prompt: str,
        conf_threshold: float = 0.1,
        iou_threshold: float = 0.5,
        **kwargs,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Run inference on multiple images using batch processing.

        Args:
            image_paths: List of image paths
            prompt: Text prompt
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            **kwargs: Additional arguments

        Returns:
            List of (boxes, confidences) tuples
        """
        if len(image_paths) == 0:
            return []
        
        # Set classes once for the batch
        target_classes = [prompt, ""]
        if self._class_names != target_classes:
            self.model = YOLOWorld(str(self.weights_path))
            self.set_classes(target_classes)
        
        # Prepare predict kwargs
        import torch
        predict_kwargs = {
            "imgsz": 1280,
            "conf": conf_threshold,
            "iou": 0.3,  # NMS IoU
            "verbose": False,
        }
        if torch.cuda.is_available():
            predict_kwargs["device"] = "cuda"
        predict_kwargs.update(kwargs)
        
        # Convert paths to strings for batch processing
        image_path_strs = [str(path) for path in image_paths]
        
        # Run batch inference - YOLO World can process multiple images at once
        batch_results = self.model.predict(image_path_strs, **predict_kwargs)
        
        # Process results
        results = []
        for r in batch_results:
            if r.boxes.xyxy.numel() == 0:
                results.append((np.zeros((0, 4), dtype=np.float64), np.zeros(0, dtype=np.float64)))
                continue
            
            boxes = r.boxes.xyxy.cpu().numpy()
            confidences = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)
            
            # Filter out background class (index 1 when we have [prompt, ""])
            target_mask = classes == 0
            boxes = boxes[target_mask]
            confidences = confidences[target_mask]
            
            results.append((boxes, confidences))
        
        return results

    @property
    def model_name(self) -> str:
        """Return model name."""
        return "YOLO World"
