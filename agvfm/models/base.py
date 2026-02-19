"""Base model interface for zero-shot detection/segmentation models."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


class BaseModel(ABC):
    """Abstract base class for zero-shot vision models."""

    @abstractmethod
    def predict(
        self,
        image_path: Path,
        prompt: str,
        conf_threshold: float = 0.1,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run inference on a single image.

        Args:
            image_path: Path to image file
            prompt: Text prompt describing objects to detect
            conf_threshold: Confidence threshold for filtering predictions
            **kwargs: Additional model-specific arguments

        Returns:
            Tuple of (boxes, confidences):
            - boxes: Array of shape (N, 4) with boxes in xyxy format
            - confidences: Array of shape (N,) with confidence scores
        """
        pass

    @abstractmethod
    def predict_batch(
        self,
        image_paths: List[Path],
        prompt: str,
        conf_threshold: float = 0.1,
        **kwargs,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Run inference on multiple images.

        Args:
            image_paths: List of paths to image files
            prompt: Text prompt describing objects to detect
            conf_threshold: Confidence threshold for filtering predictions
            **kwargs: Additional model-specific arguments

        Returns:
            List of (boxes, confidences) tuples, one per image
        """
        pass

    @abstractmethod
    def set_classes(self, class_names: List[str]) -> None:
        """
        Set class names for multi-class detection (if supported).

        Args:
            class_names: List of class name strings
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name/identifier."""
        pass
