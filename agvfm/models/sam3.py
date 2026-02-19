"""SAM3 model implementation using Hugging Face Transformers."""

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from agvfm.models.base import BaseModel


class SAM3Model(BaseModel):
    """SAM3 model wrapper for zero-shot segmentation."""

    def __init__(
        self,
        model_id: str = "facebook/sam3",
        device: Optional[str] = None,
        token: Optional[str] = None,
    ):
        """
        Initialize SAM3 model from Hugging Face.

        Args:
            model_id: Hugging Face model ID (default: "facebook/sam3")
            device: Device to run on (default: "cuda" if available, else "cpu")
            token: Hugging Face token for gated models (or use HF_TOKEN env var)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_id = model_id
        self.device = device
        self.token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

        try:
            from transformers import Sam3Model, Sam3Processor

            self._Sam3Model = Sam3Model
            self._Sam3Processor = Sam3Processor
        except ImportError:
            raise ImportError(
                "transformers library not installed. Install with: pip install transformers"
            )

        # Load model and processor
        self.processor = self._Sam3Processor.from_pretrained(model_id, token=self.token)
        self.model = self._Sam3Model.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device,
            token=self.token,
        )
        self.model.eval()

    def set_classes(self, class_names: List[str]) -> None:
        """
        Set class names (not used for SAM3, kept for interface compatibility).

        Args:
            class_names: List of class names (ignored)
        """
        pass  # SAM3 uses text prompts directly, not pre-set classes

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
            prompt: Text prompt describing objects to segment
            conf_threshold: Confidence threshold for filtering
            **kwargs: Additional arguments (ignored for now)

        Returns:
            Tuple of (boxes, confidences) in xyxy format
        """
        # Load image
        image = Image.open(image_path).convert("RGB")

        # Prepare inputs
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process results
        results_list = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=conf_threshold,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist(),
        )

        if not results_list or len(results_list) == 0:
            return np.zeros((0, 4), dtype=np.float64), np.zeros(0, dtype=np.float64)

        # Extract results from first (and only) image in batch
        results_dict = results_list[0]
        boxes = results_dict["boxes"].cpu().numpy()  # (N, 4) in xyxy format
        scores = results_dict["scores"].cpu().numpy()  # (N,)

        # Filter by confidence
        if len(boxes) > 0:
            keep = scores >= conf_threshold
            boxes = boxes[keep]
            scores = scores[keep]

        return boxes.astype(np.float64), scores.astype(np.float64)

    def predict_batch(
        self,
        image_paths: List[Path],
        prompt: str,
        conf_threshold: float = 0.1,
        **kwargs,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Run inference on multiple images in a true batch.

        Args:
            image_paths: List of image paths
            prompt: Text prompt
            conf_threshold: Confidence threshold
            **kwargs: Additional arguments

        Returns:
            List of (boxes, confidences) tuples
        """
        # Load all images
        images = [Image.open(img_path).convert("RGB") for img_path in image_paths]
        
        # Prepare inputs for batch processing
        inputs = self.processor(images=images, text=prompt, return_tensors="pt").to(self.device)
        
        # Run inference on batch
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process results
        results_list = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=conf_threshold,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist(),
        )
        
        # Extract results for each image
        batch_results = []
        for result_dict in results_list:
            if result_dict is None or len(result_dict) == 0:
                batch_results.append((np.zeros((0, 4), dtype=np.float64), np.zeros(0, dtype=np.float64)))
                continue
                
            boxes = result_dict["boxes"].cpu().numpy()  # (N, 4) in xyxy format
            scores = result_dict["scores"].cpu().numpy()  # (N,)
            
            # Filter by confidence
            if len(boxes) > 0:
                keep = scores >= conf_threshold
                boxes = boxes[keep]
                scores = scores[keep]
            
            batch_results.append((boxes.astype(np.float64), scores.astype(np.float64)))
        
        return batch_results

    @property
    def model_name(self) -> str:
        """Return model name."""
        return "SAM3"
