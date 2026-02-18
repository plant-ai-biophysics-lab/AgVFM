"""
Wrappers for Hugging Face transformer models (Grounding DINO, OwlV2)
to mimic the Ultralytics YOLO inference interface used in run_eval.
"""

# install torch
import torch
from PIL import Image
import numpy as np

# install transformers
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

class HuggingFaceModelWrapper:
    """Base wrapper for HF zero-shot detection models."""
    def __init__(self, model_id, device=None, processor_id=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(processor_id or model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        self.model.eval()

    def predict(self, source, **kwargs):
        """
        Run inference on 'source' (file path or PIL Image).
        Returns a result object with a structure similar to Ultralytics:
          result[0].boxes.xyxy (tensor)
          result[0].boxes.conf (tensor)
          result[0].boxes.cls (tensor)
        
        kwargs used:
          - text: list of class names or a single prompt string.
          - conf: confidence threshold (default 0.1)
          - iou: NMS threshold (not always applicable directly in model out, but for post-process)
        """
        # Load image
        if isinstance(source, str):
            image = Image.open(source).convert("RGB")
        else:
            image = source

        # Text prompt handling
        # standard run_eval passes 'classes' list, or we use a default text prompt
        text_queries = kwargs.get("text", kwargs.get("classes", ["object"]))
        if isinstance(text_queries, str):
            text_queries = [text_queries]
        
        # Determine format based on model specific needs
        # Grounding DINO expects a single string with . separator 
        # OwlV2 expects list of strings usually
        
        return self._predict_internal(image, text_queries, **kwargs)

    def _predict_internal(self, image, text_queries, **kwargs):
        raise NotImplementedError

class MockResult:
    """Mimics Ultralytics Result object structure."""
    def __init__(self, boxes, conf, cls_):
        self.boxes = MockBoxes(boxes, conf, cls_)

class MockBoxes:
    def __init__(self, xyxy, conf, cls_):
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls_
        self.data = None # Ultralytics has .data too


class GroundingDinoWrapper(HuggingFaceModelWrapper):
    def _predict_internal(self, image, text_queries, **kwargs):
        # GD typically expects a single string prompt with classes separated by .
        # e.g. "cat . dog ."
        # If text_queries is a list, join them.
        text_prompt = ". ".join(text_queries)
        if not text_prompt.endswith("."):
            text_prompt += "."
            
        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_grounded_object_detection(
            outputs, threshold=kwargs.get("conf", 0.1), target_sizes=target_sizes
        )[0]
        
        # map labels to class indices (0 to len(text_queries)-1)
        # Check if results["labels"] is a tensor; if not, assume it's a list (likely strings or tokens).
        # Newer transformers versions might return list of strings if configured.
        labels = results["labels"]
        
        if isinstance(labels, list):
            # Probably text labels. We need to map them back to indices into text_queries.
            # This can be fuzzy if tokenization changed things.
            # Simple approach: try to find the label string in text_queries.
            label_indices = []
            for lbl in labels:
                try:
                    # Clean up dots if present
                    lbl_clean = lbl.strip().rstrip('.')
                    # Try exact match first
                    if lbl_clean in text_queries:
                        label_indices.append(text_queries.index(lbl_clean))
                    else:
                        # Fallback: substring or just 0? 
                        # Let's try matching stripped
                        found = False
                        for i, q in enumerate(text_queries):
                            if lbl_clean == q.strip().rstrip('.'):
                                label_indices.append(i)
                                found = True
                                break
                        if not found:
                             # If we can't map it, mark as -1 or 0? 0 is risky if it's not the target.
                             # If we can't find it, maybe it's not one of our classes.
                             # But GD usually returns phrases from the prompt.
                             label_indices.append(0) 
                except Exception:
                    label_indices.append(0)
            labels_tensor = torch.tensor(label_indices)
        else:
            # It's a tensor (indices), move to cpu
            labels_tensor = labels.cpu()

        return [MockResult(
            results["boxes"].cpu(),
            results["scores"].cpu(),
            labels_tensor
        )]

class OwlV2Wrapper(HuggingFaceModelWrapper):
    def _predict_internal(self, image, text_queries, **kwargs):
        # OwlV2 expects list of strings
        inputs = self.processor(text=text_queries, images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
        # processors output format
        results = self.processor.post_process_grounded_object_detection(
            outputs, threshold=kwargs.get("conf", 0.1), target_sizes=target_sizes
        )[0]
        
        return [MockResult(
            results["boxes"].cpu(),
            results["scores"].cpu(),
            results["labels"].cpu()
        )]
