"""Utilities for extracting embeddings from YOLO World and SAM3 models."""

import numpy as np
import torch
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional, Union
import torch.nn.functional as F


def _get_clip_model_yolo_world(yolo_model, device=None):
    """
    Get or build the CLIP text model from a YOLO World model instance.
    Caches on the inner model so it's only loaded once.
    """
    inner = yolo_model.model
    if device is None:
        device = next(inner.model.parameters()).device
    
    # Use __dict__ to check/set to avoid triggering __getattr__
    if "clip_model" not in inner.__dict__ or inner.__dict__["clip_model"] is None:
        from ultralytics.nn.text_model import build_text_model
        inner.__dict__["clip_model"] = build_text_model("clip:ViT-B/32", device=device)
    
    return inner.__dict__["clip_model"]


def encode_texts_yolo_world(yolo_model, texts: List[str]) -> torch.Tensor:
    """
    Encode text prompts using YOLO World's CLIP text encoder.
    
    Args:
        yolo_model: YOLOWorldModel instance
        texts: List of text prompts
        
    Returns:
        Tensor of shape (len(texts), 512) with normalized embeddings
    """
    clip_model = _get_clip_model_yolo_world(yolo_model)
    device = next(clip_model.model.parameters()).device
    tokens = clip_model.tokenize(texts)
    tokens = tokens.to(device)
    
    with torch.no_grad():
        text_features = clip_model.encode_text(tokens)
        # Normalize
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    return text_features  # (N, 512)


def encode_image_crop_yolo_world(
    yolo_model,
    image: Image.Image,
    box_xyxy: Tuple[float, float, float, float],
    padding: float = 0.0,
) -> torch.Tensor:
    """
    Encode an image crop using YOLO World's CLIP vision encoder.
    
    Args:
        yolo_model: YOLOWorldModel instance
        image: PIL Image
        box_xyxy: Bounding box (x1, y1, x2, y2) in image coordinates
        padding: Padding factor (0.0 = tight crop, 0.2 = 20% padding)
        
    Returns:
        Tensor of shape (512,) with normalized embedding
    """
    # Get CLIP model
    clip_model = _get_clip_model_yolo_world(yolo_model)
    device = next(clip_model.model.parameters()).device
    
    # Extract crop with padding
    x1, y1, x2, y2 = box_xyxy
    w, h = image.size
    
    # Apply padding
    crop_w = x2 - x1
    crop_h = y2 - y1
    pad_w = crop_w * padding
    pad_h = crop_h * padding
    
    x1_pad = max(0, x1 - pad_w)
    y1_pad = max(0, y1 - pad_h)
    x2_pad = min(w, x2 + pad_w)
    y2_pad = min(h, y2 + pad_h)
    
    crop = image.crop((x1_pad, y1_pad, x2_pad, y2_pad))
    
    # Preprocess and encode
    crop_tensor = clip_model.image_preprocess(crop).unsqueeze(0).to(device)
    
    with torch.no_grad():
        image_features = clip_model.encode_image(crop_tensor)
        # Normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    return image_features.squeeze(0)  # (512,)


def encode_image_crops_batch_yolo_world(
    yolo_model,
    image: Image.Image,
    boxes_xyxy: List[Tuple[float, float, float, float]],
    padding: float = 0.0,
    batch_size: int = 15,  # Conservative default based on real-world usage with ~24GB VRAM
) -> torch.Tensor:
    """
    Encode multiple image crops in batches using YOLO World's CLIP vision encoder.
    
    This is more efficient than encoding crops one at a time, with proper batching
    to avoid memory issues.
    
    Args:
        yolo_model: YOLOWorldModel instance
        image: PIL Image
        boxes_xyxy: List of bounding boxes (x1, y1, x2, y2) in image coordinates
        padding: Padding factor (0.0 = tight crop, 0.2 = 20% padding)
        batch_size: Batch size for processing (default: 15 for YOLO World)
        
    Returns:
        Tensor of shape (len(boxes_xyxy), 512) with normalized embeddings
    """
    if len(boxes_xyxy) == 0:
        return torch.empty((0, 512))
    
    # Get CLIP model
    clip_model = _get_clip_model_yolo_world(yolo_model)
    device = next(clip_model.model.parameters()).device
    
    # Extract all crops
    w, h = image.size
    crops = []
    for x1, y1, x2, y2 in boxes_xyxy:
        # Apply padding
        crop_w = x2 - x1
        crop_h = y2 - y1
        pad_w = crop_w * padding
        pad_h = crop_h * padding
        
        x1_pad = max(0, x1 - pad_w)
        y1_pad = max(0, y1 - pad_h)
        x2_pad = min(w, x2 + pad_w)
        y2_pad = min(h, y2 + pad_h)
        
        crop = image.crop((x1_pad, y1_pad, x2_pad, y2_pad))
        crops.append(crop)
    
    # Preprocess all crops
    crop_tensors = []
    for crop in crops:
        crop_tensor = clip_model.image_preprocess(crop)
        crop_tensors.append(crop_tensor)
    
    # Process in batches
    all_features = []
    for i in range(0, len(crop_tensors), batch_size):
        batch_tensors = crop_tensors[i:i + batch_size]
        batch_tensor = torch.stack(batch_tensors).to(device)
        
        with torch.no_grad():
            batch_features = clip_model.encode_image(batch_tensor)
            # Normalize
            batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
        
        all_features.append(batch_features.cpu())
    
    # Concatenate all batches
    image_features = torch.cat(all_features, dim=0).to(device)
    
    return image_features  # (N, 512)


def encode_texts_sam3(sam3_model, texts: List[str]) -> torch.Tensor:
    """
    Encode text prompts using SAM3's CLIP text encoder.
    
    Args:
        sam3_model: SAM3Model instance
        texts: List of text prompts
        
    Returns:
        Tensor of shape (len(texts), 512) with normalized embeddings
    """
    tokenizer = sam3_model.processor.tokenizer
    text_encoder = sam3_model.model.text_encoder
    device = sam3_model.device
    
    # Tokenize
    tokens = tokenizer(texts, padding=True, return_tensors='pt').to(device)
    
    with torch.no_grad():
        text_outputs = text_encoder(**tokens)
        # Get text embeddings (try text_embeds first, fallback to pooler_output)
        if hasattr(text_outputs, 'text_embeds'):
            text_features = text_outputs.text_embeds
        elif hasattr(text_outputs, 'pooler_output'):
            text_features = text_outputs.pooler_output
        else:
            # Fallback: use last_hidden_state mean pooling
            text_features = text_outputs.last_hidden_state.mean(dim=1)
        
        # Normalize
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    return text_features  # (N, 512)


def encode_image_crop_sam3(
    sam3_model,
    image: Image.Image,
    box_xyxy: Tuple[float, float, float, float],
    padding: float = 0.0,
) -> torch.Tensor:
    """
    Encode an image crop using SAM3's vision encoder.
    
    NOTE: SAM3 vision encoder outputs patch embeddings (1024-d), but text embeddings are 512-d.
    We use mean pooling + normalization. For proper alignment, we'd need a projection layer,
    but this is a reasonable approximation for now.
    
    Args:
        sam3_model: SAM3Model instance
        image: PIL Image
        box_xyxy: Bounding box (x1, y1, x2, y2) in image coordinates
        padding: Padding factor (0.0 = tight crop, 0.2 = 20% padding)
        
    Returns:
        Tensor of shape (1024,) with normalized embedding (NOTE: dimension mismatch with text!)
    """
    vision_encoder = sam3_model.model.vision_encoder
    image_processor = sam3_model.processor.image_processor
    device = sam3_model.device
    
    # Extract crop with padding
    x1, y1, x2, y2 = box_xyxy
    w, h = image.size
    
    # Apply padding
    crop_w = x2 - x1
    crop_h = y2 - y1
    pad_w = crop_w * padding
    pad_h = crop_h * padding
    
    x1_pad = max(0, x1 - pad_w)
    y1_pad = max(0, y1 - pad_h)
    x2_pad = min(w, x2 + pad_w)
    y2_pad = min(h, y2 + pad_h)
    
    crop = image.crop((x1_pad, y1_pad, x2_pad, y2_pad))
    
    # Preprocess
    image_inputs = image_processor(crop, return_tensors='pt').to(device)
    
    with torch.no_grad():
        vision_outputs = vision_encoder(**image_inputs)
        # Get patch embeddings
        patch_embeddings = vision_outputs.last_hidden_state  # (1, num_patches, 1024)
        
        # Mean pool over patches
        image_features = patch_embeddings.mean(dim=1)  # (1, 1024)
        
        # Normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    return image_features.squeeze(0)  # (1024,)


def compute_semantic_margin(
    image_embedding: torch.Tensor,
    target_text_embeddings: torch.Tensor,
    negative_text_embeddings: torch.Tensor,
) -> float:
    """
    Compute semantic margin: max(target similarity) - max(negative similarity).
    
    Args:
        image_embedding: Image embedding (512-d for YOLO World, 1024-d for SAM3)
        target_text_embeddings: Target text embeddings (N, 512)
        negative_text_embeddings: Negative text embeddings (M, 512)
        
    Returns:
        Semantic margin (float)
    """
    # Ensure all tensors are on the same device
    device = image_embedding.device
    target_text_embeddings = target_text_embeddings.to(device)
    negative_text_embeddings = negative_text_embeddings.to(device)
    
    # Handle dimension mismatch for SAM3
    # For YOLO World: both are 512-d, can use cosine similarity directly
    # For SAM3: dimension mismatch (1024 vs 512), use L2 distance as proxy
    
    img_dim = image_embedding.shape[0]
    text_dim = target_text_embeddings.shape[1]
    
    if img_dim == text_dim:
        # Same dimension (YOLO World case) - use cosine similarity
        target_sims = torch.mm(image_embedding.unsqueeze(0), target_text_embeddings.t()).squeeze(0)
        negative_sims = torch.mm(image_embedding.unsqueeze(0), negative_text_embeddings.t()).squeeze(0)
    else:
        # Dimension mismatch (SAM3 case) - need projection or alternative approach
        # For now, we'll need to handle this differently
        # TODO: Find SAM3 vision projection to 512-d or implement alternative similarity
        raise ValueError(
            f"Dimension mismatch: image embedding is {img_dim}-d, text embeddings are {text_dim}-d. "
            f"SAM3 requires a projection layer to align dimensions. "
            f"See approach_phase5.md for details on this critical checkpoint."
        )
    
    max_target_sim = target_sims.max().item()
    max_negative_sim = negative_sims.max().item()
    
    margin = max_target_sim - max_negative_sim
    
    return margin


def compute_semantic_margins_batch(
    image_embeddings: torch.Tensor,
    target_text_embeddings: torch.Tensor,
    negative_text_embeddings: torch.Tensor,
) -> torch.Tensor:
    """
    Compute semantic margins for a batch of image embeddings.
    
    Args:
        image_embeddings: Image embeddings (N, 512) for YOLO World
        target_text_embeddings: Target text embeddings (M, 512)
        negative_text_embeddings: Negative text embeddings (K, 512)
        
    Returns:
        Tensor of margins (N,)
    """
    # Ensure all tensors are on the same device
    device = image_embeddings.device
    target_text_embeddings = target_text_embeddings.to(device)
    negative_text_embeddings = negative_text_embeddings.to(device)
    
    # Compute similarities for all images at once
    # image_embeddings: (N, 512)
    # target_text_embeddings: (M, 512)
    # negative_text_embeddings: (K, 512)
    
    target_sims = torch.mm(image_embeddings, target_text_embeddings.t())  # (N, M)
    negative_sims = torch.mm(image_embeddings, negative_text_embeddings.t())  # (N, K)
    
    # Get max similarity for each image
    max_target_sims = target_sims.max(dim=1)[0]  # (N,)
    max_negative_sims = negative_sims.max(dim=1)[0]  # (N,)
    
    # Compute margins
    margins = max_target_sims - max_negative_sims
    
    return margins  # (N,)


def compute_multi_scale_margin(
    model,
    image: Image.Image,
    box_xyxy: Tuple[float, float, float, float],
    target_text_embeddings: torch.Tensor,
    negative_text_embeddings: torch.Tensor,
    model_type: str = "yolo_world",
    scales: List[float] = [0.0, 0.2],
) -> float:
    """
    Compute semantic margin across multiple crop scales, return maximum.
    
    Args:
        model: Model instance (YOLOWorldModel or SAM3Model)
        image: PIL Image
        box_xyxy: Bounding box (x1, y1, x2, y2)
        target_text_embeddings: Target text embeddings
        negative_text_embeddings: Negative text embeddings
        model_type: "yolo_world" or "sam3"
        scales: List of padding factors (0.0 = tight, 0.2 = 20% padding)
        
    Returns:
        Maximum margin across scales
    """
    margins = []
    
    for scale in scales:
        if model_type == "yolo_world":
            img_emb = encode_image_crop_yolo_world(model, image, box_xyxy, padding=scale)
        else:  # sam3
            img_emb = encode_image_crop_sam3(model, image, box_xyxy, padding=scale)
        
        margin = compute_semantic_margin(
            img_emb,
            target_text_embeddings,
            negative_text_embeddings,
        )
        margins.append(margin)
    
    return max(margins)


def compute_multi_scale_margins_batch(
    model,
    image: Image.Image,
    boxes_xyxy: List[Tuple[float, float, float, float]],
    target_text_embeddings: torch.Tensor,
    negative_text_embeddings: torch.Tensor,
    model_type: str = "yolo_world",
    scales: List[float] = [0.0, 0.2],
    batch_size: int = None,
) -> List[float]:
    """
    Compute semantic margins for multiple boxes across multiple scales (batched).
    
    This is more efficient than calling compute_multi_scale_margin for each box.
    
    Args:
        model: Model instance (YOLOWorldModel or SAM3Model)
        image: PIL Image
        boxes_xyxy: List of bounding boxes
        target_text_embeddings: Target text embeddings
        negative_text_embeddings: Negative text embeddings
        model_type: "yolo_world" or "sam3"
        scales: List of padding factors
        batch_size: Batch size for processing (None = auto: 15 for YOLO World, 6 for SAM3)
        
    Returns:
        List of maximum margins (one per box)
    """
    if len(boxes_xyxy) == 0:
        return []
    
    # Set default batch sizes (conservative defaults for ~24GB VRAM)
    # These are lower than theoretical maximums to account for:
    # - Multiple models loaded simultaneously
    # - Text embeddings in memory
    # - Background sampling overhead
    if batch_size is None:
        batch_size = 15 if model_type == "yolo_world" else 6  # Conservative: can test higher with batch_size_test.py
    
    if model_type != "yolo_world":
        # For SAM3, process in batches (once dimension mismatch is fixed)
        # For now, fallback to non-batched
        return [compute_multi_scale_margin(
            model, image, box, target_text_embeddings, negative_text_embeddings, model_type, scales
        ) for box in boxes_xyxy]
    
    # For YOLO World, we can batch across scales
    all_margins = []
    
    for scale in scales:
        # Encode all crops at this scale in batches
        crop_embeddings = encode_image_crops_batch_yolo_world(
            model, image, boxes_xyxy, padding=scale, batch_size=batch_size
        )
        
        # Compute margins for all crops at once
        margins = compute_semantic_margins_batch(
            crop_embeddings,
            target_text_embeddings,
            negative_text_embeddings,
        )
        
        all_margins.append(margins.cpu().numpy())
    
    # Take max across scales for each box
    all_margins = np.array(all_margins)  # (num_scales, num_boxes)
    max_margins = all_margins.max(axis=0)  # (num_boxes,)
    
    return max_margins.tolist()
