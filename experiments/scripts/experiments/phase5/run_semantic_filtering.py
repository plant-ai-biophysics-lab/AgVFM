#!/usr/bin/env python3
"""
Phase 5: Semantic Margin + Per-Image P-Values + BH-FDR Filtering

This script implements the full ChatGPT5.2 pipeline:
1. Candidate generation (low conf threshold)
2. Semantic margin computation (target vs hard negatives)
3. Per-image background sampling and p-value computation
4. Benjamini-Hochberg FDR selection
5. Post-processing (dedup/NMS)

Currently supports YOLO World. SAM3 support pending dimension alignment.
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse

import numpy as np
import torch
from PIL import Image

# Add project root to path
project_root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(project_root))

from agvfm.data.labels import get_image_paths, load_ground_truth
from agvfm.models.yolo_world import YOLOWorldModel
from agvfm.analysis.embedding_utils import (
    encode_texts_yolo_world,
    encode_image_crop_yolo_world,
    compute_semantic_margin,
    compute_multi_scale_margin,
    compute_multi_scale_margins_batch,
)


# Cowpea prompt banks (from Phase 5 plan)
TARGET_PROMPTS = [
    "cowpea flower",
    "yellow cowpea flower",
    "cowpea flower with open petals",
    "a single yellow cowpea flower",
    "vigna unguiculata flower",
]

HARD_NEGATIVE_PROMPTS = [
    "cowpea bud",
    "flower bud",
    "cowpea calyx",
    "green calyx",
    "cowpea leaf",
    "leaf",
    "cowpea stem",
    "stem",
    "shoot",
    "cowpea pod",
    "young pod",
    "soil",
    "ground",
    "shadow on leaf",
    "shadow",
    "sun glare",
    "specular highlight",
    "bright reflection",
]


def get_data_paths():
    """Get data paths using same logic as Phase 4."""
    import os
    project_root = Path(__file__).resolve().parents[5]
    
    # Check DATA_ROOT environment variable first
    data_root = os.environ.get("DATA_ROOT")
    if data_root:
        agvfm_data = Path(data_root) / "_data" / "T4_REAL_FLOWER_SOBJ.v18-train_2023_test_location_genotype"
        if agvfm_data.exists():
            images_dir = agvfm_data / "test" / "images"
            labels_dir = agvfm_data / "test" / "labels"
            return images_dir, labels_dir
    
    # Try AgVFM2 first
    agvfm2_data = project_root / "_data" / "T4_REAL_FLOWER_SOBJ.v18-train_2023_test_location_genotype"
    if agvfm2_data.exists():
        images_dir = agvfm2_data / "test" / "images"
        labels_dir = agvfm2_data / "test" / "labels"
        return images_dir, labels_dir
    
    # Try direct path
    direct_path = Path("/data2/jmearles/AgVFM2/_data/T4_REAL_FLOWER_SOBJ.v18-train_2023_test_location_genotype")
    if direct_path.exists():
        images_dir = direct_path / "test" / "images"
        labels_dir = direct_path / "test" / "labels"
        return images_dir, labels_dir
    
    # Try AgVFM (sibling directory) as fallback
    agvfm_data = project_root.parent / "AgVFM" / "_data" / "T4_REAL_FLOWER_SOBJ.v18-train_2023_test_location_genotype"
    if agvfm_data.exists():
        images_dir = agvfm_data / "test" / "images"
        labels_dir = agvfm_data / "test" / "labels"
        return images_dir, labels_dir
    
    raise FileNotFoundError("Could not find data directory")


def sample_background_regions(
    image: Image.Image,
    candidate_boxes: List[Tuple[float, float, float, float]],
    K: int = 128,
    min_iou: float = 0.1,
    min_size: int = 20,
) -> List[Tuple[float, float, float, float]]:
    """
    Sample K background regions that don't overlap with candidates.
    
    Args:
        image: PIL Image
        candidate_boxes: List of candidate boxes (x1, y1, x2, y2)
        K: Number of background samples
        min_iou: Minimum IoU threshold to reject overlap
        min_size: Minimum box size (width and height)
        
    Returns:
        List of background region boxes (x1, y1, x2, y2)
    """
    w, h = image.size
    bg_regions = []
    max_attempts = K * 10  # Try many times to find non-overlapping regions
    
    np.random.seed(42)  # For reproducibility
    
    for _ in range(max_attempts):
        if len(bg_regions) >= K:
            break
        
        # Sample random box
        box_w = np.random.randint(min_size, min(w // 4, 200))
        box_h = np.random.randint(min_size, min(h // 4, 200))
        x1 = np.random.randint(0, max(1, w - box_w))
        y1 = np.random.randint(0, max(1, h - box_h))
        x2 = x1 + box_w
        y2 = y1 + box_h
        
        # Check overlap with candidates
        overlaps = False
        for cand_box in candidate_boxes:
            iou = compute_iou((x1, y1, x2, y2), cand_box)
            if iou > min_iou:
                overlaps = True
                break
        
        if not overlaps:
            bg_regions.append((x1, y1, x2, y2))
    
    return bg_regions


def compute_iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    """Compute IoU between two boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    inter_area = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def compute_p_value(margin: float, bg_margins: List[float]) -> float:
    """
    Compute empirical p-value: (1 + count(bg_margins >= margin)) / (1 + K)
    
    Args:
        margin: Candidate margin
        bg_margins: Sorted list of background margins
        
    Returns:
        P-value (float)
    """
    K = len(bg_margins)
    if K == 0:
        return 1.0
    
    # Count how many background margins are >= candidate margin
    count_ge = sum(1 for bg_m in bg_margins if bg_m >= margin)
    
    # Smoothed empirical p-value
    p_value = (1 + count_ge) / (1 + K)
    
    return p_value


def benjamini_hochberg_fdr(p_values: List[float], q: float = 0.10) -> List[int]:
    """
    Apply Benjamini-Hochberg FDR control.
    
    Args:
        p_values: List of p-values
        q: FDR level (default 0.10)
        
    Returns:
        List of indices to keep
    """
    if not p_values:
        return []
    
    M = len(p_values)
    
    # Sort p-values with indices
    indexed_pvals = [(i, p) for i, p in enumerate(p_values)]
    indexed_pvals.sort(key=lambda x: x[1])
    
    # Find largest k such that p_(k) <= (k/M) * q
    keep_indices = []
    for k in range(1, M + 1):
        p_k = indexed_pvals[k - 1][1]
        threshold = (k / M) * q
        
        if p_k <= threshold:
            keep_indices.append(indexed_pvals[k - 1][0])
        else:
            break
    
    return sorted(keep_indices)


def fixed_p_cutoff(p_values: List[float], p_threshold: float = 0.05) -> List[int]:
    """
    Apply fixed p-value cutoff (simpler than BH-FDR).
    
    Args:
        p_values: List of p-values
        p_threshold: P-value threshold (default 0.05)
        
    Returns:
        List of indices to keep
    """
    return [i for i, p in enumerate(p_values) if p <= p_threshold]


def top_k_selection(
    margins: List[float],
    p_values: List[float],
    k: int = 5,
    p_max: Optional[float] = None,
) -> List[int]:
    """
    Select top-k detections by margin (or p-value), optionally conditioned on p-value.
    
    Args:
        margins: List of semantic margins
        p_values: List of p-values
        k: Number of top detections to keep (default 5)
        p_max: Optional maximum p-value threshold (e.g., 0.2)
        
    Returns:
        List of indices to keep
    """
    if not margins:
        return []
    
    # Create indexed list with (index, margin, p_value)
    indexed = [(i, m, p) for i, (m, p) in enumerate(zip(margins, p_values))]
    
    # Filter by p_max if specified
    if p_max is not None:
        indexed = [(i, m, p) for i, m, p in indexed if p <= p_max]
    
    # Sort by margin (descending) and take top k
    indexed.sort(key=lambda x: x[1], reverse=True)
    keep_indices = [i for i, _, _ in indexed[:k]]
    
    return sorted(keep_indices)


def global_benjamini_hochberg_fdr(
    all_p_values: List[List[float]],
    q: float = 0.10,
) -> List[List[int]]:
    """
    Apply Benjamini-Hochberg FDR control globally across all images.
    
    Args:
        all_p_values: List of p-value lists (one per image)
        q: FDR level (default 0.10)
        
    Returns:
        List of keep_indices lists (one per image)
    """
    # Flatten all p-values with image index
    indexed_pvals = []
    for img_idx, p_vals in enumerate(all_p_values):
        for det_idx, p_val in enumerate(p_vals):
            indexed_pvals.append((img_idx, det_idx, p_val))
    
    if not indexed_pvals:
        return [[] for _ in all_p_values]
    
    M = len(indexed_pvals)
    
    # Sort by p-value
    indexed_pvals.sort(key=lambda x: x[2])
    
    # Find largest k such that p_(k) <= (k/M) * q
    keep_pairs = []
    for k in range(1, M + 1):
        p_k = indexed_pvals[k - 1][2]
        threshold = (k / M) * q
        
        if p_k <= threshold:
            keep_pairs.append((indexed_pvals[k - 1][0], indexed_pvals[k - 1][1]))
        else:
            break
    
    # Group by image
    keep_by_image = {}
    for img_idx, det_idx in keep_pairs:
        if img_idx not in keep_by_image:
            keep_by_image[img_idx] = []
        keep_by_image[img_idx].append(det_idx)
    
    # Return sorted indices per image
    result = []
    for img_idx in range(len(all_p_values)):
        if img_idx in keep_by_image:
            result.append(sorted(keep_by_image[img_idx]))
        else:
            result.append([])
    
    return result


def process_image_semantic_filtering(
    model: YOLOWorldModel,
    image_path: Path,
    target_text_embeddings: torch.Tensor,
    negative_text_embeddings: torch.Tensor,
    conf_floor: float = 0.01,
    K_bg: int = 128,
    q_fdr: float = 0.10,
    scales: List[float] = [0.0, 0.2],
    min_candidates_for_bg: int = 10,
    selection_method: str = "bh_fdr",
    p_cutoff: float = 0.05,
    top_k: int = 5,
    top_k_p_max: Optional[float] = None,
) -> Dict:
    """
    Process a single image through the semantic filtering pipeline.
    
    Args:
        model: YOLOWorldModel instance
        image_path: Path to image
        target_text_embeddings: Target text embeddings (N, 512)
        negative_text_embeddings: Negative text embeddings (M, 512)
        conf_floor: Low confidence threshold for candidate generation
        K_bg: Number of background samples
        q_fdr: FDR level for Benjamini-Hochberg
        scales: Crop scales for multi-scale margin
        min_candidates_for_bg: Skip background sampling if fewer candidates
        
    Returns:
        Dictionary with filtered detections and metadata
    """
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Step A: Candidate generation (low conf threshold)
    boxes, confidences = model.predict(
        image_path=image_path,
        prompt="cowpea flower",  # Use base prompt for candidate generation
        conf_threshold=conf_floor,
    )
    
    if len(boxes) == 0:
        return {
            "image_path": str(image_path),
            "num_candidates": 0,
            "num_kept": 0,
            "kept_boxes": np.zeros((0, 4)),
            "kept_confidences": np.zeros(0),
            "margins": [],
            "p_values": [],
        }
    
    # Convert boxes to list of tuples
    candidate_boxes = [tuple(box) for box in boxes]
    
    # Step B: Compute semantic margins for candidates (BATCHED for efficiency)
    candidate_margins = compute_multi_scale_margins_batch(
        model=model,
        image=image,
        boxes_xyxy=candidate_boxes,
        target_text_embeddings=target_text_embeddings,
        negative_text_embeddings=negative_text_embeddings,
        model_type="yolo_world",
        scales=scales,
    )
    
    # Step C: Background sampling and p-values
    if len(candidate_boxes) >= min_candidates_for_bg:
        bg_regions = sample_background_regions(image, candidate_boxes, K=K_bg)
        
        # Compute background margins (BATCHED for efficiency)
        bg_margins = compute_multi_scale_margins_batch(
            model=model,
            image=image,
            boxes_xyxy=bg_regions,
            target_text_embeddings=target_text_embeddings,
            negative_text_embeddings=negative_text_embeddings,
            model_type="yolo_world",
            scales=scales,
        )
        
        # Sort background margins for efficient p-value computation
        bg_margins_sorted = sorted(bg_margins, reverse=True)
        
        # Compute p-values
        p_values = [compute_p_value(m, bg_margins_sorted) for m in candidate_margins]
    else:
        # Skip background sampling for images with few candidates
        # Use uniform p-values (will result in no detections kept)
        p_values = [1.0] * len(candidate_margins)
        bg_margins_sorted = []
    
    # Step D: Selection method (note: global_bh is handled in main() after collecting all p-values)
    if selection_method == "bh_fdr":
        keep_indices = benjamini_hochberg_fdr(p_values, q=q_fdr)
    elif selection_method == "fixed_p":
        keep_indices = fixed_p_cutoff(p_values, p_threshold=p_cutoff)
    elif selection_method == "top_k":
        keep_indices = top_k_selection(candidate_margins, p_values, k=top_k, p_max=top_k_p_max)
    elif selection_method == "global_bh":
        # For global BH, we'll return all candidates and filter in main()
        # Return all indices here, filtering happens after collecting all p-values
        keep_indices = list(range(len(p_values)))
    else:
        raise ValueError(f"Unknown selection method: {selection_method}")
    
    # Extract kept detections
    kept_boxes = boxes[keep_indices] if len(keep_indices) > 0 else np.zeros((0, 4))
    kept_confidences = confidences[keep_indices] if len(keep_indices) > 0 else np.zeros(0)
    
    result = {
        "image_path": str(image_path),
        "num_candidates": len(boxes),
        "num_kept": len(keep_indices),
        "kept_boxes": kept_boxes,
        "kept_confidences": kept_confidences,
        "margins": candidate_margins,
        "p_values": p_values,
        "bg_margins": bg_margins_sorted if len(candidate_boxes) >= min_candidates_for_bg else [],
    }
    
    # Store original candidates for global BH
    if selection_method == "global_bh":
        result["candidate_boxes"] = boxes.tolist() if isinstance(boxes, np.ndarray) else [list(b) for b in boxes]
        result["candidate_confidences"] = confidences.tolist() if isinstance(confidences, np.ndarray) else list(confidences)
    
    return result


def main():
    """Main function to run Phase 5 semantic filtering."""
    parser = argparse.ArgumentParser(
        description="Phase 5: Semantic Margin + BH-FDR Filtering"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["yolo_world"],
        default="yolo_world",
        help="Model to use (SAM3 pending dimension fix)",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=None,
        help="Number of images to process (None = all)",
    )
    parser.add_argument(
        "--conf-floor",
        type=float,
        default=0.01,
        help="Low confidence threshold for candidate generation",
    )
    parser.add_argument(
        "--K-bg",
        type=int,
        default=128,
        help="Number of background samples per image",
    )
    parser.add_argument(
        "--q-fdr",
        type=float,
        default=0.10,
        help="FDR level for Benjamini-Hochberg (default: 0.10)",
    )
    parser.add_argument(
        "--selection-method",
        type=str,
        choices=["bh_fdr", "fixed_p", "top_k", "global_bh"],
        default="bh_fdr",
        help="Selection method: 'bh_fdr' (per-image BH-FDR), 'fixed_p' (fixed p cutoff), 'top_k' (top-k by margin), 'global_bh' (global BH-FDR)",
    )
    parser.add_argument(
        "--p-cutoff",
        type=float,
        default=0.05,
        help="P-value cutoff for fixed_p method (default: 0.05)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top detections for top_k method (default: 5)",
    )
    parser.add_argument(
        "--top-k-p-max",
        type=float,
        default=None,
        help="Optional maximum p-value for top_k method (e.g., 0.2)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results/phase5_semantic_filtering"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--results-filename",
        type=str,
        default="semantic_filtering_results.json",
        help="Filename for results JSON (default: semantic_filtering_results.json)",
    )
    parser.add_argument(
        "--results-subdir",
        type=str,
        default="results",
        help="Subdirectory within output-dir for JSON files (default: results)",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Test mode: process only 5 images to verify pipeline",
    )
    
    args = parser.parse_args()
    
    # Get data paths
    images_dir, labels_dir = get_data_paths()
    
    # Get image paths
    all_image_paths = get_image_paths(images_dir)
    if args.test_mode:
        image_paths = all_image_paths[:5]
        print("🧪 TEST MODE: Processing only 5 images")
    elif args.num_images:
        image_paths = all_image_paths[:args.num_images]
    else:
        image_paths = all_image_paths
    
    print(f"📊 Processing {len(image_paths)} images", flush=True)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"\n🔧 Loading {args.model} model...")
    start = time.time()
    if args.model == "yolo_world":
        model = YOLOWorldModel()
    else:
        raise ValueError(f"Model {args.model} not yet supported (SAM3 pending dimension fix)")
    load_time = time.time() - start
    print(f"   Model loaded in {load_time:.2f}s")
    
    # Encode text prompts (cache these)
    print(f"\n📝 Encoding text prompts...")
    start = time.time()
    target_embeddings = encode_texts_yolo_world(model, TARGET_PROMPTS)
    negative_embeddings = encode_texts_yolo_world(model, HARD_NEGATIVE_PROMPTS)
    text_time = time.time() - start
    print(f"   Text encoding: {text_time:.2f}s")
    print(f"   Target prompts: {len(TARGET_PROMPTS)}")
    print(f"   Hard negatives: {len(HARD_NEGATIVE_PROMPTS)}")
    
    # Process images
    print(f"\n🔄 Processing images...")
    if args.selection_method == "bh_fdr":
        print(f"   Config: conf_floor={args.conf_floor}, K_bg={args.K_bg}, q_fdr={args.q_fdr}, method={args.selection_method}")
    elif args.selection_method == "fixed_p":
        print(f"   Config: conf_floor={args.conf_floor}, K_bg={args.K_bg}, p_cutoff={args.p_cutoff}, method={args.selection_method}")
    elif args.selection_method == "top_k":
        print(f"   Config: conf_floor={args.conf_floor}, K_bg={args.K_bg}, top_k={args.top_k}, p_max={args.top_k_p_max}, method={args.selection_method}")
    elif args.selection_method == "global_bh":
        print(f"   Config: conf_floor={args.conf_floor}, K_bg={args.K_bg}, q_fdr={args.q_fdr}, method={args.selection_method}")
    
    all_results = []
    total_candidates = 0
    total_kept = 0
    
    process_start = time.time()
    all_p_values_for_global = []  # For global BH
    
    for img_idx, img_path in enumerate(image_paths):
        img_start = time.time()
        
        result = process_image_semantic_filtering(
            model=model,
            image_path=img_path,
            target_text_embeddings=target_embeddings,
            negative_text_embeddings=negative_embeddings,
            conf_floor=args.conf_floor,
            K_bg=args.K_bg,
            q_fdr=args.q_fdr,
            selection_method=args.selection_method,
            p_cutoff=args.p_cutoff,
            top_k=args.top_k,
            top_k_p_max=args.top_k_p_max,
        )
        
        # Store p-values for global BH if needed
        if args.selection_method == "global_bh":
            all_p_values_for_global.append(result["p_values"])
        
        all_results.append(result)
        total_candidates += result["num_candidates"]
        total_kept += result["num_kept"]
        
        img_time = time.time() - img_start
        
        if (img_idx + 1) % 10 == 0 or args.test_mode or (img_idx + 1) % 5 == 0:
            print(f"   [{img_idx + 1}/{len(image_paths)}] {img_path.name}: "
                  f"{result['num_candidates']} candidates → {result['num_kept']} kept "
                  f"({img_time:.1f}s)", flush=True)
    
    # Apply global BH if requested
    if args.selection_method == "global_bh":
        print(f"\n🌐 Applying global BH-FDR (q={args.q_fdr}) across all {total_candidates} candidates...")
        global_keep_indices = global_benjamini_hochberg_fdr(all_p_values_for_global, q=args.q_fdr)
        
        # Update results with global BH filtering
        total_kept = 0
        for img_idx, keep_indices in enumerate(global_keep_indices):
            result = all_results[img_idx]
            
            # Get original candidates (stored for global_bh method)
            if "candidate_boxes" in result:
                original_boxes = np.array(result["candidate_boxes"])
                original_confidences = np.array(result["candidate_confidences"])
            else:
                # Fallback: use kept_boxes (shouldn't happen)
                original_boxes = np.array(result["kept_boxes"]) if len(result["kept_boxes"]) > 0 else np.zeros((0, 4))
                original_confidences = np.array(result["kept_confidences"]) if len(result["kept_confidences"]) > 0 else np.zeros(0)
            
            if len(original_boxes) > 0 and len(keep_indices) > 0:
                kept_boxes = original_boxes[keep_indices]
                kept_confidences = original_confidences[keep_indices] if len(original_confidences) > 0 else np.zeros(len(keep_indices))
            else:
                kept_boxes = np.zeros((0, 4))
                kept_confidences = np.zeros(0)
            
            all_results[img_idx]["num_kept"] = len(keep_indices)
            all_results[img_idx]["kept_boxes"] = kept_boxes
            all_results[img_idx]["kept_confidences"] = kept_confidences
            total_kept += len(keep_indices)
    
    total_time = time.time() - process_start
    
    print(f"\n✅ Processing complete!")
    print(f"   Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"   Time per image: {total_time/len(image_paths):.2f}s")
    print(f"   Total candidates: {total_candidates}")
    print(f"   Total kept: {total_kept}")
    print(f"   Average kept per image: {total_kept/len(image_paths):.2f}")
    
    # Save results in subdirectory
    results_dir = args.output_dir / args.results_subdir
    results_dir.mkdir(parents=True, exist_ok=True)
    output_file = results_dir / args.results_filename
    with open(output_file, "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = []
        for result in all_results:
            json_result = {
                "image_path": result["image_path"],
                "num_candidates": int(result["num_candidates"]),
                "num_kept": int(result["num_kept"]),
                "kept_boxes": result["kept_boxes"].tolist() if len(result["kept_boxes"]) > 0 else [],
                "kept_confidences": result["kept_confidences"].tolist() if len(result["kept_confidences"]) > 0 else [],
                "margins": [float(m) for m in result["margins"]],
                "p_values": [float(p) for p in result["p_values"]],
            }
            json_results.append(json_result)
        
        config_dict = {
            "model": args.model,
            "conf_floor": args.conf_floor,
            "K_bg": args.K_bg,
            "scales": [0.0, 0.2],
            "selection_method": args.selection_method,
        }
        if args.selection_method == "bh_fdr" or args.selection_method == "global_bh":
            config_dict["q_fdr"] = args.q_fdr
        elif args.selection_method == "fixed_p":
            config_dict["p_cutoff"] = args.p_cutoff
        elif args.selection_method == "top_k":
            config_dict["top_k"] = args.top_k
            if args.top_k_p_max is not None:
                config_dict["top_k_p_max"] = args.top_k_p_max
        
        json.dump({
            "config": config_dict,
            "summary": {
                "num_images": len(image_paths),
                "total_candidates": int(total_candidates),
                "total_kept": int(total_kept),
                "avg_kept_per_image": float(total_kept / len(image_paths)),
            },
            "results": json_results,
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Print summary statistics
    print(f"\n📈 Summary Statistics:")
    num_candidates_list = [r["num_candidates"] for r in all_results]
    num_kept_list = [r["num_kept"] for r in all_results]
    print(f"   Candidates per image: mean={np.mean(num_candidates_list):.1f}, "
          f"median={np.median(num_candidates_list):.1f}, "
          f"max={np.max(num_candidates_list)}")
    print(f"   Kept per image: mean={np.mean(num_kept_list):.1f}, "
          f"median={np.median(num_kept_list):.1f}, "
          f"max={np.max(num_kept_list)}")
    
    # Count empty images (0 kept)
    empty_count = sum(1 for r in all_results if r["num_kept"] == 0)
    print(f"   Images with 0 detections: {empty_count}/{len(image_paths)} ({100*empty_count/len(image_paths):.1f}%)")


if __name__ == "__main__":
    main()
