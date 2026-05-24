"""Utility to test maximum batch sizes for GPU memory constraints."""

# Import torch first to avoid naming conflict with agvfm.utils.logging
import torch
from typing import Optional, Dict, Tuple
from pathlib import Path
from PIL import Image
import numpy as np


def test_batch_size_yolo_world(
    yolo_model,
    image_size: Tuple[int, int] = (640, 640),
    start_batch_size: int = 1,
    max_batch_size: int = 100,
    device: Optional[str] = None,
) -> Dict:
    """
    Test maximum batch size for YOLO World crop encoding.
    
    Args:
        yolo_model: YOLOWorldModel instance
        image_size: Size of test image crops (width, height)
        start_batch_size: Starting batch size to test
        max_batch_size: Maximum batch size to test (will stop before if OOM)
        device: Device to test on (None = auto-detect)
        
    Returns:
        Dictionary with:
            - max_batch_size: Maximum batch size that works
            - tested_sizes: List of tested batch sizes and their results
            - memory_info: GPU memory information
    """
    from agvfm.analysis.embedding_utils import _get_clip_model_yolo_world
    
    if device is None:
        device = next(yolo_model.model.model.parameters()).device
    
    # Convert device to index if it's a torch.device
    if isinstance(device, torch.device):
        device_index = device.index if device.index is not None else 0
    elif isinstance(device, str):
        if device.startswith('cuda:'):
            device_index = int(device.split(':')[1])
        elif device == 'cuda':
            device_index = 0
        else:
            device_index = None
    else:
        device_index = device if isinstance(device, int) else 0
    
    clip_model = _get_clip_model_yolo_world(yolo_model)
    
    # Get memory info
    if torch.cuda.is_available() and device_index is not None:
        torch.cuda.empty_cache()
        memory_total = torch.cuda.get_device_properties(device_index).total_memory / 1e9  # GB
        memory_allocated = torch.cuda.memory_allocated(device_index) / 1e9  # GB
        memory_reserved = torch.cuda.memory_reserved(device_index) / 1e9  # GB
    else:
        memory_total = memory_allocated = memory_reserved = 0.0
        device_index = None
    
    # Create dummy image crop
    dummy_crop = Image.new('RGB', image_size, color='red')
    crop_tensor = clip_model.image_preprocess(dummy_crop)
    
    tested_sizes = []
    max_working_size = 0
    
    print(f"Testing YOLO World batch sizes (GPU: {memory_total:.1f}GB total VRAM)")
    print(f"Starting from batch_size={start_batch_size}, testing up to {max_batch_size}")
    print()
    
    # Binary search for maximum batch size
    low = start_batch_size
    high = max_batch_size
    best_size = start_batch_size
    
    while low <= high:
        batch_size = (low + high) // 2
        
        try:
            # Clear cache
            torch.cuda.empty_cache()
            
            # Test batch
            batch_tensors = [crop_tensor] * batch_size
            batch_input = torch.stack(batch_tensors).to(device)
            
            with torch.no_grad():
                _ = clip_model.encode_image(batch_input)
            
            # Success - this batch size works
            tested_sizes.append((batch_size, True, None))
            best_size = batch_size
            max_working_size = batch_size
            print(f"  ✅ batch_size={batch_size}: SUCCESS")
            
            # Try larger batch size
            low = batch_size + 1
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "CUDA out of memory" in str(e):
                # OOM - this batch size is too large
                tested_sizes.append((batch_size, False, "OOM"))
                print(f"  ❌ batch_size={batch_size}: OOM")
                high = batch_size - 1
            else:
                # Other error
                tested_sizes.append((batch_size, False, str(e)))
                print(f"  ❌ batch_size={batch_size}: ERROR - {e}")
                high = batch_size - 1
    
    # Clear cache
    torch.cuda.empty_cache()
    
    print()
    print(f"Maximum working batch size: {max_working_size}")
    
    return {
        "max_batch_size": max_working_size,
        "best_batch_size": best_size,
        "tested_sizes": tested_sizes,
        "memory_info": {
            "total_gb": memory_total,
            "allocated_gb": memory_allocated,
            "reserved_gb": memory_reserved,
        },
    }


def test_batch_size_sam3(
    sam3_model,
    image_size: Tuple[int, int] = (640, 640),
    start_batch_size: int = 1,
    max_batch_size: int = 50,
) -> Dict:
    """
    Test maximum batch size for SAM3 vision encoding.
    
    Args:
        sam3_model: SAM3Model instance
        image_size: Size of test image crops (width, height)
        start_batch_size: Starting batch size to test
        max_batch_size: Maximum batch size to test (will stop before if OOM)
        
    Returns:
        Dictionary with:
            - max_batch_size: Maximum batch size that works
            - tested_sizes: List of tested batch sizes and their results
            - memory_info: GPU memory information
    """
    device = sam3_model.device
    vision_encoder = sam3_model.model.vision_encoder
    image_processor = sam3_model.processor.image_processor
    
    # Convert device to index
    if isinstance(device, torch.device):
        device_index = device.index if device.index is not None else 0
    elif isinstance(device, str):
        if device.startswith('cuda:'):
            device_index = int(device.split(':')[1])
        elif device == 'cuda':
            device_index = 0
        else:
            device_index = None
    else:
        device_index = device if isinstance(device, int) else 0
    
    # Get memory info
    if torch.cuda.is_available() and device_index is not None:
        torch.cuda.empty_cache()
        memory_total = torch.cuda.get_device_properties(device_index).total_memory / 1e9  # GB
        memory_allocated = torch.cuda.memory_allocated(device_index) / 1e9  # GB
        memory_reserved = torch.cuda.memory_reserved(device_index) / 1e9  # GB
    else:
        memory_total = memory_allocated = memory_reserved = 0.0
        device_index = None
    
    # Create dummy image crop
    dummy_crop = Image.new('RGB', image_size, color='red')
    
    tested_sizes = []
    max_working_size = 0
    
    print(f"Testing SAM3 batch sizes (GPU: {memory_total:.1f}GB total VRAM)")
    print(f"Starting from batch_size={start_batch_size}, testing up to {max_batch_size}")
    print()
    
    # Binary search for maximum batch size
    low = start_batch_size
    high = max_batch_size
    best_size = start_batch_size
    
    while low <= high:
        batch_size = (low + high) // 2
        
        try:
            # Clear cache
            torch.cuda.empty_cache()
            
            # Test batch
            batch_images = [dummy_crop] * batch_size
            batch_inputs = image_processor(batch_images, return_tensors='pt').to(device)
            
            with torch.no_grad():
                outputs = vision_encoder(**batch_inputs)
                # Use the output to ensure computation happens
                _ = outputs.last_hidden_state.mean()
            
            # Success - this batch size works
            tested_sizes.append((batch_size, True, None))
            best_size = batch_size
            max_working_size = batch_size
            print(f"  ✅ batch_size={batch_size}: SUCCESS")
            
            # Try larger batch size
            low = batch_size + 1
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "CUDA out of memory" in str(e):
                # OOM - this batch size is too large
                tested_sizes.append((batch_size, False, "OOM"))
                print(f"  ❌ batch_size={batch_size}: OOM")
                high = batch_size - 1
            else:
                # Other error
                tested_sizes.append((batch_size, False, str(e)))
                print(f"  ❌ batch_size={batch_size}: ERROR - {e}")
                high = batch_size - 1
    
    # Clear cache
    torch.cuda.empty_cache()
    
    print()
    print(f"Maximum working batch size: {max_working_size}")
    
    return {
        "max_batch_size": max_working_size,
        "best_batch_size": best_size,
        "tested_sizes": tested_sizes,
        "memory_info": {
            "total_gb": memory_total,
            "allocated_gb": memory_allocated,
            "reserved_gb": memory_reserved,
        },
    }


def find_optimal_batch_sizes(
    yolo_model=None,
    sam3_model=None,
    image_size: Tuple[int, int] = (640, 640),
    safety_margin: float = 0.8,
) -> Dict:
    """
    Find optimal batch sizes for both models with a safety margin.
    
    Args:
        yolo_model: Optional YOLOWorldModel instance (will create if None)
        sam3_model: Optional SAM3Model instance (will create if None)
        image_size: Size of test image crops (width, height)
        safety_margin: Fraction of max batch size to use as safe default (0.8 = 80%)
        
    Returns:
        Dictionary with optimal batch sizes and test results
    """
    results = {}
    
    if yolo_model is None:
        print("Loading YOLO World model for batch size testing...")
        from agvfm.models.yolo_world import YOLOWorldModel
        yolo_model = YOLOWorldModel()
    
    if yolo_model is not None:
        print("\n" + "="*80)
        print("YOLO WORLD BATCH SIZE TEST")
        print("="*80)
        yolo_results = test_batch_size_yolo_world(
            yolo_model,
            image_size=image_size,
            start_batch_size=1,
            max_batch_size=100,
        )
        results["yolo_world"] = {
            "max_batch_size": yolo_results["max_batch_size"],
            "recommended_batch_size": int(yolo_results["max_batch_size"] * safety_margin),
            "test_results": yolo_results,
        }
    
    if sam3_model is None:
        print("\nLoading SAM3 model for batch size testing...")
        from agvfm.models.sam3 import SAM3Model
        sam3_model = SAM3Model()
    
    if sam3_model is not None:
        print("\n" + "="*80)
        print("SAM3 BATCH SIZE TEST")
        print("="*80)
        sam3_results = test_batch_size_sam3(
            sam3_model,
            image_size=image_size,
            start_batch_size=1,
            max_batch_size=50,
        )
        results["sam3"] = {
            "max_batch_size": sam3_results["max_batch_size"],
            "recommended_batch_size": int(sam3_results["max_batch_size"] * safety_margin),
            "test_results": sam3_results,
        }
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    if "yolo_world" in results:
        print(f"YOLO World:")
        print(f"  Maximum batch size: {results['yolo_world']['max_batch_size']}")
        print(f"  Recommended batch size (80% safety): {results['yolo_world']['recommended_batch_size']}")
    if "sam3" in results:
        print(f"SAM3:")
        print(f"  Maximum batch size: {results['sam3']['max_batch_size']}")
        print(f"  Recommended batch size (80% safety): {results['sam3']['recommended_batch_size']}")
    
    return results


if __name__ == "__main__":
    """Run batch size tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test maximum batch sizes for GPU memory")
    parser.add_argument(
        "--model",
        type=str,
        choices=["yolo_world", "sam3", "both"],
        default="both",
        help="Model to test",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[640, 640],
        metavar=("WIDTH", "HEIGHT"),
        help="Test image crop size (default: 640 640)",
    )
    parser.add_argument(
        "--safety-margin",
        type=float,
        default=0.8,
        help="Safety margin for recommended batch size (default: 0.8 = 80%%)",
    )
    
    args = parser.parse_args()
    
    yolo_model = None
    sam3_model = None
    
    if args.model in ["yolo_world", "both"]:
        from agvfm.models.yolo_world import YOLOWorldModel
        yolo_model = YOLOWorldModel()
    
    if args.model in ["sam3", "both"]:
        from agvfm.models.sam3 import SAM3Model
        sam3_model = SAM3Model()
    
    results = find_optimal_batch_sizes(
        yolo_model=yolo_model,
        sam3_model=sam3_model,
        image_size=tuple(args.image_size),
        safety_margin=args.safety_margin,
    )
