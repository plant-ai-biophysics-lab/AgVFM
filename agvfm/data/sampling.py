"""Dataset sampling and split utilities."""

from pathlib import Path
from typing import List, Set


def _load_dev_manifest(manifest_path: Path) -> Set[str]:
    """Load dev manifest file and return set of dev image names."""
    dev_names = set()
    manifest = Path(manifest_path)
    if manifest.exists():
        with open(manifest) as f:
            dev_names = {line.strip() for line in f if line.strip()}
    return dev_names


def get_holdout_test_paths(
    test_images_dir: Path,
    manifest_path: Path,
) -> List[Path]:
    """
    Return test images EXCLUDING the dev/tune images.
    
    This creates the clean holdout test set that was never used for prompt development.
    
    Args:
        test_images_dir: Directory containing test images
        manifest_path: Path to dev manifest file listing dev image names
        
    Returns:
        List of image paths in the holdout test set (excluding dev images)
    """
    from agvfm.data.labels import get_image_paths
    
    dev_names = _load_dev_manifest(manifest_path)
    all_test = get_image_paths(test_images_dir)
    holdout = [p for p in all_test if p.name not in dev_names]
    return holdout


def get_full_test_paths(
    test_images_dir: Path,
) -> List[Path]:
    """
    Return ALL test images (full test set, 158 images).
    
    This includes both the holdout test set and the dev images.
    Use this when you want to evaluate on the complete test set.
    
    Args:
        test_images_dir: Directory containing test images
        
    Returns:
        List of all image paths in the test set
    """
    from agvfm.data.labels import get_image_paths
    
    return get_image_paths(test_images_dir)
