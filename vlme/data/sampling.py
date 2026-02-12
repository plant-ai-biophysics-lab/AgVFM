"""Image path sampling for datasets."""

import random
from pathlib import Path
from typing import List, Union


def get_image_paths(images_dir: Union[str, Path]) -> List[Path]:
    """Collect all image paths (.jpg, .png) under images_dir, sorted."""
    path = Path(images_dir)
    paths = sorted(path.glob("*.jpg")) + sorted(path.glob("*.png"))
    return paths


def sample_image_paths(
    images_dir: Union[str, Path],
    num_images: int,
    random_seed: int = 42,
) -> List[Path]:
    """
    Return a random subset of image paths from images_dir.

    Uses random_seed for repeatability. If num_images exceeds available
    images, returns all available.
    """
    all_paths = get_image_paths(images_dir)
    n = min(num_images, len(all_paths))
    rng = random.Random(random_seed)
    return rng.sample(all_paths, n)
