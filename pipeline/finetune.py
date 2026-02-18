'''Finetuning logic for AgVFM pipeline
1) Load pretrained model
2) Randomly initialize text embeddings and remove text input layer if needed (e.g. for Grounding DINO)
3) Select k synthetic images for k-shot finetuning
4) Finetune text embeddings on k synthetic images
5) Evaluate finetuned model on test set and compare to zero-shot performance
...
6) Finetune on real images to test generalization and evaluate again
'''

model_list = ["yolo", "grounding_dino", "owlv2"]
weights_dict = {"yolo": "../model_weights/yolov8x-worldv2.pt",
                "grounding_dino": "IDEA-Research/grounding-dino-base",
                "owlv2": "google/owlvit-large-patch14"
}

import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch

# local utilities
from vlme.data.sampling import get_image_paths
from vlme.data.labels import parse_yolo_label


def sample_k_pairs(images_dir: str, labels_dir: str, k: int, seed: int = 42) -> List[Tuple[Path, Path]]:
    """Randomly sample k image/label pairs from given directories.

    Args:
        images_dir: directory containing images (.jpg/.png)
        labels_dir: directory containing yolo .txt labels (same stem as images) OR a folder produced by convert_coco (yolo labels dir)
        k: number of samples
        seed: random seed

    Returns:
        list of tuples (image_path, label_path)
    """
    images = get_image_paths(images_dir)
    rng = random.Random(seed)
    if k >= len(images):
        chosen = images
    else:
        chosen = rng.sample(images, k)

    pairs = []
    for p in chosen:
        label_path = Path(labels_dir) / (p.stem + ".txt")
        pairs.append((p, label_path))
    return pairs


def build_k_shot_dataset(pairs: List[Tuple[Path, Path]], class_names: Optional[List[str]] = None) -> List[Dict]:
    """Load images and parse YOLO labels for the sampled pairs.

    Returns a list of dicts with keys: image_path, gt_boxes (xyxy), gt_classes (int list), prompts (text prompts)
    """
    out = []
    class_names = class_names or ["flower"]
    for img_path, label_path in pairs:
        # parse_yolo_label returns list of (x1,y1,x2,y2,class_id) in absolute coords according to existing helper
        boxes = parse_yolo_label(label_path, img_width=None, img_height=None) if label_path.exists() else []
        # parse_yolo_label in this codebase expects width/height; if signature differs, caller may adapt.
        # We will normalize to xyxy and class ids
        gt_xyxy = []
        gt_cls = []
        for entry in boxes:
            if len(entry) == 5:
                x1, y1, x2, y2, cid = entry
            else:
                # fallback: ignore
                continue
            gt_xyxy.append([x1, y1, x2, y2])
            gt_cls.append(int(cid))

        # create simple prompts per ground-truth class for the image (e.g., 'small yellow flower')
        prompts = []
        for cid in gt_cls:
            if 0 <= cid < len(class_names):
                prompts.append(class_names[cid])
            else:
                prompts.append(class_names[0])

        out.append({"image_path": img_path, "gt_xyxy": gt_xyxy, "gt_cls": gt_cls, "prompts": prompts})
    return out


def finetune_text_embeddings(
    model_type: str,
    model_wrapper,  # instance of wrapper (for HF models) or None to load
    k_shot_dataset: List[Dict],
    class_names: List[str],
    epochs: int = 5,
    lr: float = 1e-4,
    device: Optional[str] = None,
    dry_run: bool = True,
) -> Dict:
    """Finetune text embeddings for HF models using the small k-shot dataset.

    This is a conservative scaffolding function that prepares parameters and an optimizer.
    The actual loss design depends on your chosen supervision signal (e.g., grounding scores, contrastive losses).

    Args:
        model_type: one of 'grounding_dino' or 'owlv2' (YOLO fine-tuning not implemented here)
        model_wrapper: an instantiated wrapper (GroundingDinoWrapper or OwlV2Wrapper). If None and model_type is HF, the caller should instantiate and pass it.
        k_shot_dataset: list returned by build_k_shot_dataset
        class_names: mapping of class ids to text names
        epochs, lr: training hyperparameters
        device: device string
        dry_run: if True, only inspect and report trainable params without running updates

    Returns:
        dict with info about trainable params and a placeholder for saved embedding path (if any)
    """
    if model_type not in ("grounding_dino", "owlv2"):
        raise ValueError("finetune_text_embeddings supports only HF models: grounding_dino or owlv2")

    if model_wrapper is None:
        raise ValueError("Please provide an instantiated HF model wrapper (GroundingDinoWrapper or OwlV2Wrapper)")

    hf_model = getattr(model_wrapper, "model", None)
    if hf_model is None:
        raise ValueError("Provided wrapper does not expose .model")

    # Try to locate text embedding / encoder parameters
    text_params = []
    for name, param in hf_model.named_parameters():
        lname = name.lower()
        if "text" in lname or "token" in lname or "embedding" in lname:
            # heuristically include likely text-encoder params
            text_params.append((name, param))

    info = {"n_text_params": len(text_params), "text_param_names": [n for n, _ in text_params]}

    if dry_run:
        print("Dry run: identified text params to finetune:\n", info)
        return info

    # If the user opted into real training, construct optimizer over these params
    params = [p for _, p in text_params if p.requires_grad is True or True]
    for _, p in text_params:
        p.requires_grad = True

    optim = torch.optim.Adam(params, lr=lr)

    # Training loop skeleton: user should implement a proper loss using outputs of the model
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for sample in k_shot_dataset:
            image_path = sample["image_path"]
            prompts = sample.get("prompts", class_names[:1])
            # TODO: design a loss. For now we raise to prevent accidental runs.
            raise NotImplementedError(
                "Finetuning loop is a scaffold. Implement a task-specific loss (e.g., maximize grounding scores for correct prompts and minimize for negatives)."
            )

    # Placeholder: if training completes, save modified model or embeddings
    saved = None
    return {"trained": True, "saved_path": saved}


if __name__ == "__main__":
    # Example usage: sample k images and build dataset (dry run)
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--labels_dir", type=str, required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pairs = sample_k_pairs(args.images_dir, args.labels_dir, args.k, args.seed)
    dataset = build_k_shot_dataset(pairs, class_names=["flower"])
    print(f"Built {len(dataset)} k-shot samples (dry run). Call finetune_text_embeddings with dry_run=False to implement training.")

