#!/usr/bin/env python3
"""
viz_median_predictions.py — Visualize median-performing image for a given model + prompt.

For each prompt (or only the best prompt), the script:
  1. Runs inference on every image in the dataset at ``best_conf``.
  2. Keeps only images with ≥ ``--min-gt-boxes`` ground-truth boxes (default 5).
  3. Computes per-image F1 at the IoU-0.5 threshold.
  4. Selects the image at the *median* F1 rank — the "average" image, not cherry-picked.
  5. Renders GT boxes (green) and predicted boxes (colour-coded by confidence) on
     that image and saves a PNG to ``<output-dir>/``.

All-models mode (--model-key omitted)
--------------------------------------
When ``--model-key`` is **not** supplied the script reads a
``best_prompt_summary.json`` file (via ``--summary-json``, or auto-discovered
as ``<results-dir>/best_prompt_summary.json``) to obtain the best prompt and
``best_conf`` for every model.

Algorithm
~~~~~~~~~
1. Load SAM3 → run the full image-sweep at SAM3's best_conf → find the
   median-F1 image.  SAM3 is then unloaded (``del`` + ``torch.cuda.empty_cache``).
2. For each of the 4 models (SAM3, YOLO World, GroundingDINO, OWLv2):
   a. Load model.
   b. Determine the image to visualize:
      - Default (``--per-model-median`` absent): use the SAM3 median image for
        all models.
      - With ``--per-model-median``: run a fresh sweep for this model's own
        best prompt to find *its* median image.  Falls back to SAM3 median if
        no qualifying images are found.
   c. Run inference on the chosen image with the model's own best prompt at its
      best_conf.
   d. Run inference again on the same image with the baseline prompt
      ``"a flower"`` at the *same* best_conf (reference comparison).
   e. Save a **side-by-side PNG**: left = best prompt, right = baseline.
      Output name: ``median_pred_allmodels_<img_stem>_<model_key>.png``
   f. Unload model.
3. Save the SAM3 F1-distribution histogram (and, with ``--per-model-median``,
   one per non-SAM3 model as well).

Prompt source priority (single-model mode)
-------------------------------------------
  1. ``--results-dir``  → reads ``eval_<model-key>_prompts.json``
     * Default mode: visualize **only** the ``best_prompt`` block.
     * With ``--all-prompts``: visualize every prompt in the ``results`` block.
  2. ``--prompts-override PATH``  → a JSON file that specifies prompts directly.
     Override JSON schema (flat):
       {
         "img_dir":   "/data/images",
         "lbl_dir":   "/data/labels",
         "prompts": ["a cowpea flower", "white flower"]
       }
     When an override is supplied, img_dir / lbl_dir from the override are used
     (unless ``--img-dir`` / ``--lbl-dir`` are also given, which always win).

Usage examples
--------------
# All-models mode from best_prompt_summary.json (auto-discovered):
    python viz_median_predictions.py \\
        --results-dir /path/to/FINAL-CVPR/SYN-FLOWER \\
        --img-dir /data/images --lbl-dir /data/labels

# All-models mode with explicit summary JSON:
    python viz_median_predictions.py \\
        --summary-json /path/to/best_prompt_summary.json \\
        --img-dir /data/images --lbl-dir /data/labels

# Single-model mode (original behaviour, explicit --model-key required):
    python viz_median_predictions.py \\
        --model-key sam3 \\
        --results-dir /path/to/results \\
        --img-dir /data/images --lbl-dir /data/labels

# Visualize all prompts (single-model):
    python viz_median_predictions.py \\
        --model-key sam3 \\
        --results-dir experiments/results/my_eval \\
        --img-dir /data/images --lbl-dir /data/labels \\
        --all-prompts

CLI flags
---------
--results-dir       Directory used both for loading JSON and for auto-discovering
                    best_prompt_summary.json (all-models mode).
--summary-json      Explicit path to best_prompt_summary.json (all-models mode).
--prompts-override  Override-prompts JSON file (single-model mode only).
--model-key         Which model to load (single-model mode).
                    Omit to activate all-models mode.
--img-dir           Image directory (required in all modes unless inside JSON).
--lbl-dir           Label directory (required in all modes unless inside JSON).
--all-prompts       Visualize every prompt (single-model mode only).
--baseline-prompt   Baseline prompt shown in the right panel of each all-models
                    output (default: "a flower").
--min-gt-boxes      Minimum GT boxes per image to include in ranking (default: 5).
--iou-threshold     IoU threshold for TP matching (default: 0.5).
--device            Compute device, e.g. ``cuda`` or ``cpu`` (default: cuda).
--yolo-weights      Path to YOLO World weights (required if model-key=yolo_world).
--gdino-model-id    HF model ID for GroundingDINO.
--owlv2-model-id    HF model ID for OWLv2.
--sam3-model-id     HF model ID for SAM3.
--output-dir        Output directory for PNGs (default: <results-dir>/plots or ./plots).
--conf-threshold    Inference conf threshold for the sweep pass (default: 0.05).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — __file__ is
#   experiments/scripts/visualization/viz_median_predictions.py
# parent×4 = project root (AgVFM)
# ---------------------------------------------------------------------------
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from agvfm.data import get_full_test_paths, load_ground_truth
from agvfm.evaluation.metrics import match_predictions_to_gt


# ---------------------------------------------------------------------------
# Helpers — serialisation
# ---------------------------------------------------------------------------

def _ascii(text: str) -> str:
    """Strip non-ASCII characters (e.g. emoji) so matplotlib doesn't warn."""
    return text.encode("ascii", errors="ignore").decode("ascii")


def _load_json(path: Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def _load_summary_json(path: Path) -> Dict[str, Dict]:
    """
    Parse ``best_prompt_summary.json`` and return a dict keyed by model key:

    .. code-block:: python

       {
         "yolo_world": {
             "prompt":       "a single yellow bean flower ...",
             "prompt_name":  "comb_clr_spp_anat_...",
             "best_conf":    0.2416,
         },
         "grounding_dino": { ... },
         ...
       }

    Accepts both ``fulldata_best_conf_05`` (preferred) and
    ``sample_best_conf_05`` as the conf source, falling back to 0.05 if
    neither is present.
    """
    data = _load_json(path)
    models_list = data.get("models", [])
    if not models_list:
        raise ValueError(f"No 'models' list found in summary JSON: {path}")

    result: Dict[str, Dict] = {}
    for entry in models_list:
        key = entry.get("model")
        if key is None:
            continue
        prompt      = entry.get("best_prompt") or entry.get("prompt", "")
        prompt_name = entry.get("best_prompt_name", key)
        # Prefer fulldata conf; fall back to sample conf then 0.05
        best_conf   = float(
            entry.get("fulldata_best_conf_05")
            or entry.get("sample_best_conf_05")
            or entry.get("metrics_by_iou", {}).get("iou_0.5", {}).get("best_conf", 0.05)
            or 0.05
        )
        result[key] = dict(prompt=prompt, prompt_name=prompt_name, best_conf=best_conf)
    return result


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def _load_prompts(args: argparse.Namespace) -> List[Dict]:
    """
    Return a list of prompt dicts, each with keys:
      - ``name``       : str label used in filenames / titles
      - ``prompt``     : str text prompt
      - ``best_conf``  : float confidence threshold for visualization
      - ``img_dir``    : str path to image directory
      - ``lbl_dir``    : str path to label directory

    Priority:
      1. ``--prompts-override``  → use prompts from the override JSON
      2. ``--results-dir``       → use eval_<model-key>_prompts.json
         * ``--all-prompts``     → all prompts in the "results" block
         * default              → only the "best_prompt" block
    """
    img_dir_override = args.img_dir
    lbl_dir_override = args.lbl_dir

    # ------------------------------------------------------------------ #
    # Branch 1: override JSON
    # ------------------------------------------------------------------ #
    if args.prompts_override is not None:
        override_path = Path(args.prompts_override)
        if not override_path.exists():
            raise FileNotFoundError(f"Prompts-override file not found: {override_path}")
        data = _load_json(override_path)

        img_dir = str(img_dir_override or data.get("img_dir") or "")
        lbl_dir = str(lbl_dir_override or data.get("lbl_dir") or "")
        if not img_dir or not lbl_dir:
            raise ValueError(
                "img_dir and lbl_dir must be specified via CLI (--img-dir / --lbl-dir) "
                "or inside the override JSON."
            )

        prompts_raw = data.get("prompts", [])
        if not prompts_raw:
            raise ValueError(f"No prompts found in override JSON: {override_path}")

        # Default best_conf when none in the override file
        default_conf = float(data.get("best_conf", args.conf_threshold))

        result = []
        for i, entry in enumerate(prompts_raw):
            if isinstance(entry, str):
                name, prompt, conf = f"prompt_{i}", entry, default_conf
            elif isinstance(entry, dict):
                name   = entry.get("name", f"prompt_{i}")
                prompt = entry["prompt"]
                conf   = float(entry.get("best_conf", default_conf))
            else:
                raise ValueError(f"Unexpected prompt entry type: {type(entry)}")
            result.append(
                dict(name=name, prompt=prompt, best_conf=conf,
                     img_dir=img_dir, lbl_dir=lbl_dir)
            )
        return result

    # ------------------------------------------------------------------ #
    # Branch 2: results dir JSON
    # ------------------------------------------------------------------ #
    if args.results_dir is None:
        raise ValueError(
            "Either --results-dir or --prompts-override must be supplied."
        )
    results_dir = Path(args.results_dir)
    json_path   = results_dir / f"eval_{args.model_key}_prompts.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"Expected results file not found: {json_path}\n"
            f"Use --prompts-override to supply prompts directly."
        )
    data = _load_json(json_path)

    # Resolve img / lbl dirs from file, allow CLI override
    json_img_dir = data.get("img_dir", "")
    json_lbl_dir = data.get("lbl_dir", "")
    img_dir = str(img_dir_override or json_img_dir)
    lbl_dir = str(lbl_dir_override or json_lbl_dir)
    if not img_dir or not lbl_dir:
        raise ValueError(
            "img_dir / lbl_dir not found in results JSON. "
            "Pass --img-dir and --lbl-dir explicitly."
        )

    result = []

    if args.all_prompts:
        # All prompts in the "results" block
        results_block = data.get("results", {})
        if not results_block:
            raise ValueError(f"No 'results' block found in {json_path}")
        for name, pdata in results_block.items():
            prompt   = pdata["prompt"]
            best_conf = float(
                pdata.get("metrics_by_iou", {})
                     .get("iou_0.5", {})
                     .get("best_conf", args.conf_threshold)
            )
            result.append(
                dict(name=name, prompt=prompt, best_conf=best_conf,
                     img_dir=img_dir, lbl_dir=lbl_dir)
            )
    else:
        # Only the best_prompt block
        bp = data.get("best_prompt")
        if bp is None:
            raise ValueError(f"No 'best_prompt' block found in {json_path}")
        name      = bp.get("name", "best_prompt")
        prompt    = bp["prompt"]
        best_conf = float(bp.get("best_conf_05", args.conf_threshold))
        result.append(
            dict(name=name, prompt=prompt, best_conf=best_conf,
                 img_dir=img_dir, lbl_dir=lbl_dir)
        )

    if not result:
        raise ValueError("No prompts could be loaded from the results file.")
    return result


# ---------------------------------------------------------------------------
# Model builder (mirrors eval_prompts.py / load_and_run.py)
# ---------------------------------------------------------------------------

_ALL_MODELS = ["yolo_world", "grounding_dino", "owlv2", "sam3"]


def _build_model(model_key: str, args: argparse.Namespace):
    """Instantiate and return a model object for the given key."""
    from agvfm.models import YOLOWorldModel, SAM3Model, GroundingDINOModel, OWLv2Model

    device = args.device

    if model_key == "yolo_world":
        weights = getattr(args, "yolo_weights", None)
        if weights is None:
            candidates = [
                project_root / "model_weights" / "yolov8x-worldv2.pt",
                project_root / "_data" / "yolov8x-worldv2.pt",
                Path("/group/jmearlesgrp/GEMINI/lars/grounding/model_weights/yolov8x-worldv2.pt"),
            ]
            for p in candidates:
                if p.exists():
                    weights = str(p)
                    break
        if weights is None:
            raise FileNotFoundError("YOLO World weights not found. Pass --yolo-weights PATH.")
        print(f"Loading YOLO World: {weights} …")
        return YOLOWorldModel(weights_path=weights, device=device)

    if model_key == "grounding_dino":
        model_id    = getattr(args, "gdino_model_id", "IDEA-Research/grounding-dino-base")
        box_thresh  = getattr(args, "gdino_box_threshold", 0.3)
        text_thresh = getattr(args, "gdino_text_threshold", 0.25)
        print(f"Loading GroundingDINO: {model_id} …")
        return GroundingDINOModel(
            model_id=model_id, device=device,
            box_threshold=box_thresh, text_threshold=text_thresh,
        )

    if model_key == "owlv2":
        model_id = getattr(args, "owlv2_model_id", "google/owlv2-base-patch16-ensemble")
        print(f"Loading OWLv2: {model_id} …")
        return OWLv2Model(model_id=model_id, device=device)

    if model_key == "sam3":
        model_id = getattr(args, "sam3_model_id", "facebook/sam3")
        print(f"Loading SAM3: {model_id} …")
        return SAM3Model(model_id=model_id, device=device)

    raise ValueError(f"Unknown model key: {model_key!r}")


# ---------------------------------------------------------------------------
# Per-image F1 computation
# ---------------------------------------------------------------------------

def _compute_per_image_f1(
    pred_xyxy: np.ndarray,
    pred_conf: np.ndarray,
    gt_xyxy: np.ndarray,
    conf_threshold: float,
    iou_threshold: float,
) -> float:
    """
    Compute F1 for a single image after filtering predictions to conf >= conf_threshold.

    Returns F1 in [0, 1].  Returns 0.0 if there are no GT boxes (should never
    happen after the ≥5 GT filter, but kept for safety).
    """
    n_gt = len(gt_xyxy)
    if n_gt == 0:
        return 0.0

    # Filter predictions by conf threshold
    if len(pred_conf) > 0:
        keep = pred_conf >= conf_threshold
        pred_xyxy = pred_xyxy[keep]
        pred_conf = pred_conf[keep]

    n_pred = len(pred_xyxy)
    if n_pred == 0:
        # FN = n_gt, TP = FP = 0  →  F1 = 0
        return 0.0

    tp_mask, _ = match_predictions_to_gt(
        pred_xyxy, pred_conf, gt_xyxy, iou_threshold=iou_threshold
    )
    tp = int(np.sum(tp_mask))
    fp = n_pred - tp
    fn = n_gt - tp  # gt_matched count = tp
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Median image finder
# ---------------------------------------------------------------------------

def find_median_image(
    model,
    image_paths: List[str],
    lbl_dir: str,
    prompt: str,
    best_conf: float,
    min_gt_boxes: int,
    iou_threshold: float,
    conf_sweep: float,
    nonzero_median: bool = False,
) -> Tuple[Optional[str], List[Tuple[str, float]]]:
    """
    Run inference on all images, filter to ≥ min_gt_boxes GT boxes, compute
    per-image F1 at best_conf, and return:
      - path of the median-F1 image (or None if no qualifying images)
      - list of (img_path, f1) pairs sorted by F1 ascending (all qualifying)

    If ``nonzero_median`` is True, the median is computed only over images
    with F1 > 0 (i.e. at least one TP prediction).  If no such images exist,
    falls back to the full scored list.
    """
    scored: List[Tuple[str, float]] = []  # (path, f1)

    lbl_dir = Path(lbl_dir)  # ensure Path so labels_dir / stem works
    n_total = len(image_paths)
    for idx, img_path in enumerate(image_paths):
        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"  [{idx + 1}/{n_total}] {Path(img_path).name}")

        # Ground truth
        gt_xyxy = load_ground_truth(img_path, lbl_dir)
        gt_xyxy = np.asarray(gt_xyxy).reshape(-1, 4)
        if len(gt_xyxy) < min_gt_boxes:
            continue

        # Inference at low sweep threshold to get all plausible boxes
        try:
            pred_xyxy, pred_conf = model.predict(img_path, prompt,
                                                  conf_threshold=conf_sweep)
        except Exception as exc:
            print(f"  ⚠️  Inference failed for {Path(img_path).name}: {exc}")
            continue

        pred_xyxy = np.asarray(pred_xyxy).reshape(-1, 4)
        pred_conf = np.asarray(pred_conf).ravel()

        f1 = _compute_per_image_f1(
            pred_xyxy, pred_conf, gt_xyxy,
            conf_threshold=best_conf,
            iou_threshold=iou_threshold,
        )
        scored.append((img_path, f1))

    if not scored:
        return None, []

    scored.sort(key=lambda x: x[1])          # sort by F1 ascending

    # Optionally restrict median selection to images with F1 > 0
    if nonzero_median:
        nonzero = [s for s in scored if s[1] > 0.0]
        median_pool = nonzero if nonzero else scored
        if not nonzero:
            print("  ⚠️  --nonzero-median: no images with F1 > 0; using full scored list.")
    else:
        median_pool = scored

    median_idx  = len(median_pool) // 2
    median_path = median_pool[median_idx][0]
    return median_path, scored


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _conf_colour(conf: float) -> Tuple[float, float, float]:
    """Map confidence in [0,1] to a red→yellow colour (matplotlib RGB tuple)."""
    # Low conf → red, high conf → orange/yellow
    r = 1.0
    g = min(1.0, conf * 1.5)
    b = 0.0
    return (r, g, b)


def visualize_prompt(
    img_path: str,
    model,
    prompt: str,
    best_conf: float,
    lbl_dir: str,
    conf_sweep: float,
    iou_threshold: float,
    scored_images: List[Tuple[str, float]],
    output_path: Path,
    prompt_name: str,
    model_key: str,
    min_gt_boxes: int,
) -> None:
    """
    Generate and save one visualization PNG for a single (prompt, median image) pair.
    Dynamically computes figsize to perfectly match image aspect ratio.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from PIL import Image as PILImage

    # ---- load image ----
    img_pil = PILImage.open(img_path).convert("RGB")
    img_w, img_h = img_pil.size

    lbl_dir = Path(lbl_dir)

    # ---- run inference at sweep threshold ----
    pred_xyxy, pred_conf = model.predict(img_path, prompt, conf_threshold=conf_sweep)
    pred_xyxy = np.asarray(pred_xyxy).reshape(-1, 4)
    pred_conf = np.asarray(pred_conf).ravel()

    # ---- ground truth ----
    gt_xyxy = load_ground_truth(img_path, lbl_dir)
    gt_xyxy = np.asarray(gt_xyxy).reshape(-1, 4)

    # ---- filter preds to best_conf ----
    if len(pred_conf) > 0:
        keep = pred_conf >= best_conf
        pred_xyxy_vis = pred_xyxy[keep]
        pred_conf_vis = pred_conf[keep]
    else:
        pred_xyxy_vis = pred_xyxy
        pred_conf_vis = pred_conf

    # ---- compute per-image F1 ----
    f1 = _compute_per_image_f1(
        pred_xyxy, pred_conf, gt_xyxy,
        conf_threshold=best_conf, iou_threshold=iou_threshold,
    )

    # ---- rank info from scored list ----
    n_qualifying = len(scored_images)
    median_rank  = n_qualifying // 2 + 1

    # ---- Dynamically compute figure dimensions to match aspect ratio ----
    fig_w = 12.0
    panel_h_inches = fig_w * (img_h / img_w)
    title_h_inches = 1.2
    fig_h = panel_h_inches + title_h_inches
    panel_h_frac = panel_h_inches / fig_h

    # ---- figure ----
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([0.0, 0.0, 1.0, panel_h_frac])
    ax.imshow(img_pil)
    ax.axis("off")

    # GT boxes — green, solid
    for box in gt_xyxy:
        x1, y1, x2, y2 = box
        rect = mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor="lime", facecolor="none",
        )
        ax.add_patch(rect)

    # Predicted boxes — red→yellow, colour by confidence
    for box, conf in zip(pred_xyxy_vis, pred_conf_vis):
        x1, y1, x2, y2 = box
        rect = mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor="red", facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            x1, max(y1 - 3, 0), f"{conf:.2f}",
            color="red", fontsize=7, fontweight="bold",
            clip_on=True,
        )

    # Legend
    gt_patch   = mpatches.Patch(edgecolor="lime",   facecolor="none", linewidth=2, label=f"GT  ({len(gt_xyxy)} boxes)")
    pred_patch = mpatches.Patch(edgecolor="red", facecolor="none", linewidth=2, label=f"Pred ({len(pred_xyxy_vis)} boxes, conf≥{best_conf:.2f})")
    ax.legend(handles=[gt_patch, pred_patch], loc="upper right", fontsize=9,
              framealpha=0.8)

    # Title
    img_name = Path(img_path).name
    title = (
        f"Model: {model_key}  |  Prompt: \"{_ascii(prompt)}\"\n"
        f"Image: {img_name}  (rank {median_rank}/{n_qualifying}, F1={f1:.3f})\n"
        f"conf≥{best_conf:.2f}, IoU≥{iou_threshold:.1f}, "
        f"qualifying images (≥{min_gt_boxes} GT): {n_qualifying}"
    )
    fig.text(0.0, panel_h_frac + 0.01, title, fontsize=10, va="bottom", ha="left")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved: {output_path}")


# ---------------------------------------------------------------------------
# F1 distribution plot (optional; always generated)
# ---------------------------------------------------------------------------

def plot_f1_distribution(
    scored_images: List[Tuple[str, float]],
    median_path: str,
    prompt_name: str,
    output_path: Path,
) -> None:
    """Save a histogram of per-image F1 scores with median marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f1_values = [s[1] for s in scored_images]
    median_f1 = f1_values[len(f1_values) // 2]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(f1_values, bins=20, color="steelblue", edgecolor="white", linewidth=0.5)
    ax.axvline(median_f1, color="red", linestyle="--", linewidth=1.5,
               label=f"Median F1 = {median_f1:.3f}")
    ax.set_xlabel("Per-image F1 @ IoU-0.5", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"F1 Distribution — {prompt_name}\nn={len(f1_values)} qualifying images",
                 fontsize=11)
    ax.legend(fontsize=10)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved F1 distribution: {output_path}")


# ---------------------------------------------------------------------------
# Two-panel visualisation (best-prompt  |  baseline-prompt)
# ---------------------------------------------------------------------------

def visualize_two_panel(
    img_path: str,
    model,
    best_prompt: str,
    baseline_prompt: str,
    best_conf: float,
    lbl_dir: str,
    conf_sweep: float,
    iou_threshold: float,
    output_path: Path,
    model_key: str,
    model_label: str,
    scored_images: List[Tuple[str, float]],
    min_gt_boxes: int,
) -> None:
    """
    Save a side-by-side PNG with two panels tightly bound to the aspect ratio
    of the image.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from PIL import Image as PILImage

    img_pil  = PILImage.open(img_path).convert("RGB")
    gt_xyxy  = np.asarray(load_ground_truth(img_path, Path(lbl_dir))).reshape(-1, 4)

    n_qualifying = len(scored_images)
    median_rank  = n_qualifying // 2 + 1

    def _run(prompt: str):
        """Return (pred_xyxy_vis, pred_conf_vis, f1) filtered to best_conf."""
        try:
            raw_xyxy, raw_conf = model.predict(img_path, prompt, conf_threshold=conf_sweep)
        except Exception as exc:
            print(f"  ⚠️  {model_key} inference failed ({prompt!r}): {exc}")
            return np.zeros((0, 4)), np.zeros(0), 0.0
        raw_xyxy = np.asarray(raw_xyxy).reshape(-1, 4)
        raw_conf = np.asarray(raw_conf).ravel()
        if len(raw_conf) > 0:
            keep = raw_conf >= best_conf
            vis_xyxy, vis_conf = raw_xyxy[keep], raw_conf[keep]
        else:
            vis_xyxy, vis_conf = raw_xyxy, raw_conf
        f1 = _compute_per_image_f1(raw_xyxy, raw_conf, gt_xyxy,
                                   conf_threshold=best_conf,
                                   iou_threshold=iou_threshold)
        return vis_xyxy, vis_conf, f1

    best_xyxy,  best_conf_arr,  f1_best  = _run(best_prompt)
    base_xyxy,  base_conf_arr,  f1_base  = _run(baseline_prompt)

    img_w, img_h = img_pil.size
    
    # ---- Dynamically compute figure dimensions ----
    fig_w = 22.0
    gap_frac   = 0.004
    panel_w_frac = (1.0 - gap_frac) / 2.0
    
    panel_w_inches = fig_w * panel_w_frac
    panel_h_inches = panel_w_inches * (img_h / img_w)
    
    title_h_inches = 1.0  # Room for the two lines of text at the top
    fig_h = panel_h_inches + title_h_inches
    
    panel_h_frac = panel_h_inches / fig_h

    # Add axes with 0 margin on left/right/bottom to eliminate whitespace
    fig = plt.figure(figsize=(fig_w, fig_h))
    axes = [
        fig.add_axes([0.0,                     0.0, panel_w_frac, panel_h_frac]),
        fig.add_axes([panel_w_frac + gap_frac, 0.0, panel_w_frac, panel_h_frac]),
    ]

    panel_data = [
        (axes[0], base_xyxy,  base_conf_arr,  f1_base,  baseline_prompt, "Baseline"),
        (axes[1], best_xyxy,  best_conf_arr,  f1_best,  best_prompt,     "Best prompt"),
    ]

    for ax, pred_xyxy_vis, pred_conf_vis, f1, prompt_text, panel_label in panel_data:
        ax.imshow(img_pil)
        ax.axis("off")

        # GT — green
        for box in gt_xyxy:
            x1, y1, x2, y2 = box
            ax.add_patch(mpatches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor="lime", facecolor="none",
            ))

        # Predictions — red→yellow by conf
        for box, conf in zip(pred_xyxy_vis, pred_conf_vis):
            x1, y1, x2, y2 = box
            ax.add_patch(mpatches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor="red", facecolor="none",
            ))
            ax.text(x1, max(y1 - 3, 0), f"{conf:.2f}",
                    color="red", fontsize=7, fontweight="bold", clip_on=True)

        gt_patch   = mpatches.Patch(edgecolor="lime", facecolor="none",
                                    linewidth=2, label=f"GT ({len(gt_xyxy)})")
        pred_patch = mpatches.Patch(edgecolor="red",  facecolor="none",
                                    linewidth=2,
                                    label=f"Pred ({len(pred_xyxy_vis)}, conf≥{best_conf:.2f})")
        ax.legend(handles=[gt_patch, pred_patch], loc="upper right",
                  fontsize=9, framealpha=0.8)

        # Title in figure coordinates securely above the perfectly-fit panel
        ax_x = ax.get_position().x0
        fig.text(
            ax_x, panel_h_frac + 0.01,
            f"[{panel_label}]  {model_label}\n"
            f"Prompt: \"{_ascii(prompt_text)}\"\n"
            f"rank {median_rank}/{n_qualifying}, F1={f1:.3f}  |  "
            f"conf≥{best_conf:.2f}, IoU≥{iou_threshold:.1f}",
            fontsize=10, va="bottom", ha="left",
        )

    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved: {output_path}")


# ---------------------------------------------------------------------------
# All-models mode orchestrator
# ---------------------------------------------------------------------------

_MODEL_ORDER = ["sam3", "yolo_world", "grounding_dino", "owlv2"]
_MODEL_LABELS = {
    "sam3":            "SAM3",
    "yolo_world":      "YOLO World",
    "grounding_dino":  "GroundingDINO",
    "owlv2":           "OWLv2",
}


def run_all_models_mode(args: argparse.Namespace, out_dir: Path) -> None:
    """
    All-models mode:
      1. Load summary JSON → get best prompt + best_conf for each model.
      2. Build SAM3 → find median image → unload SAM3.
      3. For each model sequentially: load → two-panel viz → unload.
      4. Save F1-distribution for SAM3.
    """
    import gc
    try:
        import torch
        _has_torch = True
    except ImportError:
        _has_torch = False

    def _free(model) -> None:
        del model
        gc.collect()
        if _has_torch:
            torch.cuda.empty_cache()

    # ---- locate summary JSON ----
    summary_path: Optional[Path] = None
    if args.summary_json:
        summary_path = Path(args.summary_json)
    elif args.results_dir:
        candidate = Path(args.results_dir) / "best_prompt_summary.json"
        if candidate.exists():
            summary_path = candidate
    if summary_path is None or not summary_path.exists():
        raise FileNotFoundError(
            "Could not find best_prompt_summary.json.\n"
            "Pass --summary-json PATH or place the file in --results-dir."
        )
    print(f"Loading summary JSON: {summary_path}")
    summary = _load_summary_json(summary_path)

    # ---- validate img/lbl dirs ----
    img_dir = args.img_dir
    lbl_dir = args.lbl_dir
    if not img_dir or not lbl_dir:
        raise ValueError("--img-dir and --lbl-dir are required in all-models mode.")

    # ---- discover images ----
    print(f"Discovering images in: {img_dir}")
    image_paths = get_full_test_paths(img_dir)
    if not image_paths:
        raise ValueError(f"No images found in: {img_dir}")
    print(f"  → {len(image_paths)} image(s)")

    baseline_prompt: str = args.baseline_prompt

    # Per-model prompt overrides from CLI flags
    _PROMPT_OVERRIDES: Dict[str, Optional[str]] = {
        "sam3":           getattr(args, "sam3_prompt",       None),
        "yolo_world":     getattr(args, "yolo_world_prompt", None),
        "grounding_dino": getattr(args, "gdino_prompt",      None),
        "owlv2":          getattr(args, "owlv2_prompt",      None),
    }

    # ---- Step 1: SAM3 → find median image ----
    sam3_info = summary.get("sam3")
    if sam3_info is None:
        raise ValueError("'sam3' entry not found in summary JSON.")

    print(f"\n[Step 1/2] Building SAM3 to find median image …")
    sam3_model = _build_model("sam3", args)
    print(f"  SAM3 best prompt: \"{sam3_info['prompt']}\"  (conf={sam3_info['best_conf']:.4f})")
    print(f"  Sweeping {len(image_paths)} images …")

    median_path, scored = find_median_image(
        model=sam3_model,
        image_paths=image_paths,
        lbl_dir=lbl_dir,
        prompt=sam3_info["prompt"],
        best_conf=sam3_info["best_conf"],
        min_gt_boxes=args.min_gt_boxes,
        iou_threshold=args.iou_threshold,
        conf_sweep=args.conf_threshold,
        nonzero_median=args.nonzero_median,
    )

    if median_path is None:
        print(f"  ⚠️  No qualifying images (≥{args.min_gt_boxes} GT boxes). Aborting.")
        _free(sam3_model)
        return

    n_qualifying = len(scored)
    median_f1    = scored[n_qualifying // 2][1]
    img_stem     = Path(median_path).stem
    print(f"  ✓ Median image: {Path(median_path).name}  F1={median_f1:.3f} "
          f"({n_qualifying} qualifying images)")

    # Save SAM3 F1 distribution
    dist_path = out_dir / f"f1_dist_allmodels_{img_stem}.png"
    plot_f1_distribution(
        scored_images=scored,
        median_path=median_path,
        prompt_name=f"SAM3 — {sam3_info['prompt_name']}",
        output_path=dist_path,
    )

    # ---- Step 2: loop all models ----
    print(f"\n[Step 2/2] Rendering all models on median image: {Path(median_path).name}")
    for model_key in _MODEL_ORDER:
        minfo = summary.get(model_key)
        if minfo is None:
            print(f"  ⚠️  No entry for '{model_key}' in summary JSON — skipping.")
            continue

        label = _MODEL_LABELS.get(model_key, model_key)
        print(f"\n  [{model_key}]  Loading {label} …")
        model = _build_model(model_key, args)

        # Apply per-model prompt override if supplied
        best_prompt = _PROMPT_OVERRIDES.get(model_key) or minfo["prompt"]
        if _PROMPT_OVERRIDES.get(model_key):
            print(f"  ⚡ Prompt override active: \"{best_prompt}\"")

        # Per-model median: re-run the image sweep for this model's own prompt
        if args.per_model_median and model_key != "sam3":
            print(f"  --per-model-median: sweeping {len(image_paths)} images for {label} …")
            this_median_path, this_scored = find_median_image(
                model=model,
                image_paths=image_paths,
                lbl_dir=lbl_dir,
                prompt=best_prompt,
                best_conf=minfo["best_conf"],
                min_gt_boxes=args.min_gt_boxes,
                iou_threshold=args.iou_threshold,
                conf_sweep=args.conf_threshold,
                nonzero_median=args.nonzero_median,
            )
            if this_median_path is None:
                print(f"  ⚠️  No qualifying images for {label}. Falling back to SAM3 median.")
                this_median_path = median_path
                this_scored      = scored
            else:
                this_f1 = this_scored[len(this_scored) // 2][1]
                print(f"  ✓ {label} median: {Path(this_median_path).name}  F1={this_f1:.3f}")
                # Save per-model F1 distribution
                model_dist_path = out_dir / f"f1_dist_allmodels_{Path(this_median_path).stem}_{model_key}.png"
                plot_f1_distribution(
                    scored_images=this_scored,
                    median_path=this_median_path,
                    prompt_name=f"{label} — {minfo['prompt_name']}",
                    output_path=model_dist_path,
                )
        else:
            this_median_path = median_path
            this_scored      = scored

        out_path = out_dir / f"median_pred_{model_key}_allmodels_{Path(this_median_path).stem}.png"
        print(f"  Rendering two-panel viz → {out_path.name}")
        visualize_two_panel(
            img_path=this_median_path,
            model=model,
            best_prompt=best_prompt,
            baseline_prompt=baseline_prompt,
            best_conf=minfo["best_conf"],
            lbl_dir=lbl_dir,
            conf_sweep=args.conf_threshold,
            iou_threshold=args.iou_threshold,
            output_path=out_path,
            model_key=model_key,
            model_label=label,
            scored_images=this_scored,
            min_gt_boxes=args.min_gt_boxes,
        )

        print(f"  Unloading {label} …")
        _free(model)

    print(f"\nAll-models mode done. Outputs in: {out_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # ---- output directory ----
    if args.output_dir:
        out_dir = Path(args.output_dir)
    elif args.results_dir:
        out_dir = Path(args.results_dir) / "plots"
    else:
        out_dir = Path("plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- branch: all-models mode vs single-model mode ----
    if args.model_key is None:
        run_all_models_mode(args, out_dir)
        return

    # ------------------------------------------------------------------ #
    # Single-model mode (original behaviour)
    # ------------------------------------------------------------------ #

    # ---- load prompts ----
    print("Loading prompt definitions …")
    prompts = _load_prompts(args)
    print(f"  → {len(prompts)} prompt(s): {[p['name'] for p in prompts]}")

    # All prompts share the same img_dir / lbl_dir (first entry is authoritative)
    img_dir = prompts[0]["img_dir"]
    lbl_dir = prompts[0]["lbl_dir"]

    # ---- discover images ----
    print(f"Discovering images in: {img_dir}")
    image_paths = get_full_test_paths(img_dir)
    if not image_paths:
        raise ValueError(f"No images found in: {img_dir}")
    print(f"  → {len(image_paths)} image(s)")

    # ---- build model ----
    print(f"Building model: {args.model_key}")
    model = _build_model(args.model_key, args)
    print("  → model ready")

    # ---- process each prompt ----
    for pidx, pinfo in enumerate(prompts):
        pname     = pinfo["name"]
        prompt    = pinfo["prompt"]
        best_conf = pinfo["best_conf"]

        print(f"\n[{pidx + 1}/{len(prompts)}] Prompt: \"{prompt}\"  (name={pname}, best_conf={best_conf:.3f})")
        print(f"  Sweeping {len(image_paths)} images at conf_sweep={args.conf_threshold:.3f}, "
              f"min_gt={args.min_gt_boxes} …")

        median_path, scored = find_median_image(
            model=model,
            image_paths=image_paths,
            lbl_dir=lbl_dir,
            prompt=prompt,
            best_conf=best_conf,
            min_gt_boxes=args.min_gt_boxes,
            iou_threshold=args.iou_threshold,
            conf_sweep=args.conf_threshold,
            nonzero_median=args.nonzero_median,
        )

        if median_path is None:
            print(f"  ⚠️  No qualifying images (≥{args.min_gt_boxes} GT boxes). Skipping.")
            continue

        n_qualifying = len(scored)
        median_f1    = scored[n_qualifying // 2][1]
        print(f"  Median image: {Path(median_path).name}  F1={median_f1:.3f}  "
              f"({n_qualifying} qualifying images)")

        # Safe filename from prompt name
        safe_name = pname.replace("/", "_").replace(" ", "_")

        # Main visualization
        viz_path = out_dir / f"median_pred_{args.model_key}_{safe_name}.png"
        print(f"  Rendering visualization → {viz_path.name}")
        visualize_prompt(
            img_path=median_path,
            model=model,
            prompt=prompt,
            best_conf=best_conf,
            lbl_dir=lbl_dir,
            conf_sweep=args.conf_threshold,
            iou_threshold=args.iou_threshold,
            scored_images=scored,
            output_path=viz_path,
            prompt_name=pname,
            model_key=args.model_key,
            min_gt_boxes=args.min_gt_boxes,
        )

        # F1 distribution histogram
        dist_path = out_dir / f"f1_dist_{args.model_key}_{safe_name}.png"
        plot_f1_distribution(
            scored_images=scored,
            median_path=median_path,
            prompt_name=pname,
            output_path=dist_path,
        )

    print(f"\nDone. Outputs in: {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Visualize median-performing image for a model + prompt.\n"
            "Omit --model-key to activate all-models mode (reads best_prompt_summary.json)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Prompt / data sources
    src = p.add_argument_group("Prompt / data sources")
    src.add_argument(
        "--results-dir", default=None,
        help=(
            "Directory containing eval_<model-key>_prompts.json (single-model mode) "
            "or best_prompt_summary.json (all-models mode, auto-discovered)."
        ),
    )
    src.add_argument(
        "--summary-json", default=None, metavar="PATH",
        help="Explicit path to best_prompt_summary.json (all-models mode).",
    )
    src.add_argument(
        "--prompts-override", default=None, metavar="PATH",
        help="Override-prompts JSON file (single-model mode only).",
    )
    src.add_argument(
        "--img-dir", default=None,
        help="Image directory.",
    )
    src.add_argument(
        "--lbl-dir", default=None,
        help="Label directory.",
    )
    src.add_argument(
        "--all-prompts", action="store_true",
        help="Visualize every prompt in the results block (single-model mode only).",
    )

    # Model
    mdl = p.add_argument_group("Model")
    mdl.add_argument(
        "--model-key", default=None,
        choices=["yolo_world", "grounding_dino", "owlv2", "sam3"],
        help=(
            "Which model to load for single-model mode.  "
            "Omit (default) to activate all-models mode."
        ),
    )
    mdl.add_argument("--device", default="cuda",
                     help="Compute device (cuda / cpu).")
    mdl.add_argument("--yolo-weights", default=None,
                     help="Path to YOLO World weights (.pt).")
    mdl.add_argument("--gdino-model-id",
                     default="IDEA-Research/grounding-dino-base",
                     help="HuggingFace model ID for GroundingDINO.")
    mdl.add_argument("--gdino-box-threshold", type=float, default=0.3)
    mdl.add_argument("--gdino-text-threshold", type=float, default=0.25)
    mdl.add_argument("--owlv2-model-id",
                     default="google/owlv2-base-patch16-ensemble",
                     help="HuggingFace model ID for OWLv2.")
    mdl.add_argument("--sam3-model-id", default="facebook/sam3",
                     help="HuggingFace model ID for SAM3.")

    # Per-model prompt overrides (all-models mode only)
    # When supplied, replaces the best_prompt from best_prompt_summary.json
    # for that model only.  The best_conf from the summary is still used
    # unless the override prompt warrants a different threshold.
    ovr = p.add_argument_group("Per-model prompt overrides (all-models mode)")
    ovr.add_argument("--sam3-prompt",       default=None, metavar="PROMPT",
                     help="Override SAM3 best prompt for visualization.")
    ovr.add_argument("--yolo-world-prompt", default=None, metavar="PROMPT",
                     help="Override YOLO World best prompt for visualization.")
    ovr.add_argument("--gdino-prompt",      default=None, metavar="PROMPT",
                     help="Override GroundingDINO best prompt for visualization.")
    ovr.add_argument("--owlv2-prompt",      default=None, metavar="PROMPT",
                     help="Override OWLv2 best prompt for visualization.")

    # Ranking / filtering
    rank = p.add_argument_group("Ranking / filtering")
    rank.add_argument(
        "--min-gt-boxes", type=int, default=5,
        help="Minimum number of GT boxes for an image to be included in ranking.",
    )
    rank.add_argument(
        "--iou-threshold", type=float, default=0.5,
        help="IoU threshold for TP matching.",
    )
    rank.add_argument(
        "--conf-threshold", type=float, default=0.05,
        help="Low inference conf threshold for the sweep pass.",
    )
    rank.add_argument(
        "--baseline-prompt", default="a flower",
        help="Baseline reference prompt shown in the right panel of each all-models output.",
    )
    rank.add_argument(
        "--per-model-median", action="store_true",
        help=(
            "All-models mode: find each model's own median image (using its own best "
            "prompt + best_conf sweep) instead of reusing the SAM3 median for all models. "
            "SAM3 still runs first and its median is used as the anchor for the F1 "
            "distribution histogram. Each other model also saves its own F1 histogram."
        ),
    )
    rank.add_argument(
        "--nonzero-median", action="store_true",
        help=(
            "When finding the median image, restrict the median selection to images "
            "with F1 > 0 (i.e. at least one correct prediction).  Images with zero "
            "predictions are still included in the scored list and F1 histogram but "
            "are excluded from median selection.  Falls back to the full list if no "
            "image has F1 > 0."
        ),
    )

    # Output
    out_grp = p.add_argument_group("Output")
    out_grp.add_argument(
        "--output-dir", default=None,
        help="Output directory for PNGs (default: <results-dir>/plots or ./plots).",
    )

    return p.parse_args()


if __name__ == "__main__":
    main()