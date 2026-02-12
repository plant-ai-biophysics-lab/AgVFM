# AgVFM

Vision-language and detection utilities for agronomic VFM (e.g. flower/bud detection with YOLO World): data loading, evaluation (mAP@0.5, P/R), and visualizations.

## Install

Recommended using Python 3.14. Using a conda environment is recommended:

```bash
conda create -n agvfm python=3.14
conda activate agvfm
cd /path/to/AgVFM
pip install -e .
```

From the repo root without conda:

```bash
pip install -e .
```

## Data and weights

- **Data:** Place your dataset under `_data/` with YOLO-style layout (e.g. `_data/.../test/images`, `_data/.../test/labels`). These paths are gitignored.
- **Weights:** Put YOLO World `.pt` files in `model_weights/` (e.g. `model_weights/yolov8x-worldv2.pt`). Also gitignored.

## Usage

```python
from vlme import run_gt_vs_predictions, run_evaluation, OPEN_SET_CLASS_NAMES
from pathlib import Path

IMAGES_DIR = Path("_data/T4_REAL_FLOWER_SOBJ.../test/images")
LABELS_DIR = Path("_data/T4_REAL_FLOWER_SOBJ.../test/labels")

# Qualitative: sample images with GT vs predictions
run_gt_vs_predictions(IMAGES_DIR, LABELS_DIR, num_images=6, random_seed=42)

# Evaluation: mAP@0.5, precision, recall (full set or limited)
metrics = run_evaluation(IMAGES_DIR, LABELS_DIR, num_images=None, plot_summary=True, plot_pr_curve=True)
# Or subset: num_images=100, plot_sample=6 for a few qualitative panels
```

Device is chosen automatically: CUDA if available, else MPS (Mac), else CPU. Override with `predict_kwargs=dict(device="cpu")` if needed.

## Package layout

- `vlme/` – main package: `data` (labels, sampling), `models` (YOLO World), `viz` (boxes, eval plots), `evaluation` (metrics, run_eval), `run` (high-level pipelines).
