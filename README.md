# AgVFM

AgVFM is an agronomic open-vocabulary / open-set detection experimentation pipeline centered on two stages:

- **Phase 1 (OFAT factor analysis):** isolate and score prompt factors (taxonomy, color, size, phenology, negation, anatomy, grammar, emoji).
- **Phase 2 (combinatorial search):** build and evaluate prompt combinations using the strongest Phase 1 signals, then expand with negation and emoji variants.

Primary entry points:

- `experiments/scripts/experiments/load_and_run.py` for end-to-end batch runs.
- `notebooks/run_pipeline.ipynb` for interactive run/inspect/visualize workflow.

## Install

From the repo root:

```bash
pip install -e .
```

## Data and model weights

- **Images + labels:** point `--img-dir` and `--lbl-dir` at your evaluation set if using `load_and_run.py`, set macro in `run_pipeline.ipynb` otherwise.
- **Label formats:**
	- YOLO `.txt` labels are used directly.
	- COCO `.json` labels are auto-converted to YOLO format by `load_and_run.py`.
- **YOLO World weights:** place `.pt` files in `model_weights/` (or pass `--yolo-weights`).

## Main workflow (`load_and_run.py`)

Script: `experiments/scripts/experiments/load_and_run.py`

### What it does

1. Validates paths and loads test image list.
2. Converts COCO labels to YOLO format when needed.
3. Resolves factor axes from one of:
	 - `--axes-file` (explicit),
	 - existing `factor_axes.json` in results directory,
	 - LLM translation (Qwen3-4B) when crop text is provided,
	 - default built-in axes.
4. Runs Phase 1 and/or Phase 2 for one model or all models.
5. Optionally samples images via `--sample-size`, then re-validates the best prompt on full data.
6. Writes per-model JSON outputs plus a global best-prompt summary.

### Qwen factor-axis translation prompt

When crop text is provided (for `--run-ph1` and/or `--run-ph2`) and no prior axes file is reused, `load_and_run.py` calls Qwen (`Qwen/Qwen3-4B`) with a strict JSON-only prompt protocol.

System prompt behavior:

- Forces JSON-only output (no explanation, no chain-of-thought).
- Preserves axis names: `taxonomy`, `color`, `size`, `phenology`, `negation`, `anatomy`, `grammar`, `emoji`.
- Requires each axis object to contain exactly: `name`, `values`, `baseline`.
- Instructs grammar to remain unchanged.
- Requests realistic negation confounders and 3-5 relevant emoji values.

User prompt template behavior:

- Injects the target crop/object string.
- Injects the original cowpea-flower `FACTOR_AXES` JSON.
- Asks for translated `values` and `baseline` fields for each axis.

Expected output schema:

```json
[
	{
		"name": "taxonomy",
		"values": ["...", "..."],
		"baseline": "..."
	}
]
```

Validation and fallback behavior:

- The script strips accidental code fences and extracts the outer JSON array.
- If parsing or schema validation fails, it falls back to default built-in axes (problematic silent failure point as of now, so this should be changed; failures are avoided by ensuring Qwen avoids "thinking", but this was an issue before).
- Successful translations are saved to `factor_axes.json` in the results directory for reuse.

### Supported models (as of now)

- `yolo_world`
- `grounding_dino`
- `owlv2`
- `sam3`
- `all` (sequentially runs all models)

*Note*: Other models can be added as desired by adding their necessary functionality to a `AgVFM/agvfm/models/{MODEL_NAME}.py` file and adding imports and logic to the Phase 1 + 2 analysis scripts.

### Example commands

Run full Phase 1 + 2 on all models:

```bash
python experiments/scripts/experiments/load_and_run.py \
	--run-ph1 "cowpea flower" \
	--run-ph2 "cowpea flower" \
	--model all \
	--img-dir data/all_flower_test \
	--lbl-dir data/all_flower_test \
	--results-dir experiments/results/load_and_run/SYN-FLOWER \
	--device cuda
```

Fast debug run on a sample:

```bash
python experiments/scripts/experiments/load_and_run.py \
	--run-ph1 "cowpea flower" \
	--run-ph2 "cowpea flower" \
	--model yolo_world \
	--img-dir data/all_flower_test \
	--lbl-dir data/all_flower_test \
	--sample-size 200
```

Reuse existing axes and run only Phase 2:

```bash
python experiments/scripts/experiments/load_and_run.py \
	--run-ph2 "cowpea flower" \
	--axes-file experiments/results/load_and_run/SYN-FLOWER/factor_axes.json \
	--model grounding_dino \
	--img-dir data/all_flower_test \
	--lbl-dir data/all_flower_test
```

## Notebook workflow (`run_pipeline.ipynb`)

Notebook: `notebooks/run_pipeline.ipynb`

The notebook is designed to:

1. Configure paths and run tags.
2. Optionally launch `load_and_run.py` (toggle with `RUN_PIPELINE=True`).
3. Load Phase 1/2 JSON artifacts.
4. Produce primary visual summaries:
	 - spider summary grid,
	 - per-axis spider comparisons.
5. Produce secondary comparisons:
	 - per-model all-metrics factor contributions,
	 - GroundingDINO vs OWLv2 comparisons,
	 - four-model comparison plots.
6. Build a prompt discovery table (Phase 1 vs Phase 2 best prompts).
7. Optionally generate median-image qualitative overlays (baseline prompt vs best prompt).

## Output artifacts

Key outputs written under your `--results-dir` include:

- `ph1_<model>_factor_analysis.json`
- `ph2_<model>_combinations.json`
- `best_prompt_fulldata_<model>.json` (when sampling is used)
- `best_prompt_summary.json`
- `factor_axes.json` (when axes are translated/saved)

Notebook visualization exports are saved under:

- `experiments/results/notebook_visualizations/<RUN_TAG>/`

## Notes

- Current analysis defaults focus on IoU=0.5 for ranking/reporting; for scenarios requiring stricter bounding boxes, IoU for mAP calculation can be increased.
- `--map-coco` is optional and adds  `mAP@0.5:0.95` computation; by default, ranking/reporting remains based on mAP@0.5. Enabling this is a good option for punishing loose bounding boxes.
- If no crop text is provided for a phase, default axis definitions are used.

## Package layout

- `agvfm/`: core package (data, config, models, experiments, evaluation, visualization, utils).
- `experiments/scripts/experiments/`: experiment runners (including `load_and_run.py`).
- `experiments/scripts/visualization/`: plotting utilities used by notebook workflows.
- `notebooks/`: interactive analysis and reporting notebooks.
