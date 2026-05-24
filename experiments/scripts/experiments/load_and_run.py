#!/usr/bin/env python3
"""
load_and_run.py — General-purpose Phase 1 / Phase 2 runner for arbitrary crops.

Workflow
--------
1. (Optional) Convert COCO labels to YOLO format if --lbl-dir contains a .json.
2. (Optional) Query Qwen/Qwen3-4B to translate FACTOR_AXES for the target crop,
   returning a JSON array of {name, values, baseline} objects.
   Pass --axes-file to skip the LLM and reuse a previously saved axes file.
   Translated axes are saved to <results-dir>/factor_axes.json.
3. Phase 1 (--run-ph1 CROP):  OFAT factor analysis → ph1_<model>_factor_analysis.json
4. Phase 2 (--run-ph2 CROP):  Combinatorial tests + negation variants on top-N →
                                ph2_<model>_combinations.json

Both result files use the same JSON schema as the existing
run_factor_analysis_hf.py / run_combinations_hf.py outputs.

Usage examples
--------------
# Full pipeline for a new crop with COCO labels:
python load_and_run.py \\
    --run-ph1 "strawberry fruit" --run-ph2 "strawberry fruit" \\
    --model yolo_world \\
    --img-dir /data/strawberry/images \\
    --lbl-dir /data/strawberry/annotations \\
    --results-dir experiments/results/strawberry

# Reuse saved axes (skip LLM):
python load_and_run.py \\
    --run-ph2 "strawberry fruit" \\
    --model yolo_world \\
    --img-dir /data/strawberry/images \\
    --lbl-dir /data/strawberry/labels \\
    --axes-file experiments/results/strawberry/factor_axes.json \\
    --results-dir experiments/results/strawberry

CLI flags
---------
--run-ph1 [CROP]       Run Phase 1 (OFAT); CROP optional — omit to use default axes
--run-ph2 [CROP]       Run Phase 2 (combinations + negation variants); CROP optional
--yolo-weights PATH    Path to YOLO World weights (default: auto-search)
--img-dir PATH         Directory of test images (required when running phases)
--lbl-dir PATH         Directory of labels:
                         .json inside → auto-convert via convert_coco
                         .txt inside  → use as-is (YOLO format already)
--axes-file PATH       Load pre-saved factor_axes.json; skip LLM entirely
--model                yolo_world | grounding_dino | owlv2 | sam3 | all (default: all)
--gdino-model-id       HF model ID for GroundingDINO
--owlv2-model-id       HF model ID for OWLv2
--sam3-model-id        HF model ID for SAM3
--device               cuda | cpu (default: cuda)
--results-dir PATH     Output directory (default: experiments/results/load_and_run/<crop_slug>)
--top-n-negation N     Apply negation variants to top-N Phase 2 prompts (default: 3)
--no-emoji             Skip emoji-axis variants in Phase 1
--no-resume            Start fresh, ignore existing partial results
--sample-size N        Randomly sample N images (seed=42)
--map-coco             Also compute mAP@0.5:0.95 for every prompt (optional; off by default)
--llm-device           Device for the Qwen LLM (default: same as --device)
"""

import argparse
import json
import logging
import os
import random
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Force unbuffered output for cluster logging
# ---------------------------------------------------------------------------
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None
sys.stderr.reconfigure(line_buffering=True) if hasattr(sys.stderr, "reconfigure") else None

# ---------------------------------------------------------------------------
# Path setup — __file__ is
#   experiments/scripts/experiments/load_and_run.py
# parent×4 = project root (AgVFM)
# ---------------------------------------------------------------------------
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from agvfm.config.experiments import (
    FACTOR_AXES as _DEFAULT_FACTOR_AXES,
    build_prompt_from_components,
    generate_factor_combinations,
)
from agvfm.data import get_full_test_paths
from agvfm.experiments import Evaluator
from agvfm.utils import setup_logging, log_experiment_start, log_experiment_complete

# ---------------------------------------------------------------------------
# Local dataclasses (mirrors agvfm.config.experiments; used for LLM output)
# ---------------------------------------------------------------------------

@dataclass
class FactorAxis:
    name: str
    values: List[str]
    baseline: Optional[str] = None


@dataclass
class PromptConfig:
    name: str
    prompt: str
    description: str = ""
    absorber_classes: Optional[List[str]] = None
    target_indices: Optional[List[int]] = None


# ---------------------------------------------------------------------------
# Serialisation helpers  (identical to existing runners)
# ---------------------------------------------------------------------------

def _convert_to_native(obj):
    """Recursively convert numpy scalars/arrays to plain Python types."""
    import numpy as np
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _convert_to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert_to_native(v) for v in obj]
    return obj


def _load_existing(path: Path) -> Dict:
    if path.exists():
        try:
            with open(path) as fh:
                return json.load(fh)
        except (json.JSONDecodeError, ValueError) as exc:
            import shutil
            backup = path.with_suffix(".json.backup")
            print(f"⚠️  Corrupt results file ({exc}); backed up to {backup}")
            shutil.copy2(path, backup)
    return {"results": {}}


def _save(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(_convert_to_native(data), fh, indent=2)


# ---------------------------------------------------------------------------
# COCO → YOLO label conversion
# ---------------------------------------------------------------------------

def _convert_labels_if_needed(lbl_dir: Path) -> Path:
    """
    If *lbl_dir* contains a .json file, run convert_coco and return the
    resulting labels/ sub-directory.  Otherwise return *lbl_dir* unchanged.

    The ultralytics converter outputs to  <cwd>/labels/<json_stem>/
    so we change the working directory to *lbl_dir* first to keep
    converted labels co-located with the annotations.
    """
    json_files = list(lbl_dir.glob("*.json"))
    if not json_files:
        # Already YOLO .txt format
        return lbl_dir

    if len(json_files) > 1:
        print(f"⚠️  Multiple JSON files in {lbl_dir}: {[f.name for f in json_files]}")
        print(f"   Using first: {json_files[0].name}")

    ann_file = json_files[0]
    output_labels_dir = lbl_dir / "coco_converted" /  "labels" / ann_file.stem
    if output_labels_dir.exists() and any(output_labels_dir.glob("*.txt")):
        print(f"✅ Converted labels already exist at {output_labels_dir}")
        return output_labels_dir

    print(f"🔄 Converting COCO labels → YOLO format ...")
    print(f"   Annotation file : {ann_file}")
    from ultralytics.data.converter import convert_coco
    # convert_coco writes to <cwd>/labels/<json_stem>/ — set cwd to lbl_dir
    orig_cwd = os.getcwd()
    try:
        os.chdir(str(lbl_dir))
        convert_coco(
            labels_dir=str(lbl_dir),
            use_segments=False,
            use_keypoints=False,
        )
    finally:
        os.chdir(orig_cwd)

    if not output_labels_dir.exists():
        raise RuntimeError(
            f"convert_coco did not produce expected directory: {output_labels_dir}\n"
            "Check the ultralytics version or inspect the labels/ tree manually."
        )
    print(f"✅ Converted labels at: {output_labels_dir}")
    return output_labels_dir


# ---------------------------------------------------------------------------
# Axes persistence helpers
# ---------------------------------------------------------------------------

def _axes_to_json(axes: List[FactorAxis]) -> List[Dict]:
    return [{"name": ax.name, "values": ax.values, "baseline": ax.baseline} for ax in axes]


def _axes_from_json(data: List[Dict]) -> List[FactorAxis]:
    return [FactorAxis(name=d["name"], values=d["values"], baseline=d.get("baseline")) for d in data]


def _save_axes(axes: List[FactorAxis], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(_axes_to_json(axes), fh, indent=2)
    print(f"💾 Axes saved to: {path}")


def _load_axes(path: Path) -> List[FactorAxis]:
    with open(path) as fh:
        return _axes_from_json(json.load(fh))


# ---------------------------------------------------------------------------
# LLM axis translation (Qwen/Qwen3-4B, JSON output)
# ---------------------------------------------------------------------------

_LLM_SYSTEM_PROMPT = """\
/no_think
You are a botanical prompt engineering assistant.
Your task is to translate a list of FACTOR_AXES originally written for detecting \
cowpea flowers into equivalent axes for a different crop and object.

RULES:
- RETURN ONLY A VALID JSON ARRAY. ABSOLUTELY NO EXPLANATION OR CHAIN OF THOUGHT.
- Each element must be an object with exactly three keys: "name", "values", "baseline".
- Preserve the same axis names (taxonomy, color, size, phenology, negation, anatomy, grammar, emoji).
- Update the values and baseline to match the new crop/object.
- Keep grammar axis unchanged. Follow current standards for baselines.
- For the negation axis, list the most common classes that a detector might confuse with the target.
- For emoji axis, try to provide a set of 3-5 relevant emojis for the target.
"""

_LLM_USER_TEMPLATE = """\
Target crop/object: {crop}

Original FACTOR_AXES (cowpea flower):
{axes_json}

Translate the values and baseline for each axis to suit the target crop/object above.
Return ONLY the JSON array.
"""


def _query_llm_for_axes(
    crop: str,
    original_axes: List[FactorAxis],
    llm_device: str = "cuda",
) -> List[FactorAxis]:
    """
    Query Qwen/Qwen3-4B to translate FACTOR_AXES for *crop*.

    Returns translated axes, or *original_axes* with a warning if parsing fails.
    """
    print(f"\n🤖 Querying Qwen/Qwen3-4B to translate FACTOR_AXES for '{crop}' …")
    try:
        from transformers import pipeline as hf_pipeline
    except ImportError:
        print("⚠️  transformers not installed; skipping LLM translation — using original axes.")
        return original_axes

    axes_json_str = json.dumps(_axes_to_json(original_axes), indent=2)
    user_msg = _LLM_USER_TEMPLATE.format(crop=crop, axes_json=axes_json_str)

    messages = [
        {"role": "system", "content": _LLM_SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]

    try:
        pipe = hf_pipeline(
            "text-generation",
            model="Qwen/Qwen3-4B",
            device_map=llm_device,
            torch_dtype="auto",
        )
        output = pipe(
            messages,
            max_new_tokens=1024,
            temperature=0.1,
            do_sample=True,
            return_full_text=False,
        )
        raw_text: str = output[0]["generated_text"]
    except Exception as exc:
        print(f"⚠️  LLM inference failed ({exc}); using original axes.")
        return original_axes
    finally:
        # Free GPU memory used by the LLM before loading the eval model
        try:
            import torch
            del pipe
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    print(f"   Raw LLM output ({len(raw_text)} chars):")
    print("   " + raw_text[:500].replace("\n", "\n   "))

    # Extract the JSON array — strip any accidental code fences
    json_text = raw_text.strip()
    # Remove ```json ... ``` or ``` ... ``` if present
    json_text = re.sub(r"^```(?:json)?\s*", "", json_text)
    json_text = re.sub(r"\s*```$", "", json_text)
    json_text = json_text.strip()

    # Find the outermost JSON array
    start = json_text.find("[")
    end   = json_text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        print("⚠️  Could not locate JSON array in LLM output; using original axes.")
        return original_axes

    json_text = json_text[start : end + 1]

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as exc:
        print(f"⚠️  JSON parse error ({exc}); using original axes.")
        return original_axes

    # Validate structure
    if not isinstance(data, list) or not data:
        print("⚠️  LLM returned empty or non-list JSON; using original axes.")
        return original_axes

    required_keys = {"name", "values", "baseline"}
    for item in data:
        if not isinstance(item, dict) or not required_keys.issubset(item.keys()):
            print(f"⚠️  Malformed axis entry: {item!r}; using original axes.")
            return original_axes
        if not isinstance(item["values"], list):
            print(f"⚠️  'values' is not a list for axis {item['name']!r}; using original axes.")
            return original_axes

    translated = _axes_from_json(data)
    print(f"✅ Translated {len(translated)} axes for '{crop}'")
    return translated


# ---------------------------------------------------------------------------
# Prompt construction (Phase 2 combinations)
# ---------------------------------------------------------------------------

def _generate_botanical_prompt(
    taxonomy: str,
    grammar: str = "",
    size: str = "",
    color: str = "",
    phenology: str = "",
    anatomy: str = "",
    negation: str = "",
) -> str:
    """
    Construct a botanical detection prompt.

    Order: [grammar] [size] [color] [phenology] [taxonomy] [anatomy], [negation]
    """
    parts = [grammar, size, color, phenology, taxonomy, anatomy]
    main = " ".join(p for p in parts if p).strip()
    if negation:
        return f"{main}, {negation}"
    return main


def _best_axis_value_from_ph1(
    ph1_results: Dict,
    axis_name: str,
    axes: List[FactorAxis],
    fallback: str,
) -> str:
    """
    Scan Phase 1 OFAT results and return the value of *axis_name* that achieved
    the highest mAP@0.5 among all combos that varied *only* that axis away from
    its baseline.  Returns *fallback* if no useful results are found.

    Also considers the baseline combo: if the baseline outperforms every
    non-baseline value the fallback (baseline value) is returned unchanged.
    """
    ax_map = {a.name: a for a in axes}
    if axis_name not in ax_map:
        return fallback

    axis = ax_map[axis_name]
    baseline_vals = {a.name: (a.baseline if a.baseline is not None else "") for a in axes}

    best_val: str = fallback
    best_map: float = -1.0

    for _key, entry in ph1_results.items():
        components = entry.get("components", {})
        if not components:
            continue
        # Check this combo only varies axis_name (or is the pure baseline)
        varies = [
            k for k, v in components.items()
            if v != baseline_vals.get(k, "")
        ]
        # We want: either the baseline (varies=[]) or exactly this axis varies
        if varies and varies != [axis_name]:
            continue

        try:
            map_val = entry["metrics_by_iou"]["iou_0.5"]["map"]
        except (KeyError, TypeError):
            continue

        value = components.get(axis_name, "")
        if map_val > best_map:
            best_map = map_val
            best_val = value

    return best_val


def get_configs(
    axes: List[FactorAxis],
    ph1_results: Optional[Dict] = None,
) -> List[PromptConfig]:
    """
    Generate Phase 2 combinatorial configs from *axes*.

    When *ph1_results* is provided (the ``"results"`` sub-dict from the Phase 1
    output JSON), the best grammar and best taxonomy are derived from the OFAT
    scores rather than using fixed defaults:

    * **best_grammar**: the grammar value that achieved the highest mAP@0.5 in
      Phase 1 OFAT combos that only varied the grammar axis.
    * **best_taxonomy**: the taxonomy value that achieved the highest mAP@0.5
      in Phase 1 OFAT combos that only varied the taxonomy axis.  An empty
      taxonomy (i.e. "no taxonomy") is a valid winner — some crops have less
      standardised naming than cowpea flower.

    Combinations generated:
      1. Colour × Size (at best_taxonomy, best_grammar).
      2. Grammar × Colour × Taxonomy — all grammars × colours against
         best_taxonomy (skips grammar already used as best_grammar).
      3. Anatomy variants with best_color × best_grammar × best_taxonomy.
    """
    ax = {a.name: a for a in axes}
    configs: List[PromptConfig] = []

    # ── Derive best grammar from Phase 1 (fallback: axis baseline or "a") ──
    grammar_axis_baseline = ax["grammar"].baseline or "a"
    if ph1_results:
        grammar_baseline = _best_axis_value_from_ph1(
            ph1_results, "grammar", axes, fallback=grammar_axis_baseline
        )
        print(f"ℹ️  Phase 2 grammar selected from Ph1 OFAT: {grammar_baseline!r}")
    else:
        grammar_baseline = grammar_axis_baseline
        print(f"ℹ️  No Ph1 results — using grammar baseline: {grammar_baseline!r}")

    # ── Derive best taxonomy from Phase 1 (fallback: first axis value) ──
    taxonomy_fallback = ax["taxonomy"].values[0] if ax["taxonomy"].values else ""
    if ph1_results:
        primary_taxonomy = _best_axis_value_from_ph1(
            ph1_results, "taxonomy", axes, fallback=taxonomy_fallback
        )
        print(f"ℹ️  Phase 2 taxonomy selected from Ph1 OFAT: {primary_taxonomy!r}")
    else:
        primary_taxonomy = taxonomy_fallback
        print(f"ℹ️  No Ph1 results — using first taxonomy value: {primary_taxonomy!r}")

    # 1. Colour × Size
    for color in ax["color"].values:
        for size in ax["size"].values:
            if not color and not size:
                continue  # pure baseline handled in OFAT
            prompt = _generate_botanical_prompt(
                grammar=grammar_baseline,
                size=size,
                color=color,
                taxonomy=primary_taxonomy,
            )
            configs.append(PromptConfig(
                name=f"comb_{_slug(size) or 'nosize'}_{_slug(color) or 'noclr'}_spp",
                prompt=prompt,
                description=f"size={size!r} color={color!r} taxonomy={primary_taxonomy!r}",
            ))

    # 2. Best grammar × colour × taxonomy
    for grammar in ax["grammar"].values:
        if grammar == grammar_baseline:
            continue
        for color in ax["color"].values:
            if not color:
                continue
            prompt = _generate_botanical_prompt(
                grammar=grammar,
                color=color,
                taxonomy=primary_taxonomy,
            )
            configs.append(PromptConfig(
                name=f"comb_{_slug(grammar)}_clr_{_slug(color)}_spp",
                prompt=prompt,
                description=f"grammar={grammar!r} color={color!r} taxonomy={primary_taxonomy!r}",
            ))

    # 3. Anatomy variants with best colour + best grammar
    best_color = next((v for v in ax["color"].values if v), "")
    best_grammar = ax["grammar"].values[0] if ax["grammar"].values else grammar_baseline
    for anatomy in ax["anatomy"].values:
        if not anatomy:
            continue
        prompt = _generate_botanical_prompt(
            grammar=best_grammar,
            color=best_color,
            taxonomy=primary_taxonomy,
            anatomy=anatomy,
        )
        configs.append(PromptConfig(
            name=f"comb_clr_spp_anat_{_slug(anatomy)}",
            prompt=prompt,
            description=f"grammar={best_grammar!r} color={best_color!r} taxonomy={primary_taxonomy!r} anatomy={anatomy!r}",
        ))

    # Deduplicate by name (keep first occurrence)
    seen: Dict[str, bool] = {}
    deduped: List[PromptConfig] = []
    for cfg in configs:
        if cfg.name not in seen:
            seen[cfg.name] = True
            deduped.append(cfg)

    return deduped


def _slug(text: str) -> str:
    """Convert a phrase to a compact snake_case slug for config names."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _find_best_prompt(results: Dict) -> Dict:
    """
    Scan a results dict (from Phase 1 or Phase 2) and return the entry with the
    highest mAP@0.5 (101-point interpolated AP at IoU=0.5).

    Returns a dict:
        {
            "name":         <combo/config key>,
            "prompt":       <prompt string>,
            "map_05":       <float>,   # 101-pt mAP@0.5 (primary ranking metric)
            "f1_05":        <float>,   # max-F1 at IoU=0.5
            "best_conf_05": <float>,   # confidence threshold that maximises F1
            "metrics_by_iou": <dict>,  # full metrics blob for the best entry
        }
    or an empty dict if no evaluable results exist.
    """
    best_key: Optional[str] = None
    best_map: float = -1.0

    for key, entry in results.items():
        try:
            m = entry["metrics_by_iou"]["iou_0.5"]["map"]
        except (KeyError, TypeError):
            continue
        if m > best_map:
            best_map = m
            best_key = key

    if best_key is None:
        return {}

    entry = results[best_key]
    prompt_str = (
        entry.get("prompt")
        or entry.get("config_name", best_key)
    )
    iou05 = entry["metrics_by_iou"]["iou_0.5"]
    return {
        "name":            best_key,
        "prompt":          prompt_str,
        "map_05":          iou05["map"],
        "f1_05":           iou05.get("f1", float("nan")),
        "best_conf_05":    iou05.get("best_conf", float("nan")),
        "metrics_by_iou":  entry["metrics_by_iou"],
    }


# ---------------------------------------------------------------------------
# Phase 1: OFAT factor analysis
# ---------------------------------------------------------------------------

def run_phase1(
    model_name: str,
    model,
    images_dir: Path,
    labels_dir: Path,
    results_dir: Path,
    axes: List[FactorAxis],
    iou_thresholds: Optional[List[float]] = None,
    resume: bool = True,
    image_paths: Optional[List[Path]] = None,
    no_emoji: bool = False,
    map_coco_style: bool = False,
) -> Dict:
    """
    Run Phase 1 (OFAT) factor analysis using translated *axes*.

    Produces the same result JSON schema as run_factor_analysis_hf.py.
    """
    if iou_thresholds is None:
        # removed 0.3 from iou threshold for time saving; 
        # only using 0.5 iou for results analysis
        iou_thresholds = [0.5]

    evaluator = Evaluator(
        model=model,
        images_dir=images_dir,
        labels_dir=labels_dir,
        conf_threshold=0.1,
        image_paths=image_paths,
        batch_size=1,
    )

    results_file = results_dir / f"ph1_{model_name}_factor_analysis.json"
    existing = _load_existing(results_file) if resume else {"results": {}}
    n_done = len(existing.get("results", {}))
    if n_done:
        print(f"📂 Loaded {n_done} existing Phase 1 results")

    # Build OFAT combinations using the (possibly translated) axes.
    # generate_factor_combinations() is hardcoded to FACTOR_AXES from the package,
    # so we replicate its logic here against our local axes.
    combinations = _generate_ofat_combinations(axes)

    if no_emoji:
        combinations = [c for c in combinations if not c.get("emoji", "")]
        print("⚙️  --no-emoji: emoji variants excluded from Phase 1.")
    print("=" * 80)
    print(f"Phase 1: OFAT Factor Analysis — {model_name}")
    print("=" * 80)
    print(f"Total combinations : {len(combinations)}")
    print(f"IoU thresholds     : {iou_thresholds}")
    print(f"Images             : {len(evaluator.image_paths)}")
    print(f"Results file       : {results_file}")
    print()

    results = existing.get("results", {})
    logger = logging.getLogger(__name__)

    for idx, combo in enumerate(combinations, 1):
        # Use build_prompt_from_components from agvfm.config.experiments when axes
        # are the default FACTOR_AXES (preserves all special-case logic); otherwise
        # fall back to the simple botanical prompt builder for translated axes.
        prompt = _build_prompt_for_combo(combo, axes)

        combo_name = _combo_name(combo, axes)

        if resume and combo_name in results:
            print(f"[{idx}/{len(combinations)}] {combo_name} — SKIPPED")
            continue

        print(f"[{idx}/{len(combinations)}] {combo_name}")
        print(f"  Prompt: {prompt}")
        logger.info(f"[{idx}/{len(combinations)}] {combo_name} — '{prompt}'")

        try:
            eval_result = evaluator.evaluate_prompt(
                prompt=prompt,
                iou_thresholds=iou_thresholds,
                map_coco_style=map_coco_style,
            )
            results[combo_name] = {"components": combo, "prompt": prompt, **eval_result}

            m05 = eval_result["metrics_by_iou"]["iou_0.5"]
            print(f"  ✓ mAP@0.5: {m05['map']:.4f}  F1: {m05['f1']:.4f}")
            logger.info(f"  {combo_name} — mAP@0.5: {m05['map']:.4f}  F1: {m05['f1']:.4f}")

            _save(results_file, {
                "experiment_type": "factor_analysis",
                "model": model_name,
                "n_combinations": len(combinations),
                "iou_thresholds": iou_thresholds,
                "n_images": len(evaluator.image_paths),
                "results": results,
            })
            print(f"  💾 Saved ({len(results)}/{len(combinations)} done)")

        except Exception as exc:
            print(f"  ❌ Error: {exc}")
            traceback.print_exc()

        print()

    best = _find_best_prompt(results)

    full_results = {
        "experiment_type": "factor_analysis",
        "model": model_name,
        "n_combinations": len(combinations),
        "iou_thresholds": iou_thresholds,
        "n_images": len(evaluator.image_paths),
        "results": results,
        "best_prompt": best,
    }
    _save(results_file, full_results)

    print("=" * 80)
    print(f"✅ Phase 1 complete — {model_name}")
    if best:
        print(f"   🏆 Best prompt : '{best['prompt']}'  [{best['name']}]")
        print(f"      mAP@0.5    : {best['map_05']:.4f}   F1: {best['f1_05']:.4f}   best_conf: {best['best_conf_05']:.3f}")
    print(f"   {results_file}")
    print("=" * 80)
    return full_results


# ---------------------------------------------------------------------------
# Phase 2: Combination tests + negation variants
# ---------------------------------------------------------------------------

def run_phase2(
    model_name: str,
    model,
    images_dir: Path,
    labels_dir: Path,
    results_dir: Path,
    axes: List[FactorAxis],
    iou_thresholds: Optional[List[float]] = None,
    resume: bool = True,
    image_paths: Optional[List[Path]] = None,
    top_n_negation: int = 3,
    no_emoji: bool = False,
    map_coco_style: bool = False,
    ph1_results: Optional[Dict] = None,
) -> Dict:
    """
    Run Phase 2 combinatorial tests and append negation variants to top-N prompts.

    *ph1_results* is the ``"results"`` sub-dict from the Phase 1 output.  When
    provided, ``get_configs`` uses it to select the best grammar and best taxonomy
    from the Phase 1 OFAT scores rather than falling back to axis defaults.

    Produces the same result JSON schema as run_combinations_hf.py.
    """
    if iou_thresholds is None:
        # removed 0.3 from iou threshold for time saving; 
        # only using 0.5 iou for results analysis
        iou_thresholds = [0.5]

    evaluator = Evaluator(
        model=model,
        images_dir=images_dir,
        labels_dir=labels_dir,
        conf_threshold=0.1,
        image_paths=image_paths,
        batch_size=1,
    )

    results_file = results_dir / f"ph2_{model_name}_combinations.json"
    existing = _load_existing(results_file) if resume else {"results": {}}
    n_done = len(existing.get("results", {}))
    if n_done:
        print(f"📂 Loaded {n_done} existing Phase 2 results")

    base_configs = get_configs(axes, ph1_results=ph1_results)

    print("=" * 80)
    print(f"Phase 2: Combination Tests — {model_name}")
    print("=" * 80)
    print(f"Base configurations: {len(base_configs)}")
    print(f"IoU thresholds     : {iou_thresholds}")
    print(f"Images             : {len(evaluator.image_paths)}")
    print(f"Results file       : {results_file}")
    print()

    results = existing.get("results", {})
    logger = logging.getLogger(__name__)

    # ── Step 1: evaluate base combinations ──────────────────────────────────
    for idx, cfg in enumerate(base_configs, 1):
        if resume and cfg.name in results:
            print(f"[{idx}/{len(base_configs)}] {cfg.name} — SKIPPED")
            continue

        print(f"[{idx}/{len(base_configs)}] {cfg.name}")
        print(f"  Prompt      : {cfg.prompt}")
        print(f"  Description : {cfg.description}")
        logger.info(f"[{idx}/{len(base_configs)}] {cfg.name} — '{cfg.prompt}'")

        try:
            eval_result = evaluator.evaluate_prompt(
                prompt=cfg.prompt,
                iou_thresholds=iou_thresholds,
                map_coco_style=map_coco_style,
            )
            eval_result["config_name"]  = cfg.name
            eval_result["description"]  = cfg.description
            results[cfg.name] = eval_result

            m05 = eval_result["metrics_by_iou"]["iou_0.5"]
            print(f"  ✓ mAP@0.5: {m05['map']:.4f}  F1: {m05['f1']:.4f}")
            logger.info(f"  {cfg.name} — mAP@0.5: {m05['map']:.4f}")

            _save(results_file, {
                "experiment_type": "combination_tests",
                "model": model_name,
                "n_configurations": len(base_configs),
                "iou_thresholds": iou_thresholds,
                "n_images": len(evaluator.image_paths),
                "results": results,
            })
            print(f"  💾 Saved ({len(results)} done)")

        except Exception as exc:
            print(f"  ❌ Error: {exc}")
            traceback.print_exc()

        print()

    # ── Step 2: negation variants on top-N ──────────────────────────────────
    neg_axis = next((ax for ax in axes if ax.name == "negation"), None)
    neg_values = [v for v in (neg_axis.values if neg_axis else []) if v]

    if not neg_values:
        print("ℹ️  No negation axis values found — skipping negation variants.")
    else:
        # Sort completed base results by mAP@0.5 descending
        base_keys = [cfg.name for cfg in base_configs]
        scored = [
            (k, results[k]["metrics_by_iou"]["iou_0.5"]["map"])
            for k in base_keys
            if k in results and "metrics_by_iou" in results[k]
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        top_keys = [k for k, _ in scored[:top_n_negation]]

        print(f"\n➕ Negation variants for top {top_n_negation} prompts:")
        for k, score in scored[:top_n_negation]:
            print(f"   {k}  (mAP@0.5={score:.4f})")
        print()

        neg_configs: List[PromptConfig] = []
        for base_key in top_keys:
            base_prompt = results[base_key].get("prompt") or results[base_key].get("config_name", "")
            # Prefer the prompt stored in results; fall back to matching config
            if not base_prompt:
                matched = next((c for c in base_configs if c.name == base_key), None)
                base_prompt = matched.prompt if matched else base_key
            for neg in neg_values:
                neg_name = f"{base_key}_neg_{_slug(neg)}"
                neg_prompt = f"{base_prompt}, {neg}"
                neg_configs.append(PromptConfig(
                    name=neg_name,
                    prompt=neg_prompt,
                    description=f"Negation variant of {base_key!r}: {neg!r}",
                ))

        total_neg = len(neg_configs)
        for idx, cfg in enumerate(neg_configs, 1):
            if resume and cfg.name in results:
                print(f"[neg {idx}/{total_neg}] {cfg.name} — SKIPPED")
                continue

            print(f"[neg {idx}/{total_neg}] {cfg.name}")
            print(f"  Prompt: {cfg.prompt}")
            logger.info(f"[neg {idx}/{total_neg}] {cfg.name} — '{cfg.prompt}'")

            try:
                eval_result = evaluator.evaluate_prompt(
                    prompt=cfg.prompt,
                    iou_thresholds=iou_thresholds,
                    map_coco_style=map_coco_style,
                )
                eval_result["config_name"] = cfg.name
                eval_result["description"] = cfg.description
                results[cfg.name] = eval_result

                m05 = eval_result["metrics_by_iou"]["iou_0.5"]
                print(f"  ✓ mAP@0.5: {m05['map']:.4f}  F1: {m05['f1']:.4f}")
                logger.info(f"  {cfg.name} — mAP@0.5: {m05['map']:.4f}")

                _save(results_file, {
                    "experiment_type": "combination_tests",
                    "model": model_name,
                    "n_configurations": len(base_configs) + total_neg,
                    "iou_thresholds": iou_thresholds,
                    "n_images": len(evaluator.image_paths),
                    "results": results,
                })
                print(f"  💾 Saved ({len(results)} done)")

            except Exception as exc:
                print(f"  ❌ Error: {exc}")
                traceback.print_exc()

            print()

    # ── Step 3: emoji variants on top-N ──────────────────────────────────
    if not no_emoji:
        top_n_emoji = 1 # change as desired: using 1 for this test to see if emoji improves best
        emoji_axis = next((ax for ax in axes if ax.name == "emoji"), None)
        emoji_values = [v for v in (emoji_axis.values if emoji_axis else []) if v]

        if not emoji_values:
            print("ℹ️  No emoji axis values found — skipping emoji variants.")
        else:
            # Sort completed base results AND neg results by mAP@0.5 descending
            base_keys = [cfg.name for cfg in base_configs]
            neg_keys = [cfg.name for cfg in neg_configs]
            all_keys = base_keys + neg_keys
            scored = [
                (k, results[k]["metrics_by_iou"]["iou_0.5"]["map"])
                for k in all_keys
                if k in results and "metrics_by_iou" in results[k]
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            top_keys = [k for k, _ in scored[:top_n_emoji]]
    
            print(f"\n➕ Emoji variants for top {top_n_emoji} prompts:")
            for k, score in scored[:top_n_emoji]:
                print(f"   {k}  (mAP@0.5={score:.4f})")
            print()

            emoji_configs: List[PromptConfig] = []
            for base_key in top_keys:
                base_prompt = results[base_key].get("prompt") or results[base_key].get("config_name", "")
                # Prefer the prompt stored in results; fall back to matching config
                if not base_prompt:
                    matched = next((c for c in base_configs if c.name == base_key), None)
                    base_prompt = matched.prompt if matched else base_key
                for emoji_idx, emoji in enumerate(emoji_values, 1):
                    # Use 1-based index since _slug() strips emoji chars to empty string
                    emoji_name = f"{base_key}_emoji_{emoji_idx}"
                    emoji_prompt = f"{base_prompt} {emoji}"
                    emoji_configs.append(PromptConfig(
                        name=emoji_name,
                        prompt=emoji_prompt,
                        description=f"Emoji variant {emoji_idx} of {base_key!r}: {emoji!r}",
                    ))

            total_emoji = len(emoji_configs)
            for idx, cfg in enumerate(emoji_configs, 1):
                if resume and cfg.name in results:
                    print(f"[emoji {idx}/{total_emoji}] {cfg.name} — SKIPPED")
                    continue

                print(f"[emoji {idx}/{total_emoji}] {cfg.name}")
                print(f"  Prompt: {cfg.prompt}")
                logger.info(f"[emoji {idx}/{total_emoji}] {cfg.name} — '{cfg.prompt}'")

                try:
                    eval_result = evaluator.evaluate_prompt(
                        prompt=cfg.prompt,
                        iou_thresholds=iou_thresholds,
                        map_coco_style=map_coco_style,
                    )
                    eval_result["config_name"] = cfg.name
                    eval_result["description"] = cfg.description
                    results[cfg.name] = eval_result

                    m05 = eval_result["metrics_by_iou"]["iou_0.5"]
                    print(f"  ✓ mAP@0.5: {m05['map']:.4f}  F1: {m05['f1']:.4f}")
                    logger.info(f"  {cfg.name} — mAP@0.5: {m05['map']:.4f}")

                    _save(results_file, {
                        "experiment_type": "combination_tests",
                        "model": model_name,
                        "n_configurations": len(base_configs) + total_emoji + total_neg,
                        "iou_thresholds": iou_thresholds,
                        "n_images": len(evaluator.image_paths),
                        "results": results,
                    })
                    print(f"  💾 Saved ({len(results)} done)")

                except Exception as exc:
                    print(f"  ❌ Error: {exc}")
                    traceback.print_exc()

                print()
    best = _find_best_prompt(results)

    # ── Final save ────────────────────────────────────────────────────────
    full_results = {
        "experiment_type": "combination_tests",
        "model": model_name,
        "n_configurations": len(results),
        "iou_thresholds": iou_thresholds,
        "n_images": len(evaluator.image_paths),
        "results": results,
        "best_prompt": best,
    }
    _save(results_file, full_results)

    print("=" * 80)
    print(f"✅ Phase 2 complete — {model_name}")
    if best:
        print(f"   🏆 Best prompt : '{best['prompt']}'  [{best['name']}]")
        print(f"      mAP@0.5    : {best['map_05']:.4f}   F1: {best['f1_05']:.4f}   best_conf: {best['best_conf_05']:.3f}")
    print(f"   {results_file}")
    print("=" * 80)
    return full_results


# ---------------------------------------------------------------------------
# OFAT helpers (replicate generate_factor_combinations() for arbitrary axes)
# ---------------------------------------------------------------------------

def _generate_ofat_combinations(axes: List[FactorAxis]) -> List[Dict]:
    """Return OFAT combo list for *axes* (mirrors agvfm.config.experiments logic)."""
    baseline = {ax.name: ax.baseline if ax.baseline is not None else "" for ax in axes}
    combos = [baseline.copy()]
    for axis in axes:
        for value in axis.values:
            if value == (axis.baseline if axis.baseline is not None else ""):
                continue
            combo = baseline.copy()
            combo[axis.name] = value
            combos.append(combo)
    return combos


def _are_default_axes(axes: List[FactorAxis]) -> bool:
    """Return True if *axes* are identical to the package FACTOR_AXES."""
    if len(axes) != len(_DEFAULT_FACTOR_AXES):
        return False
    for a, b in zip(axes, _DEFAULT_FACTOR_AXES):
        if a.name != b.name or a.values != list(b.values) or a.baseline != b.baseline:
            return False
    return True


def _build_prompt_for_combo(combo: Dict, axes: List[FactorAxis]) -> str:
    """
    Build a prompt from an OFAT combo dict.

    Uses build_prompt_from_components (with full special-case logic) when axes
    are the default FACTOR_AXES; otherwise falls back to the simple botanical
    prompt builder.
    """
    if _are_default_axes(axes):
        return build_prompt_from_components(combo)
    # Simple path for translated axes
    return _generate_botanical_prompt(
        taxonomy=combo.get("taxonomy", ""),
        grammar=combo.get("grammar", ""),
        size=combo.get("size", ""),
        color=combo.get("color", ""),
        phenology=combo.get("phenology", ""),
        anatomy=combo.get("anatomy", ""),
        negation=combo.get("negation", ""),
    )


def _combo_name(combo: Dict, axes: List[FactorAxis]) -> str:
    """Derive a stable name for an OFAT combo (mirrors run_factor_analysis_hf.py)."""
    is_baseline = all(
        combo.get(ax.name, "") == (ax.baseline if ax.baseline is not None else "")
        for ax in axes
    )
    if is_baseline:
        return "baseline"
    return "_".join(f"{k}:{v}" for k, v in combo.items() if v and v != "")


# ---------------------------------------------------------------------------
# Full-dataset validation helper (used when --sample-size is given)
# ---------------------------------------------------------------------------

def _run_full_dataset_validation(
    model_name: str,
    model,
    best_prompt_info: Dict,
    images_dir: Path,
    labels_dir: Path,
    results_dir: Path,
    iou_thresholds: Optional[List[float]] = None,
    map_coco_style: bool = True,
) -> Dict:
    """
    Re-evaluate the best prompt found during the sampled run on the **full** dataset.

    *best_prompt_info* is the dict returned by _find_best_prompt().
    The model must still be loaded (call this before del model / cuda.empty_cache()).

    Saves results to <results_dir>/best_prompt_fulldata_<model_name>.json and
    returns a summary dict.
    """
    if not best_prompt_info:
        print("⚠️  No best prompt available for full-dataset validation.")
        return {}

    if iou_thresholds is None:
        iou_thresholds = [0.5]

    prompt     = best_prompt_info["prompt"]
    prompt_key = best_prompt_info["name"]

    print()
    print("─" * 80)
    print(f"🔬 Full-dataset validation — {model_name}")
    print(f"   Best prompt (from sample) : '{prompt}'  [{prompt_key}]")
    print(f"   Sample mAP@0.5            : {best_prompt_info['map_05']:.4f}")
    print(f"   Now evaluating on ALL images in {images_dir} …")
    print("─" * 80)

    # Build an Evaluator over the full image set (image_paths=None → uses dir scan)
    evaluator = Evaluator(
        model=model,
        images_dir=images_dir,
        labels_dir=labels_dir,
        conf_threshold=0.1,
        image_paths=None,   # full dataset
        batch_size=1,
    )
    print(f"   Full dataset size: {len(evaluator.image_paths)} images")

    try:
        eval_result = evaluator.evaluate_prompt(
            prompt=prompt,
            iou_thresholds=iou_thresholds,
            map_coco_style=map_coco_style,
        )
        m05 = eval_result["metrics_by_iou"]["iou_0.5"]
        print(f"   ✓ Full-data mAP@0.5 : {m05['map']:.4f}  F1: {m05['f1']:.4f}  best_conf: {m05.get('best_conf', float('nan')):.3f}")
    except Exception as exc:
        print(f"   ❌ Full-dataset evaluation failed: {exc}")
        traceback.print_exc()
        return {}

    summary = {
        "model":              model_name,
        "best_prompt_name":   prompt_key,
        "best_prompt":        prompt,
        "sample_map_05":      best_prompt_info["map_05"],
        "sample_f1_05":       best_prompt_info["f1_05"],
        "sample_best_conf_05": best_prompt_info["best_conf_05"],
        "fulldata_map_05":    m05["map"],
        "fulldata_f1_05":     m05.get("f1", float("nan")),
        "fulldata_best_conf_05": m05.get("best_conf", float("nan")),
        "n_images_full":      len(evaluator.image_paths),
        "iou_thresholds":     iou_thresholds,
        "metrics_by_iou":     eval_result["metrics_by_iou"],
    }

    out_file = results_dir / f"best_prompt_fulldata_{model_name}.json"
    _save(out_file, summary)
    print(f"   💾 Saved full-data result → {out_file}")
    print("─" * 80)
    return summary


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

_ALL_MODELS = ["yolo_world", "grounding_dino", "owlv2", "sam3"]


def _build_single_model(model_key: str, args: argparse.Namespace):
    """Instantiate and return a single (model_name, model) pair."""
    from agvfm.models import YOLOWorldModel, SAM3Model, GroundingDINOModel, OWLv2Model

    device = args.device

    if model_key == "yolo_world":
        weights = args.yolo_weights
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
                raise FileNotFoundError(
                    "YOLO World weights not found. Pass --yolo-weights PATH."
                )
        print(f"Loading YOLO World: {weights} …")
        return "yolo_world", YOLOWorldModel(weights_path=weights, device=device)

    if model_key == "grounding_dino":
        model_id = getattr(args, "gdino_model_id", "IDEA-Research/grounding-dino-base")
        box_thresh = getattr(args, "gdino_box_threshold", 0.3)
        text_thresh = getattr(args, "gdino_text_threshold", 0.25)
        print(f"Loading GroundingDINO: {model_id} …")
        return "grounding_dino", GroundingDINOModel(
            model_id=model_id,
            device=device,
            box_threshold=box_thresh,
            text_threshold=text_thresh,
        )

    if model_key == "owlv2":
        model_id = getattr(args, "owlv2_model_id", "google/owlv2-base-patch16-ensemble")
        print(f"Loading OWLv2: {model_id} …")
        return "owlv2", OWLv2Model(model_id=model_id, device=device)

    if model_key == "sam3":
        model_id = getattr(args, "sam3_model_id", "facebook/sam3")
        print(f"Loading SAM3: {model_id} …")
        return "sam3", SAM3Model(model_id=model_id, device=device)

    raise ValueError(f"Unknown model key: {model_key!r}")


def _resolve_model_keys(args: argparse.Namespace) -> List[str]:
    """Return the ordered list of model keys to run based on --model."""
    if args.model == "all":
        return list(_ALL_MODELS)
    return [args.model]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="General-purpose Phase 1 / Phase 2 OVD runner for arbitrary crops.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--run-ph1", metavar="CROP", nargs="?", const="", default=None,
                   help=(
                       "Run Phase 1 (OFAT factor analysis). "
                       "CROP is optional: if omitted, the default FACTOR_AXES are used "
                       "without LLM translation (same behaviour as --run-ph2 with no CROP)."
                   ))
    p.add_argument("--run-ph2", metavar="CROP", nargs="?", const="", default=None,
                   help=(
                       "Run Phase 2 (combinations + negation). "
                       "CROP is optional: if omitted and --run-ph1 is not active and "
                       "no --axes-file is given, the default FACTOR_AXES are used."
                   ))

    p.add_argument("--model",
                   choices=["yolo_world", "grounding_dino", "owlv2", "sam3", "all"],
                   default="all",
                   help="Model(s) to run. 'all' runs every model sequentially (default).")
    p.add_argument("--yolo-weights", default=None, metavar="PATH",
                   help="Path to YOLO World .pt weights.")
    p.add_argument("--gdino-model-id", default="IDEA-Research/grounding-dino-base")
    p.add_argument("--owlv2-model-id", default="google/owlv2-base-patch16-ensemble")
    p.add_argument("--sam3-model-id", default="facebook/sam3")
    p.add_argument("--gdino-box-threshold", type=float, default=0.3, metavar="T")
    p.add_argument("--gdino-text-threshold", type=float, default=0.25, metavar="T")

    p.add_argument("--img-dir", default=None, metavar="PATH",
                   help="Directory of test images.")
    p.add_argument("--lbl-dir", default=None, metavar="PATH",
                   help="Directory of labels (.json → auto-convert; .txt → use as-is).")

    p.add_argument("--axes-file", default=None, metavar="PATH",
                   help="Path to a saved factor_axes.json; skips LLM translation.")
    p.add_argument("--device", default="cuda",
                   help="Device for the OVD model (cuda / cpu).")
    p.add_argument("--llm-device", default=None,
                   help="Device for the Qwen LLM (default: same as --device).")

    p.add_argument("--results-dir", default=None, metavar="PATH",
                   help="Output directory for results JSON files.")
    p.add_argument("--no-resume", action="store_true",
                   help="Ignore existing partial results and start fresh.")
    p.add_argument("--top-n-negation", type=int, default=3, metavar="N",
                   help="Apply negation variants to the top-N Phase 2 prompts.")
    p.add_argument("--no-emoji", action="store_true",
                   help="Skip emoji-axis variants in Phase 1.")
    p.add_argument("--sample-size", type=int, default=None, metavar="N",
                   help="Randomly sample N images (seed=42).")
    p.add_argument("--map-coco", action="store_true",
                   help=(
                       "Also compute mAP@0.5:0.95 (COCO-style) for every prompt. "
                       "Optional and disabled by default. "
                       "Primary prompt ranking/reporting still uses 101-point AP at IoU=0.5."
                   ))

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # --run-ph1/--run-ph2 with no CROP (value == "") means "use default axes"
    ph1_crop_given = args.run_ph1 not in (None, "")
    ph2_crop_given = args.run_ph2 not in (None, "")

    crop = (args.run_ph1 if ph1_crop_given else None) or (args.run_ph2 if ph2_crop_given else None)

    if args.run_ph1 is None and args.run_ph2 is None:
        print("❌ No phase requested. Pass --run-ph1 [CROP] and/or --run-ph2 [CROP].")
        sys.exit(1)
    if args.img_dir is None or args.lbl_dir is None:
        print("❌ --img-dir and --lbl-dir are required.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Resolve results directory
    # ------------------------------------------------------------------
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        slug = _slug(crop) if crop else "default"
        results_dir = project_root / "experiments" / "results" / "load_and_run" / slug
    results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    log_file = setup_logging(results_dir / "logs", f"{args.model}_load_and_run")
    cfg_summary = {
        "run_ph1": args.run_ph1,
        "run_ph2": args.run_ph2,
        "model": args.model,
        "device": args.device,
        "img_dir": args.img_dir,
        "lbl_dir": args.lbl_dir,
        "axes_file": args.axes_file,
        "results_dir": str(results_dir),
    }
    log_experiment_start("load_and_run", cfg_summary)

    print("=" * 80)
    print("load_and_run.py")
    print("=" * 80)
    print(f"  Phase 1 crop  : {args.run_ph1}")
    print(f"  Phase 2 crop  : {args.run_ph2}")
    print(f"  Model(s)      : {args.model}")
    print(f"  Device        : {args.device}")
    print(f"  Results dir   : {results_dir}")
    print()

    # ------------------------------------------------------------------
    # Label conversion (COCO → YOLO if needed)
    # ------------------------------------------------------------------
    images_dir = Path(args.img_dir)
    lbl_dir    = Path(args.lbl_dir)

    if not images_dir.exists():
        print(f"❌ Images directory not found: {images_dir}")
        sys.exit(1)
    if not lbl_dir.exists():
        print(f"❌ Labels directory not found: {lbl_dir}")
        sys.exit(1)

    labels_dir = _convert_labels_if_needed(lbl_dir)

    # ------------------------------------------------------------------
    # Build image list
    # ------------------------------------------------------------------
    all_image_paths = get_full_test_paths(images_dir)
    n_full_images = len(all_image_paths)
    print(f"📊 Found {n_full_images} images in {images_dir}")

    if args.sample_size is not None:
        if args.sample_size >= n_full_images:
            print(f"⚠️  --sample-size {args.sample_size} ≥ dataset ({n_full_images}); using all.")
            image_paths = all_image_paths
        else:
            rng = random.Random(42)
            image_paths = rng.sample(all_image_paths, args.sample_size)
            print(f"🎲 Sampled {len(image_paths)} images (seed=42)")
    else:
        image_paths = all_image_paths

    # ------------------------------------------------------------------
    # Obtain factor axes
    # Priority:
    #   1. --axes-file explicitly given → load it
    #   2. Saved axes_file already exists in results_dir → reuse it
    #   3. A CROP was supplied (--run-ph1 or --run-ph2) → query LLM, save for reuse
    #   4. No crop given → use default FACTOR_AXES directly (no LLM needed)
    # ------------------------------------------------------------------
    axes_file = results_dir / "factor_axes.json"

    if args.axes_file:
        print(f"\n📂 Loading axes from: {args.axes_file}")
        axes = _load_axes(Path(args.axes_file))
        print(f"   Loaded {len(axes)} axes")
    elif axes_file.exists():
        print(f"\n📂 Reusing existing axes file: {axes_file}")
        axes = _load_axes(axes_file)
        print(f"   Loaded {len(axes)} axes")
    elif crop:
        # A crop name was given (ph1 or ph2) — translate axes via LLM
        llm_device = args.llm_device or args.device
        axes = _query_llm_for_axes(crop, _DEFAULT_FACTOR_AXES, llm_device=llm_device)
        _save_axes(axes, axes_file)
    else:
        # --run-ph1 and/or --run-ph2 used without a CROP: use default axes, no LLM needed
        print("\nℹ️  No crop / axes file provided — using default FACTOR_AXES.")
        axes = list(_DEFAULT_FACTOR_AXES)

    print("\nActive axes:")
    for ax in axes:
        print(f"  {ax.name:12s} baseline={ax.baseline!r:12s} values={ax.values}")
    print()

    # ------------------------------------------------------------------
    # Build evaluation model(s) and run phases — one model at a time
    # to avoid holding multiple large models in GPU memory simultaneously.
    # ------------------------------------------------------------------
    model_keys = _resolve_model_keys(args)
    resume = not args.no_resume
    using_sample = (args.sample_size is not None) and (len(image_paths) < n_full_images)

    print(f"Models to run ({len(model_keys)}): {model_keys}")
    if using_sample:
        print(f"⚠️  Sampling active: phases use {len(image_paths)} images; "
              "the best prompt per model will be re-evaluated on the full dataset.")
    print()

    # Accumulate per-model best-prompt summaries for the final report
    final_report: List[Dict] = []

    for model_idx, model_key in enumerate(model_keys, 1):
        print(f"\n{'=' * 80}")
        print(f"Model {model_idx}/{len(model_keys)}: {model_key}")
        print(f"{'=' * 80}")

        try:
            model_name, model = _build_single_model(model_key, args)
        except Exception as exc:
            print(f"❌ Failed to load {model_key}: {exc}")
            traceback.print_exc()
            print(f"   Skipping {model_key} and continuing with remaining models.")
            continue

        ph1_result: Optional[Dict] = None
        ph2_result: Optional[Dict] = None

        # ── Phase 1 ───────────────────────────────────────────────────
        if args.run_ph1 is not None:
            ph1_result = run_phase1(
                model_name=model_name,
                model=model,
                images_dir=images_dir,
                labels_dir=labels_dir,
                results_dir=results_dir,
                axes=axes,
                resume=resume,
                image_paths=image_paths,
                no_emoji=args.no_emoji,
                map_coco_style=args.map_coco,
            )

        # ── Phase 2 ───────────────────────────────────────────────────
        if args.run_ph2 is not None:
            # Pass Ph1 OFAT results so get_configs can pick the best grammar/taxonomy
            _ph1_results_for_ph2 = (
                ph1_result.get("results") if ph1_result else None
            )
            ph2_result = run_phase2(
                model_name=model_name,
                model=model,
                images_dir=images_dir,
                labels_dir=labels_dir,
                results_dir=results_dir,
                axes=axes,
                resume=resume,
                image_paths=image_paths,
                top_n_negation=args.top_n_negation,
                no_emoji=args.no_emoji,
                map_coco_style=args.map_coco,
                ph1_results=_ph1_results_for_ph2,
            )

        # ── Determine overall best prompt across both phases ───────────
        # Merge result dicts from whichever phases ran, then pick the best.
        merged_results: Dict = {}
        for phase_result in (ph1_result, ph2_result):
            if phase_result and isinstance(phase_result.get("results"), dict):
                merged_results.update(phase_result["results"])

        overall_best = _find_best_prompt(merged_results)

        if overall_best:
            print(f"\n🏆 Overall best for {model_name} (sample):")
            print(f"   Prompt     : '{overall_best['prompt']}'  [{overall_best['name']}]")
            print(f"   mAP@0.5    : {overall_best['map_05']:.4f}   F1: {overall_best['f1_05']:.4f}   best_conf: {overall_best['best_conf_05']:.3f}")

        # ── Full-dataset validation (only when --sample-size was used) ─
        model_summary: Dict = {}
        if using_sample and overall_best:
            model_summary = _run_full_dataset_validation(
                model_name=model_name,
                model=model,
                best_prompt_info=overall_best,
                images_dir=images_dir,
                labels_dir=labels_dir,
                results_dir=results_dir,
                map_coco_style=args.map_coco,
            )
        elif overall_best:
            # No sampling — the phase results are already full-dataset
            model_summary = {
                "model":              model_name,
                "best_prompt_name":   overall_best["name"],
                "best_prompt":        overall_best["prompt"],
                "fulldata_map_05":    overall_best["map_05"],
                "fulldata_f1_05":     overall_best["f1_05"],
                "fulldata_best_conf_05": overall_best["best_conf_05"],
                "n_images_full":      len(image_paths),
                "note":               "full dataset used directly (no --sample-size)",
            }

        if model_summary:
            final_report.append(model_summary)

        # Free GPU memory before loading the next model
        try:
            import torch
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"\n🧹 Freed GPU memory after {model_key}")
        except Exception:
            pass
        print()

    # ------------------------------------------------------------------
    # Final summary report — printed and saved
    # ------------------------------------------------------------------
    log_experiment_complete("load_and_run", results_dir)

    print("=" * 80)
    print("✅ All phases complete!")
    print(f"📊 Results : {results_dir}")
    print(f"📝 Logs    : {log_file}")

    if final_report:
        summary_file = results_dir / "best_prompt_summary.json"
        _save(summary_file, {"models": final_report})

        print()
        print("┌" + "─" * 78 + "┐")
        print("│  BEST PROMPT SUMMARY (full dataset)".ljust(79) + "│")
        print("├" + "─" * 78 + "┤")
        for entry in final_report:
            model_lbl  = entry.get("model", "?")
            prompt_lbl = entry.get("best_prompt", "?")
            map_lbl    = entry.get("fulldata_map_05", float("nan"))
            f1_lbl     = entry.get("fulldata_f1_05", float("nan"))
            conf_lbl   = entry.get("fulldata_best_conf_05", float("nan"))
            note       = entry.get("note", "")
            print(f"│  {model_lbl}")
            print(f"│    Prompt     : {prompt_lbl}")
            print(f"│    mAP@0.5    : {map_lbl:.4f}   F1: {f1_lbl:.4f}   best_conf: {conf_lbl:.3f}" +
                  (f"  ({note})" if note else ""))
            print("├" + "─" * 78 + "┤")
        # Replace the last separator with a closing line
        print("└" + "─" * 78 + "┘")
        print(f"\n📋 Summary saved → {summary_file}")

    print("=" * 80)


if __name__ == "__main__":
    main()
