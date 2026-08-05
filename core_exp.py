"""Full k-shot selection experiment: sweep beta/gamma/embedder-strategies across
every dataset in config.yaml's ``experiment_datasets`` list, with multi-class
datasets handled by filtering each class's image pool to its own annotations.

Also renders an illustrative (non-decision-making) comparison of embeddings
computed from detection-box crops vs. whole images, once per (dataset, class).

Usage
-----
    # Small smoke test against one already-known-good dataset
    python experiment_run.py --datasets ghai_romaine_detection --k-values 1 3 --proxy-images 5

    # Multi-class dataset — pool is filtered per-class automatically
    python experiment_run.py --datasets tomato_ripeness_detection --k-values 1 3

    # Full sweep across every dataset in experiment_datasets (long-running —
    # run in background; --skip-existing makes it resumable)
    python experiment_run.py --skip-existing

    # Widen the beta/gamma sweep and add a combined-embedder set
    python experiment_run.py --beta-values 0.25 0.5 1.0 --gamma-values 0.25 0.5 1.0 \\
        --embedder-sets clip-vit-base-patch32 clip-vit-base-patch32,dinov2-base

Config additions (optional)
---------------------------
    experiment_datasets:
      - name: apple_detection_usa
        crop: apple
        classes: [apple]
      # ... (see config.yaml for the full list)

    experiment:
      k_values: [1, 3, 5]
      embedder_sets: [[clip-vit-base-patch32]]
      beta_values: [0.5]
      gamma_values: [0.5]
      detection_model: yolo_world
      proxy_images: 20
      crop_diagnostics: true
      crop_score_threshold: 0.1
      seed: 42
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
import re
from pathlib import Path

from agml_prompt.data.loader import load_config, load_dataset
from agml_prompt.models.registry import load_model
from agml_prompt.selection.crops import (
    compare_crop_vs_image_embeddings,
    crop_detections,
    embed_crops_and_images,
    render_embedding_scatter,
)
from agml_prompt.selection.embedders import ImageEmbedder, load_embedder
from agml_prompt.selection.pipeline import PreparedPool, prepare_pool, select_from_pool

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Strategy grid
# ---------------------------------------------------------------------------

def build_strategy_grid(beta_values: list[float], gamma_values: list[float]) -> list[dict]:
    """Build the set of (name, alpha, beta, gamma) strategies to sweep.

    Always includes pure diversity; adds one detection-filter entry per beta,
    one alignment entry per gamma, and one combined entry per (beta, gamma)
    pair. Cost scales with len(beta_values) * len(gamma_values) for the
    combined strategy, so keep those lists short for a full multi-dataset sweep.
    """
    grid = [{"name": "diversity_only", "alpha": 1.0, "beta": 0.0, "gamma": 0.0}]
    for beta in beta_values:
        grid.append({"name": f"diversity_detection_b{beta}", "alpha": 1.0, "beta": beta, "gamma": 0.0})
    for gamma in gamma_values:
        grid.append({"name": f"diversity_alignment_g{gamma}", "alpha": 1.0, "beta": 0.0, "gamma": gamma})
    for beta, gamma in itertools.product(beta_values, gamma_values):
        grid.append({
            "name": f"diversity_detection_alignment_b{beta}_g{gamma}",
            "alpha": 1.0, "beta": beta, "gamma": gamma,
        })
    return grid


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sweep k-shot selection strategies across all experiment datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("config", nargs="?", default="config.yaml")
    p.add_argument("--datasets", nargs="+", metavar="NAME", help="Restrict to these dataset names")
    p.add_argument("--classes", nargs="+", metavar="NAME", help="Restrict to these class names")
    p.add_argument("--k-values", nargs="+", type=int, default=None, dest="k_values")
    p.add_argument("--embedder-sets", nargs="+", default=None, dest="embedder_sets",
                   help="Comma-joined embedder-name combos, e.g. clip-vit-base-patch32,dinov2-base")
    p.add_argument("--combine-method", default="concat", dest="combine_method", choices=["concat", "average"])
    p.add_argument("--beta-values", nargs="+", type=float, default=None, dest="beta_values")
    p.add_argument("--gamma-values", nargs="+", type=float, default=None, dest="gamma_values")
    p.add_argument("--detection-model", default=None, dest="detection_model",
                   help="Model name from registry, or 'none' to disable the detection signal")
    p.add_argument("--detection-mode", default="deviation", dest="detection_mode",
                   choices=["deviation", "low", "high"])
    p.add_argument("--proxy-images", type=int, default=None, dest="proxy_images")
    p.add_argument("--crop-diagnostics", action=argparse.BooleanOptionalAction, default=None,
                   dest="crop_diagnostics")
    p.add_argument("--crop-score-threshold", type=float, default=None, dest="crop_score_threshold")
    p.add_argument("--skip-existing", action="store_true", dest="skip_existing",
                   help="Skip (dataset, class, embedder_set) combos whose output JSON already exists")
    p.add_argument("--output-dir", type=Path, default=Path("experiment_results"), metavar="DIR")
    return p.parse_args()


def _slug(text: str) -> str:
    return re.sub(r"[^\w\-]", "_", text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    exp_cfg = config.get("experiment", {})

    def _get(cli_val, cfg_key, default):
        return cli_val if cli_val is not None else exp_cfg.get(cfg_key, default)

    k_values = _get(args.k_values, "k_values", [1, 3, 5])
    beta_values = _get(args.beta_values, "beta_values", [0.5])
    gamma_values = _get(args.gamma_values, "gamma_values", [0.5])
    proxy_images = _get(args.proxy_images, "proxy_images", 20)
    crop_diagnostics = _get(args.crop_diagnostics, "crop_diagnostics", True)
    crop_score_threshold = _get(args.crop_score_threshold, "crop_score_threshold", 0.1)
    seed = exp_cfg.get("seed", 42)

    embedder_set_names: list[list[str]]
    if args.embedder_sets is not None:
        embedder_set_names = [combo.split(",") for combo in args.embedder_sets]
    else:
        embedder_set_names = exp_cfg.get("embedder_sets", [["clip-vit-base-patch32"]])

    detection_model_name = _get(args.detection_model, "detection_model", "yolo_world")

    strategy_grid = build_strategy_grid(beta_values, gamma_values)
    logger.info(f"Strategy grid ({len(strategy_grid)} entries): {[s['name'] for s in strategy_grid]}")
    logger.info(f"Embedder sets: {embedder_set_names}")

    dataset_cfgs = config.get("experiment_datasets", [])
    if args.datasets:
        dataset_cfgs = [d for d in dataset_cfgs if d["name"] in args.datasets]
        if not dataset_cfgs:
            logger.error(f"No datasets matched filter: {args.datasets}")
            return

    train_split = config.get("optimization", {}).get("train_split", 0.8)
    data_root = config.get("data_root", None)

    embedder_cache: dict[str, ImageEmbedder] = {}

    def _get_embedder(name: str) -> ImageEmbedder:
        if name not in embedder_cache:
            embedder_cache[name] = load_embedder(name)
        return embedder_cache[name]

    detection_model = None
    if detection_model_name and detection_model_name != "none":
        detection_model = load_model(detection_model_name)

    output_dir: Path = args.output_dir
    summary_rows: list[dict] = []

    for ds_cfg in dataset_cfgs:
        logger.info(f"=== Dataset: {ds_cfg['name']} ===")
        try:
            dataset = load_dataset(
                ds_cfg["name"], ds_cfg["classes"],
                train_split=train_split, seed=seed, data_root=data_root,
            )
        except BaseException as exc:
            logger.warning(f"Skipping {ds_cfg['name']}: {type(exc).__name__}: {exc}", exc_info=True)
            continue

        class_names = ds_cfg["classes"]
        if args.classes:
            class_names = [c for c in class_names if c in args.classes]
            if not class_names:
                continue

        for class_name in class_names:
            logger.info(f"--- Class: {class_name} ---")
            class_dir = output_dir / ds_cfg["name"] / _slug(class_name)
            first_prepared: PreparedPool | None = None

            for embedder_names in embedder_set_names:
                embedder_slug = _slug(",".join(embedder_names))
                out_path = class_dir / f"{embedder_slug}.json"
                if args.skip_existing and out_path.exists():
                    logger.info(f"Skipping (exists): {out_path}")
                    continue

                embedders = [_get_embedder(n) for n in embedder_names]
                try:
                    prepared = prepare_pool(
                        dataset, class_name, ds_cfg["crop"], embedders, args.combine_method,
                        proxy_images, seed, detection_model=detection_model,
                    )
                except ValueError as exc:
                    logger.warning(f"Skipping {ds_cfg['name']}/{class_name}/{embedder_names}: {exc}")
                    continue

                if first_prepared is None:
                    first_prepared = prepared

                strategy_results = {}
                for strategy in strategy_grid:
                    for k in k_values:
                        if k > len(prepared.pool):
                            continue
                        if strategy["beta"] > 0 and prepared.detection_counts is None:
                            continue
                        if strategy["gamma"] > 0 and prepared.text_alignment is None:
                            continue
                        r = select_from_pool(
                            prepared, k,
                            alpha=strategy["alpha"], beta=strategy["beta"], gamma=strategy["gamma"],
                            detection_mode=args.detection_mode, seed=seed,
                        )
                        strategy_results.setdefault(strategy["name"], {})[k] = r
                        diag = r["diagnostics"]
                        summary_rows.append({
                            "dataset": ds_cfg["name"], "class": class_name,
                            "embedder_set": ",".join(embedder_names), "strategy": strategy["name"],
                            "alpha": strategy["alpha"], "beta": strategy["beta"], "gamma": strategy["gamma"],
                            "k": k,
                            "mean_pairwise_distance": diag["selected"]["mean_pairwise_distance"],
                            "mean_centroid_distance": diag["selected"]["mean_centroid_distance"],
                            "random_pairwise": diag["random_baseline"]["mean_pairwise_distance"],
                            "random_centroid": diag["random_baseline"]["mean_centroid_distance"],
                            "pairwise_gain": diag["selected"]["mean_pairwise_distance"] - diag["random_baseline"]["mean_pairwise_distance"],
                            "centroid_gain": diag["selected"]["mean_centroid_distance"] - diag["random_baseline"]["mean_centroid_distance"],
                        })

                _save_strategy_results(
                    out_path, ds_cfg["name"], class_name, ds_cfg["crop"], embedder_names,
                    prepared.pool, strategy_results,
                )

            if crop_diagnostics and detection_model is not None and first_prepared is not None:
                _run_crop_diagnostics(
                    class_dir, ds_cfg["name"], class_name, detection_model,
                    _get_embedder(embedder_set_names[0][0]), first_prepared,
                    crop_score_threshold, skip_existing=args.skip_existing,
                )

    _save_summary(output_dir, summary_rows)
    _print_leaderboard(summary_rows)
    logger.info(f"Done. Results written to {output_dir}/")


def _save_strategy_results(
    out_path: Path, dataset_name: str, class_name: str, crop: str,
    embedder_names: list[str], pool, strategy_results: dict,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "dataset": dataset_name,
        "class": class_name,
        "crop": crop,
        "embedders": embedder_names,
        "pool_size": len(pool),
        "strategies": {
            strategy_name: {
                str(k): {
                    "indices": r["indices"],
                    "scores": r["scores"],
                    "diagnostics": r["diagnostics"],
                }
                for k, r in by_k.items()
            }
            for strategy_name, by_k in strategy_results.items()
        },
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved → {out_path}")


def _run_crop_diagnostics(
    class_dir: Path, dataset_name: str, class_name: str, detection_model, embedder,
    prepared: PreparedPool, score_threshold: float, skip_existing: bool,
) -> None:
    json_path = class_dir / "crop_diagnostics.json"
    png_path = class_dir / "crop_diagnostics.png"
    if skip_existing and json_path.exists():
        logger.info(f"Skipping crop diagnostics (exists): {json_path}")
        return

    images = [s.image for s in prepared.pool]
    crops, parent_index = crop_detections(detection_model, images, class_name, score_threshold)
    if not crops:
        logger.info(f"No crops survived score_threshold for {dataset_name}/{class_name} — skipping crop diagnostics")
        return

    image_embeddings, crop_embeddings = embed_crops_and_images(embedder, images, crops)
    text_embedding = embedder.embed_text([class_name])[0] if embedder.supports_text else None
    diagnostics = compare_crop_vs_image_embeddings(image_embeddings, crop_embeddings, parent_index, text_embedding)

    class_dir.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump({"dataset": dataset_name, "class": class_name, **diagnostics}, f, indent=2)
    render_embedding_scatter(image_embeddings, crop_embeddings, png_path)
    logger.info(f"Saved crop diagnostics → {json_path}, {png_path}")


def _save_summary(output_dir: Path, rows: list[dict]) -> None:
    if not rows:
        logger.info("No summary rows to write.")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "summary.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Saved summary → {path}")


def _print_leaderboard(rows: list[dict]) -> None:
    if not rows:
        return
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row["dataset"], row["class"], row["k"])
        groups.setdefault(key, []).append(row)

    sep = "─" * 100
    logger.info(sep)
    logger.info("  LEADERBOARD  (embedding-space heuristic only — not validated against downstream mAP)")
    logger.info(sep)
    for (dataset, class_name, k), group in sorted(groups.items()):
        if len(group) < 2:
            continue

        def _zscore(vals: list[float]) -> list[float]:
            import numpy as np
            arr = np.array(vals, dtype=float)
            std = arr.std()
            return list((arr - arr.mean()) / std) if std > 0 else [0.0] * len(arr)

        pairwise_z = _zscore([r["pairwise_gain"] for r in group])
        centroid_z = _zscore([r["centroid_gain"] for r in group])
        combined = [(p + c) / 2 for p, c in zip(pairwise_z, centroid_z)]
        best_idx = max(range(len(group)), key=lambda i: combined[i])
        best = group[best_idx]
        logger.info(
            f"  {dataset}/{class_name} k={k}: best={best['strategy']} "
            f"(embedders={best['embedder_set']}, score={combined[best_idx]:.3f})"
        )
    logger.info(sep)


if __name__ == "__main__":
    main()
