"""Aggregate structured instrumentation logs into the paper's comparison table.

PAPER.md section 5.3: "Aggregate logs (not raw code) into a single table
format: rows = dataset × method × model, columns = mAP, wall-clock, tokens,
$, GPU-hours, labeled examples ... reproducible directly from the structured
logs in 5.1 — avoid hand-computing any of these numbers." This module is
that aggregation step, reading whatever
:class:`agvfm.instrumentation.tracking.CallRecord` rows a run produced (one
or more ``*.jsonl`` files, anywhere under a results directory tree) and
producing a CSV + a printable table with no hand-editing.

Usage
-----
    # As a library (e.g. from run_full_pipeline.py once all conditions ran)
    from agvfm.reporting.aggregate import load_records, aggregate_records, write_csv
    records = load_records(results_dir)
    rows = aggregate_records(records)
    write_csv(rows, results_dir / "comparison_table.csv")

    # As a CLI
    python -m agvfm.reporting.aggregate --logs-dir experiments/results/template --out-csv comparison_table.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Optional

from agvfm.instrumentation.tracking import RunTracker

logger = logging.getLogger(__name__)

# Group key: every one of these must match for two CallRecord rows to be
# aggregated together. dataset_id/method/model_id are the paper's stated
# grouping (PAPER.md 5.3); prompt_space/held_out_rarity are kept in the key
# too since collapsing across them would blur exactly the comparison the
# paper is trying to make (constrained vs. unconstrained; common vs. rare).
_GROUP_KEYS = ("dataset_id", "method", "model_id", "prompt_space", "held_out_rarity")

# Columns summed across every record in a group.
_SUM_FIELDS = (
    "wall_clock_seconds", "n_api_calls", "tokens_in", "tokens_out",
    "gpu_seconds", "n_labeled_examples_used", "n_unlabeled_images_used",
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_records(path: Path) -> list[dict]:
    """Load every CallRecord row under *path*.

    *path* may be a single ``.jsonl`` file or a directory — in the latter
    case every ``*.jsonl`` file found recursively is read and concatenated,
    so a whole results tree (one file per script/model run) can be pointed
    at directly.
    """
    path = Path(path)
    if path.is_file():
        return RunTracker.read_all(path)

    records: list[dict] = []
    jsonl_files = sorted(path.rglob("*.jsonl"))
    if not jsonl_files:
        logger.warning(f"No .jsonl instrumentation logs found under {path}")
    for f in jsonl_files:
        records.extend(RunTracker.read_all(f))
    logger.info(f"Loaded {len(records)} records from {len(jsonl_files)} log file(s) under {path}")
    return records


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_records(records: list[dict]) -> list[dict]:
    """Group records by (dataset, method, model, prompt_space, rarity) and summarize.

    Returns one row per group with summed cost fields, mean ``map_score``
    (``None`` if no record in the group has one), and ``usd_cost`` summed
    only across records where it isn't ``None`` — with a ``usd_cost_n_priced``
    count alongside so a partially-unpriced group is visible rather than
    silently under-reported as a plain sum.
    """
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        key = tuple(r.get(k, "") for k in _GROUP_KEYS)
        groups.setdefault(key, []).append(r)

    rows: list[dict] = []
    for key, group in sorted(groups.items()):
        row = dict(zip(_GROUP_KEYS, key))
        row["n_runs"] = len(group)

        for field in _SUM_FIELDS:
            row[field] = sum(r.get(field, 0) or 0 for r in group)

        priced = [r["usd_cost"] for r in group if r.get("usd_cost") is not None]
        row["usd_cost"] = sum(priced) if priced else None
        row["usd_cost_n_priced"] = len(priced)
        row["usd_cost_n_unpriced"] = len(group) - len(priced)

        scored = [r["map_score"] for r in group if r.get("map_score") is not None]
        row["map_score_mean"] = sum(scored) / len(scored) if scored else None
        row["map_score_n"] = len(scored)

        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

_CSV_COLUMNS = [
    "dataset_id", "method", "model_id", "prompt_space", "held_out_rarity", "n_runs",
    "map_score_mean", "map_score_n",
    "wall_clock_seconds", "n_api_calls", "tokens_in", "tokens_out",
    "usd_cost", "usd_cost_n_priced", "usd_cost_n_unpriced",
    "gpu_seconds", "n_labeled_examples_used", "n_unlabeled_images_used",
]


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in _CSV_COLUMNS})
    logger.info(f"Wrote comparison table → {out_path}")


def format_table(rows: list[dict]) -> str:
    """Render *rows* as a fixed-width text table for terminal/log output."""
    if not rows:
        return "(no instrumentation records found)"

    header = (
        f"{'Dataset':<28} {'Method':<20} {'Model':<14} {'Space':<13} {'Rarity':<7} "
        f"{'mAP':>7} {'Wall(s)':>9} {'Tok in/out':>12} {'$':>8} {'GPU-h':>7} {'LabelEx':>8} {'UnlabIm':>8}"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        map_str = f"{r['map_score_mean']:.4f}" if r["map_score_mean"] is not None else "—"
        cost_str = f"{r['usd_cost']:.4f}" if r["usd_cost"] is not None else "—"
        gpu_hours = r["gpu_seconds"] / 3600.0
        lines.append(
            f"{r['dataset_id']:<28.28} {r['method']:<20.20} {r['model_id']:<14.14} "
            f"{r['prompt_space']:<13.13} {r['held_out_rarity']:<7.7} "
            f"{map_str:>7} {r['wall_clock_seconds']:>9.1f} "
            f"{r['tokens_in']}/{r['tokens_out']:<10} {cost_str:>8} {gpu_hours:>7.3f} "
            f"{r['n_labeled_examples_used']:>8} {r['n_unlabeled_images_used']:>8}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate agvfm instrumentation logs into the cost/performance comparison table",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--logs-dir", required=True, metavar="PATH",
                   help="Directory (searched recursively for *.jsonl) or a single .jsonl file")
    p.add_argument("--out-csv", default=None, metavar="PATH",
                   help="Where to write the aggregated CSV (default: <logs-dir>/comparison_table.csv)")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    logs_dir = Path(args.logs_dir)
    out_csv = Path(args.out_csv) if args.out_csv else (
        (logs_dir if logs_dir.is_dir() else logs_dir.parent) / "comparison_table.csv"
    )

    records = load_records(logs_dir)
    rows = aggregate_records(records)
    write_csv(rows, out_csv)
    print(format_table(rows))


if __name__ == "__main__":
    main()
