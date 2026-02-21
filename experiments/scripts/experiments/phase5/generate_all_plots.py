#!/usr/bin/env python3
"""
Generate all plots for Phase 5 results.

This script:
1. Generates individual plots for each result file
2. Generates comparison plots across all methods
"""

import subprocess
import sys
from pathlib import Path

# Add project root to path
script_path = Path(__file__).resolve()
parts = script_path.parts
agvfm_idx = [i for i, part in enumerate(parts) if part == 'AgVFM2']
if agvfm_idx:
    project_root = Path(*parts[:agvfm_idx[0] + 1])
else:
    project_root = script_path.parents[6]
sys.path.insert(0, str(project_root))


def main():
    """Generate all plots."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate all Phase 5 plots")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("experiments/results/phase5_semantic_filtering/results"),
        help="Directory containing result JSON files",
    )
    parser.add_argument(
        "--compute-metrics",
        action="store_true",
        help="Compute F1, Precision, Recall for comparison (slow)",
    )
    parser.add_argument(
        "--skip-individual",
        action="store_true",
        help="Skip individual result file plots",
    )
    
    args = parser.parse_args()
    
    results_dir = project_root / args.results_dir
    plots_dir = results_dir.parent / "plots"
    
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        sys.exit(1)
    
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all result files
    result_files = sorted(results_dir.glob("semantic_filtering_results*.json"))
    
    if not result_files:
        print(f"❌ No result files found in: {results_dir}")
        sys.exit(1)
    
    print(f"📊 Found {len(result_files)} result files\n")
    
    # Generate individual plots
    if not args.skip_individual:
        print("📈 Generating individual plots for each result file...")
        visualize_script = project_root / "experiments/scripts/experiments/phase5/visualize_results.py"
        
        for result_file in result_files:
            print(f"   Processing: {result_file.name}")
            cmd = [
                sys.executable,
                str(visualize_script),
                "--results-file", str(result_file),
                "--output-dir", str(plots_dir),
            ]
            result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"   ⚠️  Error processing {result_file.name}")
                if result.stderr:
                    print(f"      {result.stderr[:200]}")
    
    # Generate comparison plot
    print(f"\n📊 Generating comparison plot...")
    comparison_script = project_root / "experiments/scripts/experiments/phase5/visualize_comparison.py"
    cmd = [
        sys.executable,
        str(comparison_script),
        "--results-dir", str(results_dir),
        "--output-dir", str(plots_dir),
    ]
    if args.compute_metrics:
        cmd.append("--compute-metrics")
    
    result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)
    if result.returncode == 0:
        print("   ✅ Comparison plot generated")
    else:
        print(f"   ⚠️  Error generating comparison plot")
        if result.stderr:
            print(f"      {result.stderr[:200]}")
    
    print(f"\n✅ All plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()
