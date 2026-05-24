#!/usr/bin/env python3
"""
Run selected Phase 5 ablations on full dataset sequentially.

Methods:
1. BH-FDR q=0.3
2. BH-FDR q=0.5
3. Fixed p=0.1
4. Top-5
"""

import subprocess
import sys
import time
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

PYTHON_BIN = "/data/apps/conda/jmearles/envs/agvfm/bin/python3"
RESULTS_DIR = project_root / "experiments/results/phase5_semantic_filtering"
LOG_FILE = RESULTS_DIR / "selected_ablations.log"

# Methods to run
METHODS = [
    {
        "name": "BH-FDR q=0.3",
        "args": [
            "--selection-method", "bh_fdr",
            "--q-fdr", "0.3",
            "--results-filename", "semantic_filtering_results_q0.3_full.json",
        ],
    },
    {
        "name": "BH-FDR q=0.5",
        "args": [
            "--selection-method", "bh_fdr",
            "--q-fdr", "0.5",
            "--results-filename", "semantic_filtering_results_q0.5_full.json",
        ],
    },
    {
        "name": "Fixed p=0.1",
        "args": [
            "--selection-method", "fixed_p",
            "--p-cutoff", "0.1",
            "--results-filename", "semantic_filtering_results_p0.1_full.json",
        ],
    },
    {
        "name": "Top-5",
        "args": [
            "--selection-method", "top_k",
            "--top-k", "5",
            "--results-filename", "semantic_filtering_results_top5_full.json",
        ],
    },
]


def run_method(method_info):
    """Run a single method."""
    name = method_info["name"]
    args = method_info["args"]
    
    script_path = project_root / "experiments/scripts/experiments/phase5/run_semantic_filtering.py"
    
    cmd = [
        PYTHON_BIN,
        "-u",  # Unbuffered output
        str(script_path),
        "--output-dir", str(RESULTS_DIR),
    ] + args
    
    print(f"\n{'='*80}")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting: {name}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    # Run with real-time output
    process = subprocess.Popen(
        cmd,
        cwd=project_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    
    # Print output in real-time
    for line in process.stdout:
        print(line, end='', flush=True)
        # Also append to log file
        with open(LOG_FILE, 'a') as f:
            f.write(line)
    
    process.wait()
    elapsed = time.time() - start_time
    
    if process.returncode == 0:
        print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] ✅ {name} complete ({elapsed/60:.1f} minutes)")
        return True
    else:
        print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] ❌ {name} failed (exit code: {process.returncode})")
        return False


def main():
    """Run all selected methods sequentially."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "results").mkdir(parents=True, exist_ok=True)
    
    # Initialize log file (append if exists)
    mode = 'a' if LOG_FILE.exists() else 'w'
    with open(LOG_FILE, mode) as f:
        f.write(f"================================================================================\n")
        f.write(f"Phase 5 Selected Ablations - Full Dataset Run\n")
        f.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"================================================================================\n\n")
    
    print(f"🚀 Starting selected ablations on full dataset")
    print(f"   Log file: {LOG_FILE}")
    print(f"   Methods: {len(METHODS)}")
    print(f"   You can safely disconnect from SSH (process will continue)")
    print()
    
    results = []
    for i, method in enumerate(METHODS, 1):
        # Check if results already exist
        results_file = RESULTS_DIR / "results" / method["args"][method["args"].index("--results-filename") + 1]
        if results_file.exists():
            print(f"\n[{i}/{len(METHODS)}] ⏭️  Skipping {method['name']} (results already exist: {results_file.name})")
            results.append((method['name'], True))
            continue
        
        print(f"\n[{i}/{len(METHODS)}] Running: {method['name']}")
        success = run_method(method)
        results.append((method['name'], success))
        
        if not success:
            print(f"\n⚠️  {method['name']} failed. Continuing with next method...")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ✅ All methods complete!")
    print(f"{'='*80}")
    print("\nSummary:")
    for name, success in results:
        status = "✅" if success else "❌"
        print(f"  {status} {name}")
    
    print(f"\n📊 Results saved to: {RESULTS_DIR}/results/")
    print(f"📈 Generate plots with:")
    print(f"   {PYTHON_BIN} experiments/scripts/experiments/phase5/visualize_comparison.py \\")
    print(f"       --results-dir {RESULTS_DIR}/results --compute-metrics")


if __name__ == "__main__":
    main()
