#!/usr/bin/env python3
"""
Run Phase 5 ablations: q sweep, fixed p cutoff, top-k, global BH.

This script runs multiple experiments with different selection methods
to compare their performance.
"""

import subprocess
import sys
from pathlib import Path

# Add project root to path
# run_ablations.py is at: experiments/scripts/experiments/phase5/run_ablations.py
# Find AgVFM2 directory in path
script_path = Path(__file__).resolve()
parts = script_path.parts
agvfm_idx = [i for i, part in enumerate(parts) if part == 'AgVFM2']
if agvfm_idx:
    project_root = Path(*parts[:agvfm_idx[0] + 1])
else:
    # Fallback: go up 6 levels
    project_root = script_path.parents[6]
sys.path.insert(0, str(project_root))


def run_experiment(
    selection_method: str,
    output_suffix: str,
    q_fdr: float = None,
    p_cutoff: float = None,
    top_k: int = None,
    top_k_p_max: float = None,
    num_images: int = None,
    test_mode: bool = False,
):
    """Run a single experiment with specified parameters."""
    script_path = project_root / "experiments/scripts/experiments/phase5/run_semantic_filtering.py"
    # Use single output directory, different JSON filenames
    output_dir = project_root / "experiments/results/phase5_semantic_filtering"
    results_filename = f"semantic_filtering_results_{output_suffix}.json"
    
    cmd = [
        sys.executable,
        str(script_path),
        "--selection-method", selection_method,
        "--output-dir", str(output_dir),
        "--results-filename", results_filename,
    ]
    
    if q_fdr is not None:
        cmd.extend(["--q-fdr", str(q_fdr)])
    if p_cutoff is not None:
        cmd.extend(["--p-cutoff", str(p_cutoff)])
    if top_k is not None:
        cmd.extend(["--top-k", str(top_k)])
    if top_k_p_max is not None:
        cmd.extend(["--top-k-p-max", str(top_k_p_max)])
    if num_images is not None:
        cmd.extend(["--num-images", str(num_images)])
    if test_mode:
        cmd.append("--test-mode")
    
    print(f"\n{'='*80}")
    print(f"Running: {selection_method} - {output_suffix}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    # Run with unbuffered output for real-time logging
    result = subprocess.run(
        cmd,
        cwd=project_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    
    # Print output in real-time
    if result.stdout:
        print(result.stdout, flush=True)
    if result.stderr:
        print(result.stderr, flush=True)
    
    return result.returncode == 0


def main():
    """Run all ablations."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Phase 5 ablations")
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode (5 images only)",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=None,
        help="Number of images to process (None = all)",
    )
    parser.add_argument(
        "--background",
        action="store_true",
        help="Run in background with nohup (allows SSH disconnect)",
    )
    parser.add_argument(
        "--skip-q-sweep",
        action="store_true",
        help="Skip q sweep experiments",
    )
    parser.add_argument(
        "--skip-fixed-p",
        action="store_true",
        help="Skip fixed p cutoff experiments",
    )
    parser.add_argument(
        "--skip-top-k",
        action="store_true",
        help="Skip top-k experiments",
    )
    parser.add_argument(
        "--skip-global-bh",
        action="store_true",
        help="Skip global BH experiments",
    )
    
    args = parser.parse_args()
    
    # If background mode, wrap the entire script execution
    if args.background:
        log_file = project_root / "experiments/results/phase5_semantic_filtering/ablations.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Re-run this script without --background flag, but with nohup
        import os
        script_path = Path(__file__).resolve()
        cmd = [
            "nohup",
            sys.executable,
            str(script_path),
        ] + [arg for arg in sys.argv[1:] if arg != "--background"]
        
        print(f"🚀 Running in background mode...")
        print(f"   Log file: {log_file}")
        print(f"   Command: {' '.join(cmd)}")
        print(f"   You can safely disconnect from SSH")
        
        with open(log_file, "w") as log:
            process = subprocess.Popen(
                cmd,
                cwd=project_root,
                stdout=log,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,  # Create new process group
            )
        
        print(f"   Process PID: {process.pid}")
        print(f"   Monitor with: tail -f {log_file}")
        return
    
    results = []
    
    # Q sweep (per-image BH-FDR)
    if not args.skip_q_sweep:
        for q in [0.1, 0.2, 0.3, 0.5]:
            success = run_experiment(
                selection_method="bh_fdr",
                output_suffix=f"q{q}",
                q_fdr=q,
                num_images=args.num_images,
                test_mode=args.test_mode,
            )
            results.append(("bh_fdr", f"q{q}", success))
    
    # Fixed p cutoff
    if not args.skip_fixed_p:
        for p_cutoff in [0.05, 0.10]:
            success = run_experiment(
                selection_method="fixed_p",
                output_suffix=f"p{p_cutoff}",
                p_cutoff=p_cutoff,
                num_images=args.num_images,
                test_mode=args.test_mode,
            )
            results.append(("fixed_p", f"p{p_cutoff}", success))
    
    # Top-k
    if not args.skip_top_k:
        for k in [3, 5, 10]:
            # Top-k without p constraint
            success = run_experiment(
                selection_method="top_k",
                output_suffix=f"top{k}",
                top_k=k,
                num_images=args.num_images,
                test_mode=args.test_mode,
            )
            results.append(("top_k", f"top{k}", success))
            
            # Top-k with p <= 0.2 constraint
            success = run_experiment(
                selection_method="top_k",
                output_suffix=f"top{k}_p0.2",
                top_k=k,
                top_k_p_max=0.2,
                num_images=args.num_images,
                test_mode=args.test_mode,
            )
            results.append(("top_k", f"top{k}_p0.2", success))
    
    # Global BH
    if not args.skip_global_bh:
        for q in [0.1, 0.2, 0.3]:
            success = run_experiment(
                selection_method="global_bh",
                output_suffix=f"global_q{q}",
                q_fdr=q,
                num_images=args.num_images,
                test_mode=args.test_mode,
            )
            results.append(("global_bh", f"q{q}", success))
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for method, suffix, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {method:15s} {suffix:20s}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
