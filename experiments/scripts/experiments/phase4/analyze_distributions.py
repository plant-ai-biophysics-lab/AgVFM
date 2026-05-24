#!/usr/bin/env python3
"""
Analyze confidence distributions and compare to Phase 3 optimal thresholds.

This script:
1. Loads raw confidence scores collected in Step 1
2. Runs distribution analysis (bimodality, KDE peaks, elbow, gaps)
3. Compares suggested thresholds to Phase 3 optimal thresholds
4. Generates visualizations
5. Saves results
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(project_root))

from agvfm.analysis.confidence_distribution import (
    analyze_confidence_distribution,
    plot_confidence_distribution,
    plot_bimodality_analysis,
)


def load_phase3_optimal_thresholds() -> Dict[str, Dict[str, float]]:
    """Load optimal thresholds from Phase 3 results."""
    phase3_dir = project_root / "experiments" / "results" / "phase3_confidence_sweeps"
    
    optimal_thresholds = {}
    
    # Load YOLO World results
    yolo_file = phase3_dir / "yolo_world_confidence_sweeps.json"
    if not yolo_file.exists():
        print(f"   ⚠️  Phase 3 YOLO file not found: {yolo_file}")
    if yolo_file.exists():
        with open(yolo_file) as f:
            yolo_data = json.load(f)
        
        for config_name, config_data in yolo_data.get("results", {}).items():
            results_by_conf = config_data.get("results_by_conf", {})
            
            # Find threshold with best F1
            best_f1 = -1
            best_threshold = None
            
            for conf_str, metrics in results_by_conf.items():
                f1 = metrics.get("metrics", {}).get("f1", 0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = float(conf_str)
            
            if best_threshold is not None:
                optimal_thresholds[f"yolo_world_{config_name}"] = {
                    "optimal_f1_threshold": best_threshold,
                    "optimal_f1": best_f1,
                }
    
    # Load SAM3 results
    sam3_file = phase3_dir / "sam3_confidence_sweeps.json"
    if sam3_file.exists():
        with open(sam3_file) as f:
            sam3_data = json.load(f)
        
        for config_name, config_data in sam3_data.get("results", {}).items():
            results_by_conf = config_data.get("results_by_conf", {})
            
            # Find threshold with best F1
            best_f1 = -1
            best_threshold = None
            
            for conf_str, metrics in results_by_conf.items():
                f1 = metrics.get("metrics", {}).get("f1", 0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = float(conf_str)
            
            if best_threshold is not None:
                optimal_thresholds[f"sam3_{config_name}"] = {
                    "optimal_f1_threshold": best_threshold,
                    "optimal_f1": best_f1,
                }
    
    return optimal_thresholds


def analyze_all_distributions(
    raw_confidence_dir: Path,
    output_dir: Path,
    compare_to_phase3: bool = True,
) -> Dict:
    """Analyze all raw confidence files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Phase 3 optimal thresholds if available
    optimal_thresholds = {}
    if compare_to_phase3:
        optimal_thresholds = load_phase3_optimal_thresholds()
        print(f"📊 Loaded {len(optimal_thresholds)} optimal thresholds from Phase 3")
    
    # Find all raw confidence files
    raw_files = list(raw_confidence_dir.glob("*_raw_confidence.json"))
    
    if not raw_files:
        print(f"❌ No raw confidence files found in {raw_confidence_dir}")
        return {}
    
    print(f"📁 Found {len(raw_files)} raw confidence files")
    print()
    
    all_results = {}
    
    for raw_file in sorted(raw_files):
        print(f"🔍 Analyzing: {raw_file.name}")
        
        # Load raw confidence data
        with open(raw_file) as f:
            raw_data = json.load(f)
        
        config_name = raw_data["config_name"]
        model = raw_data["model"]
        confidences = np.array(raw_data["confidences"])
        
        print(f"   Config: {config_name}")
        print(f"   Model: {model}")
        print(f"   N detections: {len(confidences):,}")
        
        # Run distribution analysis
        analysis = analyze_confidence_distribution(confidences, config_name=config_name)
        
        # Add model info
        analysis["model"] = model
        
        # Compare to Phase 3 optimal threshold
        key = f"{model}_{config_name}"
        if key in optimal_thresholds:
            optimal = optimal_thresholds[key]
            analysis["phase3_optimal"] = optimal
            
            # Compute differences
            suggested = analysis.get("suggested_thresholds", [])
            if suggested:
                differences = []
                for method, threshold in suggested:
                    diff = abs(threshold - optimal["optimal_f1_threshold"])
                    differences.append({
                        "method": method,
                        "suggested": threshold,
                        "optimal": optimal["optimal_f1_threshold"],
                        "difference": diff,
                        "relative_error": diff / optimal["optimal_f1_threshold"] if optimal["optimal_f1_threshold"] > 0 else None,
                    })
                analysis["threshold_comparison"] = differences
        
        # Generate standard visualization
        plot_path = plots_dir / f"{model}_{config_name}_distribution.png"
        try:
            plot_confidence_distribution(confidences, analysis, save_path=str(plot_path))
            # Use relative path from project root if possible, otherwise absolute
            try:
                analysis["plot_path"] = str(plot_path.relative_to(project_root))
            except ValueError:
                analysis["plot_path"] = str(plot_path)
            print(f"   ✅ Saved standard plot: {plot_path.name}")
        except Exception as e:
            print(f"   ⚠️  Standard plot generation failed: {e}")
            import traceback
            traceback.print_exc()
            analysis["plot_error"] = str(e)
        
        # Generate enhanced bimodality visualization (only if truly bimodal)
        if analysis.get('truly_bimodal', False):
            bimodality_plot_path = plots_dir / f"{model}_{config_name}_bimodality_analysis.png"
            try:
                plot_bimodality_analysis(confidences, analysis, save_path=str(bimodality_plot_path))
                try:
                    analysis["bimodality_plot_path"] = str(bimodality_plot_path.relative_to(project_root))
                except ValueError:
                    analysis["bimodality_plot_path"] = str(bimodality_plot_path)
                print(f"   ✅ Saved bimodality analysis plot: {bimodality_plot_path.name}")
            except Exception as e:
                print(f"   ⚠️  Bimodality plot generation failed: {e}")
                import traceback
                traceback.print_exc()
                analysis["bimodality_plot_error"] = str(e)
        
        # Print summary
        bc = analysis['bimodality_coefficient']
        is_bimodal = analysis.get('is_bimodal', False)
        truly_bimodal = analysis.get('truly_bimodal', False)
        n_peaks = analysis.get('n_peaks_detected', len(analysis.get('kde_peaks', [])))
        
        if truly_bimodal:
            bimodality_status = f"BC={bc:.4f} (truly bimodal, {n_peaks} peaks)"
        elif is_bimodal and n_peaks < 2:
            bimodality_status = f"BC={bc:.4f} (high BC but unimodal - likely highly skewed, {n_peaks} peak)"
        elif is_bimodal:
            bimodality_status = f"BC={bc:.4f} (potentially bimodal, {n_peaks} peaks)"
        else:
            bimodality_status = f"BC={bc:.4f} (unimodal, {n_peaks} peak)"
        
        print(f"   Bimodality: {bimodality_status}")
        if analysis.get("suggested_thresholds"):
            print(f"   Suggested thresholds:")
            for method, threshold in analysis["suggested_thresholds"]:
                print(f"     - {method}: {threshold:.4f}")
        if key in optimal_thresholds:
            optimal = optimal_thresholds[key]
            print(f"   Phase 3 optimal F1 threshold: {optimal['optimal_f1_threshold']:.4f} (F1={optimal['optimal_f1']:.4f})")
        print()
        
        all_results[key] = analysis
    
    # Save combined results
    results_file = output_dir / "distribution_analysis_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"✅ Saved analysis results to {results_file}")
    
    # Generate summary report
    generate_summary_report(all_results, optimal_thresholds, output_dir)
    
    return all_results


def generate_summary_report(
    results: Dict,
    optimal_thresholds: Dict,
    output_dir: Path,
):
    """Generate a text summary report."""
    report_path = output_dir / "summary_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Phase 4: Distribution Analysis Summary\n")
        f.write("=" * 80 + "\n\n")
        
        for key, analysis in sorted(results.items()):
            f.write(f"\n{key}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Model: {analysis['model']}\n")
            f.write(f"Config: {analysis['config_name']}\n")
            f.write(f"N detections: {analysis['statistics']['n']:,}\n")
            f.write(f"\nStatistics:\n")
            f.write(f"  Mean: {analysis['statistics']['mean']:.4f}\n")
            f.write(f"  Median: {analysis['statistics']['median']:.4f}\n")
            f.write(f"  Std: {analysis['statistics']['std']:.4f}\n")
            f.write(f"  Range: [{analysis['statistics']['min']:.4f}, {analysis['statistics']['max']:.4f}]\n")
            
            f.write(f"\nBimodality:\n")
            f.write(f"  BC: {analysis['bimodality_coefficient']:.4f}\n")
            f.write(f"  Is bimodal: {analysis['is_bimodal']}\n")
            
            if analysis.get('kde_peaks'):
                f.write(f"\nKDE Peaks: {analysis['kde_peaks']}\n")
            
            if analysis.get('elbow_point'):
                f.write(f"Elbow point: {analysis['elbow_point']:.4f}\n")
            
            if analysis.get('largest_gap'):
                gap = analysis['largest_gap']
                f.write(f"Largest gap: {gap[0]:.4f} - {gap[1]:.4f}\n")
            
            f.write(f"\nSuggested Thresholds:\n")
            for method, threshold in analysis.get('suggested_thresholds', []):
                f.write(f"  - {method}: {threshold:.4f}\n")
            
            if key in optimal_thresholds:
                optimal = optimal_thresholds[key]
                f.write(f"\nPhase 3 Optimal:\n")
                f.write(f"  F1 threshold: {optimal['optimal_f1_threshold']:.4f}\n")
                f.write(f"  F1 score: {optimal['optimal_f1']:.4f}\n")
                
                if analysis.get('threshold_comparison'):
                    f.write(f"\nComparison:\n")
                    for comp in analysis['threshold_comparison']:
                        f.write(f"  {comp['method']}:\n")
                        f.write(f"    Suggested: {comp['suggested']:.4f}\n")
                        f.write(f"    Optimal: {comp['optimal']:.4f}\n")
                        f.write(f"    Difference: {comp['difference']:.4f}\n")
                        if comp['relative_error']:
                            f.write(f"    Relative error: {comp['relative_error']*100:.1f}%\n")
            
            f.write("\n")
    
    print(f"✅ Saved summary report to {report_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze confidence distributions")
    parser.add_argument(
        "--raw-confidence-dir",
        type=Path,
        default=project_root / "experiments" / "results" / "phase4_unlabeled_threshold_selection",
        help="Directory containing raw confidence JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "experiments" / "results" / "phase4_unlabeled_threshold_selection",
        help="Output directory for analysis results",
    )
    parser.add_argument(
        "--no-phase3-comparison",
        action="store_true",
        help="Skip comparison to Phase 3 optimal thresholds",
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Phase 4: Distribution Analysis")
    print("=" * 80)
    print()
    
    results = analyze_all_distributions(
        args.raw_confidence_dir,
        args.output_dir,
        compare_to_phase3=not args.no_phase3_comparison,
    )
    
    print()
    print("=" * 80)
    print("✅ Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
