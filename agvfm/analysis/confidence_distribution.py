"""
Confidence distribution analysis for unlabeled threshold selection.

This module provides tools for analyzing confidence score distributions
to identify natural thresholds without labeled data.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from scipy import stats
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import seaborn as sns


def compute_bimodality_coefficient(confidences: np.ndarray) -> float:
    """
    Compute bimodality coefficient (BC) for a distribution.
    
    BC = (skewness^2 + 1) / (kurtosis + 3)
    - BC > 0.555 indicates potential bimodality
    - BC < 0.555 indicates unimodality
    
    NOTE: High BC can also result from highly skewed unimodal distributions.
    BC should be used in conjunction with visual inspection and peak detection.
    
    Args:
        confidences: Array of confidence scores
        
    Returns:
        Bimodality coefficient (0-1)
    """
    if len(confidences) < 4:
        return 0.0
    
    skew = stats.skew(confidences)
    kurt = stats.kurtosis(confidences)
    
    # Avoid division by zero
    denominator = kurt + 3
    if abs(denominator) < 1e-10:
        return 0.0
    
    bc = (skew ** 2 + 1) / denominator
    return float(bc)


def find_kde_peaks(
    confidences: np.ndarray,
    bandwidth: Optional[float] = None,
    n_peaks: int = 2,
    min_peak_height_ratio: float = 0.05,
) -> List[float]:
    """
    Find peaks in confidence distribution using kernel density estimation.
    
    Uses a more robust peak detection that filters out noise by requiring
    peaks to be at least min_peak_height_ratio of the maximum density.
    
    Args:
        confidences: Array of confidence scores
        bandwidth: KDE bandwidth (None for automatic)
        n_peaks: Number of peaks to find
        min_peak_height_ratio: Minimum peak height as fraction of max density (default 0.05 = 5%)
        
    Returns:
        List of peak locations (confidence values), sorted by location
    """
    if len(confidences) < 2:
        return []
    
    # Fit KDE
    if bandwidth is None:
        kde = gaussian_kde(confidences)
    else:
        kde = gaussian_kde(confidences, bw_method=bandwidth)
    
    # Evaluate KDE on fine grid
    x = np.linspace(confidences.min(), confidences.max(), 2000)
    density = kde(x)
    
    # Find peaks (local maxima) with minimum height threshold
    from scipy.signal import find_peaks
    min_height = min_peak_height_ratio * density.max()
    peaks, properties = find_peaks(density, height=min_height)
    
    if len(peaks) == 0:
        return []
    
    # Get peak locations and heights
    peak_locations = x[peaks]
    peak_heights = density[peaks]
    
    # Sort by height (descending) and return top n_peaks, then sort by location
    sorted_indices = np.argsort(peak_heights)[::-1]
    top_peaks = peak_locations[sorted_indices[:n_peaks]]
    
    return sorted(top_peaks.tolist())


def find_elbow_point(
    confidences: np.ndarray,
    method: str = "kneedle",
) -> Optional[float]:
    """
    Find elbow/knee point in sorted confidence distribution.
    
    The elbow represents a natural break point where the distribution
    changes slope significantly.
    
    Args:
        confidences: Array of confidence scores
        method: Method to use ("kneedle" or "curvature")
        
    Returns:
        Elbow point (confidence value) or None if not found
    """
    if len(confidences) < 10:
        return None
    
    # Sort confidences in descending order
    sorted_conf = np.sort(confidences)[::-1]
    
    if method == "kneedle":
        # Kneedle algorithm: find point of maximum curvature
        # Simplified version: find point where second derivative is maximum
        n = len(sorted_conf)
        x = np.arange(n)
        
        # Compute first and second derivatives
        dy = np.diff(sorted_conf)
        d2y = np.diff(dy)
        
        if len(d2y) == 0:
            return None
        
        # Find point of maximum curvature (negative second derivative)
        elbow_idx = np.argmax(-d2y) + 1  # +1 because of diff
        
        if elbow_idx >= n:
            elbow_idx = n - 1
        
        return float(sorted_conf[elbow_idx])
    
    elif method == "curvature":
        # Curvature-based method
        n = len(sorted_conf)
        x = np.arange(n)
        
        # Fit polynomial to approximate curve
        # Use cubic spline for smoother derivative
        from scipy.interpolate import UnivariateSpline
        spline = UnivariateSpline(x, sorted_conf, s=0, k=3)
        
        # Compute curvature
        dx = spline.derivative()(x)
        d2x = spline.derivative(n=2)(x)
        curvature = np.abs(d2x) / (1 + dx ** 2) ** 1.5
        
        # Find maximum curvature
        elbow_idx = np.argmax(curvature)
        return float(sorted_conf[elbow_idx])
    
    return None


def find_largest_gap(
    confidences: np.ndarray,
    min_gap_size: float = 0.05,
) -> Optional[Tuple[float, float]]:
    """
    Find largest gap in sorted confidence scores.
    
    Assumes TP and FP modes are separated by a gap.
    
    Args:
        confidences: Array of confidence scores
        min_gap_size: Minimum gap size to consider
        
    Returns:
        Tuple of (gap_start, gap_end) or None if no significant gap
    """
    if len(confidences) < 2:
        return None
    
    sorted_conf = np.sort(confidences)[::-1]  # Descending
    
    # Find gaps between consecutive scores
    gaps = np.diff(sorted_conf)
    gap_indices = np.where(gaps >= min_gap_size)[0]
    
    if len(gap_indices) == 0:
        return None
    
    # Find largest gap
    largest_gap_idx = gap_indices[np.argmax(gaps[gap_indices])]
    gap_start = float(sorted_conf[largest_gap_idx])
    gap_end = float(sorted_conf[largest_gap_idx + 1])
    
    return (gap_start, gap_end)


def analyze_confidence_distribution(
    confidences: np.ndarray,
    config_name: str = "",
) -> Dict:
    """
    Comprehensive analysis of confidence distribution.
    
    Args:
        confidences: Array of confidence scores
        config_name: Name of configuration (for reporting)
        
    Returns:
        Dictionary with analysis results:
        - bimodality_coefficient: float
        - is_bimodal: bool
        - kde_peaks: List[float]
        - elbow_point: Optional[float]
        - largest_gap: Optional[Tuple[float, float]]
        - statistics: Dict with mean, std, min, max, etc.
    """
    if len(confidences) == 0:
        return {
            "config_name": config_name,
            "error": "No confidence scores provided",
        }
    
    # Basic statistics
    stats_dict = {
        "mean": float(np.mean(confidences)),
        "std": float(np.std(confidences)),
        "median": float(np.median(confidences)),
        "min": float(np.min(confidences)),
        "max": float(np.max(confidences)),
        "q25": float(np.percentile(confidences, 25)),
        "q75": float(np.percentile(confidences, 75)),
        "n": len(confidences),
    }
    
    # Bimodality
    bc = compute_bimodality_coefficient(confidences)
    # BC > 0.555 suggests potential bimodality, but high BC can also result from
    # highly skewed unimodal distributions. We require both high BC AND multiple
    # detected peaks to claim bimodality.
    is_bimodal = bc > 0.555
    
    # KDE peaks (use 5% threshold to filter out noise)
    kde_peaks = find_kde_peaks(confidences, min_peak_height_ratio=0.05)
    
    # Elbow point
    elbow_point = find_elbow_point(confidences)
    
    # Largest gap
    largest_gap = find_largest_gap(confidences)
    
    # Suggested threshold (combine methods)
    suggested_thresholds = []
    
    if len(kde_peaks) >= 2:
        # Use valley between two peaks
        valley = (kde_peaks[0] + kde_peaks[1]) / 2
        suggested_thresholds.append(("kde_valley", valley))
    
    if elbow_point is not None:
        suggested_thresholds.append(("elbow", elbow_point))
    
    if largest_gap is not None:
        # Use midpoint of gap
        gap_mid = (largest_gap[0] + largest_gap[1]) / 2
        suggested_thresholds.append(("gap_midpoint", gap_mid))
    
    # Determine if truly bimodal (requires both high BC and multiple peaks)
    truly_bimodal = is_bimodal and len(kde_peaks) >= 2
    
    return {
        "config_name": config_name,
        "bimodality_coefficient": bc,
        "is_bimodal": is_bimodal,  # Based on BC alone
        "truly_bimodal": truly_bimodal,  # Requires both BC and peak detection
        "kde_peaks": kde_peaks,
        "n_peaks_detected": len(kde_peaks),
        "elbow_point": elbow_point,
        "largest_gap": largest_gap,
        "suggested_thresholds": suggested_thresholds,
        "statistics": stats_dict,
    }


def plot_confidence_distribution(
    confidences: np.ndarray,
    analysis: Dict,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 8),
) -> plt.Figure:
    """
    Plot confidence distribution with analysis annotations.
    
    Args:
        confidences: Array of confidence scores
        analysis: Results from analyze_confidence_distribution()
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f"Confidence Distribution Analysis: {analysis.get('config_name', 'Unknown')}", 
                 fontsize=14, fontweight='bold')
    
    # 1. Histogram with KDE
    ax = axes[0, 0]
    ax.hist(confidences, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Overlay KDE
    kde = gaussian_kde(confidences)
    x = np.linspace(confidences.min(), confidences.max(), 1000)
    ax.plot(x, kde(x), 'r-', linewidth=2, label='KDE')
    
    # Mark peaks
    peaks = analysis.get('kde_peaks', [])
    for i, peak in enumerate(peaks):
        ax.axvline(peak, color='green', linestyle='--', linewidth=2, 
                   label=f'Peak {i+1}' if i == 0 else '')
        ax.plot(peak, kde(peak), 'go', markersize=10)
    
    ax.set_xlabel('Confidence Score')
    ax.set_ylabel('Density')
    ax.set_title('Histogram with KDE')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Cumulative distribution
    ax = axes[0, 1]
    sorted_conf = np.sort(confidences)[::-1]
    ax.plot(sorted_conf, np.arange(len(sorted_conf)) / len(sorted_conf), 
            linewidth=2, color='blue')
    
    # Mark elbow
    elbow = analysis.get('elbow_point')
    if elbow is not None:
        elbow_idx = np.searchsorted(sorted_conf[::-1], elbow)
        ax.axvline(elbow, color='red', linestyle='--', linewidth=2, label='Elbow')
        ax.plot(elbow, elbow_idx / len(sorted_conf), 'ro', markersize=10)
    
    ax.set_xlabel('Confidence Score')
    ax.set_ylabel('Cumulative Fraction')
    ax.set_title('Cumulative Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Box plot
    ax = axes[1, 0]
    ax.boxplot(confidences, vert=True, patch_artist=True,
               boxprops=dict(facecolor='lightblue', alpha=0.7))
    ax.set_ylabel('Confidence Score')
    ax.set_title('Box Plot')
    ax.grid(True, alpha=0.3)
    
    # 4. Statistics text
    ax = axes[1, 1]
    ax.axis('off')
    
    stats_text = f"""
    Statistics:
    - Mean: {analysis['statistics']['mean']:.4f}
    - Median: {analysis['statistics']['median']:.4f}
    - Std: {analysis['statistics']['std']:.4f}
    - Min: {analysis['statistics']['min']:.4f}
    - Max: {analysis['statistics']['max']:.4f}
    - Q25: {analysis['statistics']['q25']:.4f}
    - Q75: {analysis['statistics']['q75']:.4f}
    - N: {analysis['statistics']['n']}
    
    Bimodality:
    - BC: {analysis['bimodality_coefficient']:.4f}
    - Bimodal: {analysis['is_bimodal']}
    
    Suggested Thresholds:
    """
    
    for method, threshold in analysis.get('suggested_thresholds', []):
        stats_text += f"  - {method}: {threshold:.4f}\n"
    
    if analysis.get('largest_gap'):
        gap = analysis['largest_gap']
        stats_text += f"\nLargest Gap: {gap[0]:.4f} - {gap[1]:.4f}"
    
    ax.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
            verticalalignment='center', transform=ax.transAxes)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_bimodality_analysis(
    confidences: np.ndarray,
    analysis: Dict,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (16, 10),
) -> plt.Figure:
    """
    Enhanced visualization focusing on bimodality with zoomed views.
    
    Creates a comprehensive figure showing:
    1. Full distribution with KDE and peak/valley annotations
    2. Zoomed view around the smaller second peak
    3. Separate mode distributions (if bimodal)
    4. KDE valley location clearly marked
    
    Args:
        confidences: Array of confidence scores
        analysis: Results from analyze_confidence_distribution()
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    config_name = analysis.get('config_name', 'Unknown')
    model = analysis.get('model', '')
    bc = analysis.get('bimodality_coefficient', 0.0)
    is_bimodal = analysis.get('is_bimodal', False)
    peaks = analysis.get('kde_peaks', [])
    
    # Fit KDE
    kde = gaussian_kde(confidences)
    x_full = np.linspace(confidences.min(), confidences.max(), 2000)
    density_full = kde(x_full)
    
    # Find valley between peaks if we have 2+ peaks
    valley = None
    valley_idx = None
    if len(peaks) >= 2:
        # Find minimum between the two peaks
        peak1, peak2 = sorted(peaks)[0], sorted(peaks)[1]
        mask = (x_full >= peak1) & (x_full <= peak2)
        if np.any(mask):
            valley_idx = np.argmin(density_full[mask]) + np.where(mask)[0][0]
            valley = x_full[valley_idx]
            valley_density = density_full[valley_idx]
    
    # Main title
    title = f"Distribution Analysis: {model} - {config_name}"
    truly_bimodal = analysis.get('truly_bimodal', False)
    n_peaks = analysis.get('n_peaks_detected', len(peaks))
    
    if truly_bimodal:
        title += f" (BC={bc:.3f}, Bimodal ✓, {n_peaks} peaks)"
    elif is_bimodal and n_peaks < 2:
        title += f" (BC={bc:.3f}, High BC but Unimodal - likely highly skewed)"
    elif is_bimodal:
        title += f" (BC={bc:.3f}, Potentially Bimodal, {n_peaks} peaks)"
    else:
        title += f" (BC={bc:.3f}, Unimodal)"
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    # 1. Full distribution with KDE (top, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :])
    
    # Histogram
    n_bins = min(100, max(30, len(confidences) // 100))
    counts, bins, patches = ax1.hist(confidences, bins=n_bins, density=True, 
                                     alpha=0.5, color='lightblue', edgecolor='black', linewidth=0.5)
    
    # KDE overlay
    ax1.plot(x_full, density_full, 'r-', linewidth=3, label='KDE', zorder=5)
    
    # Mark peaks with enhanced visibility
    peak_colors = ['#2E7D32', '#C62828']  # Green for peak 1, Red for peak 2
    for i, peak in enumerate(peaks):
        peak_density_arr = kde(peak)
        peak_density = float(peak_density_arr.item() if hasattr(peak_density_arr, 'item') else peak_density_arr)
        ax1.axvline(peak, color=peak_colors[i], linestyle='--', linewidth=2.5, 
                   alpha=0.8, zorder=6)
        ax1.plot(peak, peak_density, 'o', color=peak_colors[i], 
                markersize=12, markeredgecolor='white', markeredgewidth=2, zorder=7)
        ax1.annotate(f'Peak {i+1}\n{peak:.3f}', 
                    xy=(peak, peak_density),
                    xytext=(10 if i == 0 else -10, 20),
                    textcoords='offset points',
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor=peak_colors[i], 
                             alpha=0.3, edgecolor=peak_colors[i], linewidth=2),
                    arrowprops=dict(arrowstyle='->', color=peak_colors[i], lw=2),
                    zorder=8)
    
    # Mark valley
    if valley is not None:
        ax1.axvline(valley, color='orange', linestyle=':', linewidth=3, 
                   alpha=0.9, zorder=6, label='KDE Valley (Suggested Threshold)')
        ax1.plot(valley, valley_density, 's', color='orange', 
                markersize=14, markeredgecolor='white', markeredgewidth=2, zorder=7)
        ax1.annotate(f'Valley\n{valley:.3f}', 
                    xy=(valley, valley_density),
                    xytext=(0, -40),
                    textcoords='offset points',
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='orange', 
                             alpha=0.4, edgecolor='orange', linewidth=2),
                    arrowprops=dict(arrowstyle='->', color='orange', lw=2),
                    ha='center', zorder=8)
    
    # Mark Phase 3 optimal if available
    if 'phase3_optimal' in analysis:
        optimal = analysis['phase3_optimal']['optimal_f1_threshold']
        optimal_density_arr = kde(optimal)
        optimal_density = float(optimal_density_arr.item() if hasattr(optimal_density_arr, 'item') else optimal_density_arr)
        ax1.axvline(optimal, color='purple', linestyle='-.', linewidth=2, 
                   alpha=0.7, zorder=5, label='Phase 3 Optimal F1')
        ax1.plot(optimal, optimal_density, '^', color='purple', 
                markersize=10, markeredgecolor='white', markeredgewidth=1.5, zorder=6)
    
    ax1.set_xlabel('Confidence Score', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax1.set_title('Full Distribution with KDE and Peaks', fontsize=13, fontweight='bold', pad=10)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # 2. Zoom around smaller second peak (top right) - only if truly bimodal
    if len(peaks) >= 2 and truly_bimodal:
        # Identify smaller peak (by density height)
        peak1_density_arr = kde(peaks[0])
        peak2_density_arr = kde(peaks[1])
        peak1_density = float(peak1_density_arr.item() if hasattr(peak1_density_arr, 'item') else peak1_density_arr)
        peak2_density = float(peak2_density_arr.item() if hasattr(peak2_density_arr, 'item') else peak2_density_arr)
        
        if peak1_density < peak2_density:
            smaller_peak = peaks[0]
            larger_peak = peaks[1]
            smaller_idx = 0
        else:
            smaller_peak = peaks[1]
            larger_peak = peaks[0]
            smaller_idx = 1
        
        ax2 = fig.add_subplot(gs[1, 0])
        
        # Zoom window: ±0.15 around smaller peak, or ±20% of range
        zoom_range = max(0.15, (confidences.max() - confidences.min()) * 0.2)
        x_zoom = np.linspace(max(confidences.min(), smaller_peak - zoom_range),
                            min(confidences.max(), smaller_peak + zoom_range), 1000)
        density_zoom = kde(x_zoom)
        
        # Histogram in zoom range
        mask_zoom = (confidences >= x_zoom.min()) & (confidences <= x_zoom.max())
        if np.any(mask_zoom):
            ax2.hist(confidences[mask_zoom], bins=50, density=True, 
                    alpha=0.6, color='lightcoral', edgecolor='black', linewidth=0.5)
        
        # KDE in zoom range
        ax2.plot(x_zoom, density_zoom, 'r-', linewidth=3, label='KDE', zorder=5)
        
        # Mark smaller peak
        smaller_peak_density_arr = kde(smaller_peak)
        smaller_peak_density = float(smaller_peak_density_arr.item() if hasattr(smaller_peak_density_arr, 'item') else smaller_peak_density_arr)
        ax2.axvline(smaller_peak, color=peak_colors[smaller_idx], linestyle='--', 
                    linewidth=2.5, alpha=0.8, zorder=6)
        ax2.plot(smaller_peak, smaller_peak_density, 'o', 
                color=peak_colors[smaller_idx], markersize=14, 
                markeredgecolor='white', markeredgewidth=2, zorder=7)
        ax2.annotate(f'Peak {smaller_idx+1}\n{smaller_peak:.3f}', 
                    xy=(smaller_peak, smaller_peak_density),
                    xytext=(10, 15),
                    textcoords='offset points',
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.4', 
                             facecolor=peak_colors[smaller_idx], alpha=0.3,
                             edgecolor=peak_colors[smaller_idx], linewidth=2),
                    arrowprops=dict(arrowstyle='->', color=peak_colors[smaller_idx], lw=2),
                    zorder=8)
        
        # Mark valley if in range
        if valley is not None and x_zoom.min() <= valley <= x_zoom.max():
            ax2.axvline(valley, color='orange', linestyle=':', linewidth=3, 
                       alpha=0.9, zorder=6, label='Valley')
            ax2.plot(valley, valley_density, 's', color='orange', 
                    markersize=12, markeredgecolor='white', markeredgewidth=2, zorder=7)
        
        ax2.set_xlabel('Confidence Score', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Density', fontsize=11, fontweight='bold')
        ax2.set_title(f'Zoom: Peak {smaller_idx+1} (Smaller Peak)', 
                     fontsize=12, fontweight='bold', pad=8)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3, linestyle='--')
    
    # 3. Zoom around larger peak (top middle) - only if truly bimodal
    if len(peaks) >= 2 and truly_bimodal:
        ax3 = fig.add_subplot(gs[1, 1])
        
        larger_peak = larger_peak if len(peaks) >= 2 else peaks[0]
        larger_idx = 1 - smaller_idx if len(peaks) >= 2 else 0
        
        zoom_range = max(0.15, (confidences.max() - confidences.min()) * 0.2)
        x_zoom = np.linspace(max(confidences.min(), larger_peak - zoom_range),
                            min(confidences.max(), larger_peak + zoom_range), 1000)
        density_zoom = kde(x_zoom)
        
        mask_zoom = (confidences >= x_zoom.min()) & (confidences <= x_zoom.max())
        if np.any(mask_zoom):
            ax3.hist(confidences[mask_zoom], bins=50, density=True, 
                    alpha=0.6, color='lightgreen', edgecolor='black', linewidth=0.5)
        
        ax3.plot(x_zoom, density_zoom, 'r-', linewidth=3, label='KDE', zorder=5)
        
        larger_peak_density_arr = kde(larger_peak)
        larger_peak_density = float(larger_peak_density_arr.item() if hasattr(larger_peak_density_arr, 'item') else larger_peak_density_arr)
        ax3.axvline(larger_peak, color=peak_colors[larger_idx], linestyle='--', 
                   linewidth=2.5, alpha=0.8, zorder=6)
        ax3.plot(larger_peak, larger_peak_density, 'o', 
                color=peak_colors[larger_idx], markersize=14, 
                markeredgecolor='white', markeredgewidth=2, zorder=7)
        ax3.annotate(f'Peak {larger_idx+1}\n{larger_peak:.3f}', 
                    xy=(larger_peak, larger_peak_density),
                    xytext=(10, 15),
                    textcoords='offset points',
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.4', 
                             facecolor=peak_colors[larger_idx], alpha=0.3,
                             edgecolor=peak_colors[larger_idx], linewidth=2),
                    arrowprops=dict(arrowstyle='->', color=peak_colors[larger_idx], lw=2),
                    zorder=8)
        
        if valley is not None and x_zoom.min() <= valley <= x_zoom.max():
            ax3.axvline(valley, color='orange', linestyle=':', linewidth=3, 
                       alpha=0.9, zorder=6, label='Valley')
            ax3.plot(valley, valley_density, 's', color='orange', 
                    markersize=12, markeredgecolor='white', markeredgewidth=2, zorder=7)
        
        ax3.set_xlabel('Confidence Score', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Density', fontsize=11, fontweight='bold')
        ax3.set_title(f'Zoom: Peak {larger_idx+1} (Larger Peak)', 
                     fontsize=12, fontweight='bold', pad=8)
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3, linestyle='--')
    
    # 4. Zoom around valley (top right)
    if valley is not None:
        ax4 = fig.add_subplot(gs[1, 2])
        
        zoom_range = max(0.1, (confidences.max() - confidences.min()) * 0.15)
        x_zoom = np.linspace(max(confidences.min(), valley - zoom_range),
                            min(confidences.max(), valley + zoom_range), 1000)
        density_zoom = kde(x_zoom)
        
        mask_zoom = (confidences >= x_zoom.min()) & (confidences <= x_zoom.max())
        if np.any(mask_zoom):
            ax4.hist(confidences[mask_zoom], bins=50, density=True, 
                    alpha=0.6, color='wheat', edgecolor='black', linewidth=0.5)
        
        ax4.plot(x_zoom, density_zoom, 'r-', linewidth=3, label='KDE', zorder=5)
        
        # Mark valley prominently
        ax4.axvline(valley, color='orange', linestyle=':', linewidth=4, 
                   alpha=0.9, zorder=6, label='KDE Valley')
        ax4.plot(valley, valley_density, 's', color='orange', 
                markersize=16, markeredgecolor='white', markeredgewidth=3, zorder=7)
        ax4.annotate(f'Valley\n{valley:.3f}\n(Suggested Threshold)', 
                    xy=(valley, valley_density),
                    xytext=(0, 25),
                    textcoords='offset points',
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='orange', 
                             alpha=0.5, edgecolor='orange', linewidth=3),
                    arrowprops=dict(arrowstyle='->', color='orange', lw=3),
                    ha='center', zorder=8)
        
        # Mark peaks if in range
        for i, peak in enumerate(peaks):
            if x_zoom.min() <= peak <= x_zoom.max():
                peak_density_arr = kde(peak)
                peak_density = float(peak_density_arr.item() if hasattr(peak_density_arr, 'item') else peak_density_arr)
                ax4.axvline(peak, color=peak_colors[i], linestyle='--', 
                           linewidth=2, alpha=0.6, zorder=5)
                ax4.plot(peak, peak_density, 'o', color=peak_colors[i], 
                        markersize=10, markeredgecolor='white', markeredgewidth=1.5, zorder=6)
        
        ax4.set_xlabel('Confidence Score', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Density', fontsize=11, fontweight='bold')
        ax4.set_title('Zoom: KDE Valley (Threshold Region)', 
                     fontsize=12, fontweight='bold', pad=8)
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3, linestyle='--')
    
    # 5. Separate mode distributions (bottom row, 3 columns) - only if truly bimodal
    if len(peaks) >= 2 and valley is not None and truly_bimodal:
        # Split data at valley
        low_mode = confidences[confidences < valley]
        high_mode = confidences[confidences >= valley]
        
        ax5 = fig.add_subplot(gs[2, 0])
        if len(low_mode) > 0:
            ax5.hist(low_mode, bins=50, density=True, alpha=0.7, 
                    color='#FF6B6B', edgecolor='black', linewidth=0.5, label='Low Mode')
            kde_low = gaussian_kde(low_mode)
            x_low = np.linspace(low_mode.min(), low_mode.max(), 500)
            ax5.plot(x_low, kde_low(x_low), 'r-', linewidth=2.5)
        ax5.set_xlabel('Confidence Score', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Density', fontsize=11, fontweight='bold')
        ax5.set_title(f'Low Mode (Conf < {valley:.3f})\nN={len(low_mode):,}', 
                     fontsize=11, fontweight='bold')
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.3, linestyle='--')
        
        ax6 = fig.add_subplot(gs[2, 1])
        if len(high_mode) > 0:
            ax6.hist(high_mode, bins=50, density=True, alpha=0.7, 
                    color='#4ECDC4', edgecolor='black', linewidth=0.5, label='High Mode')
            kde_high = gaussian_kde(high_mode)
            x_high = np.linspace(high_mode.min(), high_mode.max(), 500)
            ax6.plot(x_high, kde_high(x_high), 'b-', linewidth=2.5)
        ax6.set_xlabel('Confidence Score', fontsize=11, fontweight='bold')
        ax6.set_ylabel('Density', fontsize=11, fontweight='bold')
        ax6.set_title(f'High Mode (Conf ≥ {valley:.3f})\nN={len(high_mode):,}', 
                     fontsize=11, fontweight='bold')
        ax6.legend(fontsize=9)
        ax6.grid(True, alpha=0.3, linestyle='--')
        
        # Statistics comparison
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis('off')
        
        stats_text = f"""
DISTRIBUTION ANALYSIS

Bimodality Coefficient: {bc:.4f}
BC Threshold (>0.555): {'PASS' if is_bimodal else 'FAIL'}
Peaks Detected (5% threshold): {n_peaks}
Status: {'TRULY BIMODAL ✓' if truly_bimodal else ('High BC but Unimodal (likely skewed)' if is_bimodal and n_peaks < 2 else 'Unimodal')}

PEAK LOCATIONS:
"""
        if len(peaks) > 0:
            for i, peak in enumerate(peaks):
                peak_density_arr = kde(peak)
                peak_density_val = float(peak_density_arr.item() if hasattr(peak_density_arr, 'item') else peak_density_arr)
                stats_text += f"  Peak {i+1}: {peak:.4f}\n"
                stats_text += f"    (Density: {peak_density_val:.4f})\n"
        else:
            stats_text += "  No significant peaks detected\n"
        
        if valley is not None and truly_bimodal:
            stats_text += f"\nKDE VALLEY: {valley:.4f}\n"
            stats_text += f"(Suggested Threshold)\n"
        elif is_bimodal and n_peaks < 2:
            stats_text += f"\nNOTE: High BC likely due to high skewness\n"
            stats_text += f"(skew={stats.skew(confidences):.2f}), not true bimodality.\n"
        
        if 'phase3_optimal' in analysis:
            optimal = analysis['phase3_optimal']['optimal_f1_threshold']
            optimal_f1 = analysis['phase3_optimal']['optimal_f1']
            stats_text += f"\nPhase 3 Optimal F1:\n"
            stats_text += f"  Threshold: {optimal:.4f}\n"
            stats_text += f"  F1 Score: {optimal_f1:.4f}\n"
            if valley is not None:
                diff = abs(valley - optimal)
                rel_err = (diff / optimal) * 100 if optimal > 0 else 0
                stats_text += f"\nValley vs Optimal:\n"
                stats_text += f"  Difference: {diff:.4f}\n"
                stats_text += f"  Relative Error: {rel_err:.1f}%"
        
        ax7.text(0.05, 0.95, stats_text, fontsize=10, family='monospace',
                verticalalignment='top', transform=ax7.transAxes,
                bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow', 
                         alpha=0.8, edgecolor='black', linewidth=1.5))
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
