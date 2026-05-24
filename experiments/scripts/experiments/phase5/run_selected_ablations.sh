#!/bin/bash
# Run selected Phase 5 ablations on full dataset in background mode

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
cd "$PROJECT_ROOT"

RESULTS_DIR="experiments/results/phase5_semantic_filtering"
LOG_FILE="$RESULTS_DIR/selected_ablations.log"
PYTHON_BIN="/data/apps/conda/jmearles/envs/agvfm/bin/python3"

# Create results directory
mkdir -p "$RESULTS_DIR/results"

# Redirect all output to log file
exec > >(tee -a "$LOG_FILE")
exec 2>&1

echo "=================================================================================="
echo "🚀 Starting selected ablations on full dataset"
echo "   Date: $(date)"
echo "   Log file: $LOG_FILE"
echo "   You can safely disconnect from SSH"
echo "=================================================================================="
echo ""

# Run selected methods sequentially
# Method 1: BH-FDR q=0.3
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Starting BH-FDR q=0.3..."
$PYTHON_BIN -u experiments/scripts/experiments/phase5/run_semantic_filtering.py \
    --selection-method bh_fdr \
    --q-fdr 0.3 \
    --results-filename semantic_filtering_results_q0.3_full.json \
    --output-dir "$RESULTS_DIR"
echo "[$(date +'%Y-%m-%d %H:%M:%S')] ✅ BH-FDR q=0.3 complete"
echo ""

# Method 2: BH-FDR q=0.5
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Starting BH-FDR q=0.5..."
$PYTHON_BIN -u experiments/scripts/experiments/phase5/run_semantic_filtering.py \
    --selection-method bh_fdr \
    --q-fdr 0.5 \
    --results-filename semantic_filtering_results_q0.5_full.json \
    --output-dir "$RESULTS_DIR"
echo "[$(date +'%Y-%m-%d %H:%M:%S')] ✅ BH-FDR q=0.5 complete"
echo ""

# Method 3: Fixed p=0.1
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Starting Fixed p=0.1..."
$PYTHON_BIN -u experiments/scripts/experiments/phase5/run_semantic_filtering.py \
    --selection-method fixed_p \
    --p-cutoff 0.1 \
    --results-filename semantic_filtering_results_p0.1_full.json \
    --output-dir "$RESULTS_DIR"
echo "[$(date +'%Y-%m-%d %H:%M:%S')] ✅ Fixed p=0.1 complete"
echo ""

# Method 4: Top-5
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Starting Top-5..."
$PYTHON_BIN -u experiments/scripts/experiments/phase5/run_semantic_filtering.py \
    --selection-method top_k \
    --top-k 5 \
    --results-filename semantic_filtering_results_top5_full.json \
    --output-dir "$RESULTS_DIR"
echo "[$(date +'%Y-%m-%d %H:%M:%S')] ✅ Top-5 complete"
echo ""

echo "=================================================================================="
echo "[$(date +'%Y-%m-%d %H:%M:%S')] ✅ All selected ablations complete!"
echo "=================================================================================="
echo ""
echo "📊 Results saved to: $RESULTS_DIR/results/"
echo "📈 Generate plots with:"
echo "   $PYTHON_BIN experiments/scripts/experiments/phase5/visualize_comparison.py \\"
echo "       --results-dir $RESULTS_DIR/results --compute-metrics"
