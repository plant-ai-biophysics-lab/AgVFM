#!/bin/bash
# Quick script to check Phase 1 experiment progress

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
RESULTS_DIR="$PROJECT_ROOT/experiments/results/phase1_factor_analysis"

echo "=== Phase 1 Factor Analysis Progress ==="
echo

# Check if process is running
if pgrep -f "run_phase1_factor_analysis" > /dev/null; then
    echo "✅ Experiment is RUNNING"
    echo
else
    echo "❌ Experiment is NOT running"
    echo
fi

# Check results file
if [ -f "$RESULTS_DIR/yolo_world_factor_analysis.json" ]; then
    COMPLETED=$(python3 -c "import json; f=open('$RESULTS_DIR/yolo_world_factor_analysis.json'); d=json.load(f); print(len(d.get('results', {})))" 2>/dev/null || echo "0")
    TOTAL=34
    PERCENT=$((COMPLETED * 100 / TOTAL))
    echo "📊 YOLO World: $COMPLETED/$TOTAL configurations ($PERCENT%)"
    
    # Check test set
    TEST_SET=$(python3 -c "import json; f=open('$RESULTS_DIR/yolo_world_factor_analysis.json'); d=json.load(f); print(d.get('test_set', 'unknown'))" 2>/dev/null || echo "unknown")
    N_IMAGES=$(python3 -c "import json; f=open('$RESULTS_DIR/yolo_world_factor_analysis.json'); d=json.load(f); print(d.get('n_images', 'unknown'))" 2>/dev/null || echo "unknown")
    echo "   Test set: $TEST_SET ($N_IMAGES images)"
else
    echo "📊 YOLO World: No results file yet"
fi

echo
echo "=== Recent Log Activity ==="
echo "Latest log entries:"
tail -5 "$RESULTS_DIR/logs/"*.log 2>/dev/null | tail -10 || echo "No log files found"

echo
echo "=== Monitor Commands ==="
echo "  Watch progress: tail -f $RESULTS_DIR/run_full_test.log"
echo "  Watch logs:    tail -f $RESULTS_DIR/logs/phase1_factor_analysis_*.log"
echo "  Check results: jq '.results | keys | length' $RESULTS_DIR/yolo_world_factor_analysis.json"
