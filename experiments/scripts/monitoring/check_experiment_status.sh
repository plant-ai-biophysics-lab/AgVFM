#!/bin/bash
# Quick script to check experiment status

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== Experiment Status ==="
echo ""

# Check if process is running
if pgrep -f "run_factor_analysis" > /dev/null; then
    echo "✅ Experiments are RUNNING"
    echo ""
    ps aux | grep "run_factor_analysis" | grep -v grep | head -1
else
    echo "⚠️  No experiment process found"
fi

echo ""
echo "=== Latest Log Output ==="
if [ -f "experiments/results/phase1_factor_analysis/run.log" ]; then
    tail -20 experiments/results/phase1_factor_analysis/run.log
else
    echo "No log file found"
fi

echo ""
echo "=== Results Files ==="
if ls experiments/results/phase1_factor_analysis/*.json 1> /dev/null 2>&1; then
    for f in experiments/results/phase1_factor_analysis/*.json; do
        if [ -f "$f" ]; then
            size=$(du -h "$f" | cut -f1)
            count=$(jq '.results | length' "$f" 2>/dev/null || echo "?")
            echo "  $(basename $f): $size ($count configs completed)"
        fi
    done
else
    echo "  No results files yet"
fi

echo ""
echo "=== Log Files ==="
if ls experiments/results/phase1_factor_analysis/logs/*.log 1> /dev/null 2>&1; then
    ls -lh experiments/results/phase1_factor_analysis/logs/*.log | tail -3
else
    echo "  No log files yet"
fi
