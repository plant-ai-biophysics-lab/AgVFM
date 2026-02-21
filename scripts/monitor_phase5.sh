#!/bin/bash
# Monitor Phase 5 run progress

RESULTS_DIR="experiments/results/phase5_semantic_filtering"
RESULTS_FILE="$RESULTS_DIR/semantic_filtering_results.json"
LOG_FILE="$RESULTS_DIR/run_log_full.txt"

echo "=== Phase 5 Run Monitor ==="
echo ""

# Check if process is running
PROCESS=$(ps aux | grep "run_semantic_filtering" | grep -v grep | grep python3)
if [ -z "$PROCESS" ]; then
    echo "❌ Process is NOT running"
    echo ""
    echo "Check log for errors:"
    tail -50 "$LOG_FILE" 2>/dev/null || echo "Log file not found"
else
    echo "✅ Process is running"
    echo "$PROCESS" | awk '{printf "  PID: %s | CPU: %s%% | MEM: %s%% | Runtime: %s\n", $2, $3, $4, $10}'
fi

echo ""

# Check results file
if [ -f "$RESULTS_FILE" ]; then
    echo "📊 Results file status:"
    ls -lh "$RESULTS_FILE" | awk '{print "  Size: " $5 " | Modified: " $6 " " $7 " " $8}'
    echo ""
    
    # Parse JSON to get progress
    python3 << EOF
import json
from pathlib import Path

results_file = Path("$RESULTS_FILE")
if results_file.exists():
    with open(results_file, 'r') as f:
        data = json.load(f)
    results = data.get('results', [])
    summary = data.get('summary', {})
    total_images = summary.get('num_images', len(results))
    
    print(f"  Images processed: {len(results)}/{total_images}")
    if len(results) > 0:
        total_candidates = sum(r.get('num_candidates', 0) for r in results)
        total_kept = sum(r.get('num_kept', 0) for r in results)
        print(f"  Total candidates: {total_candidates}")
        print(f"  Total kept: {total_kept}")
        if len(results) >= 3:
            print(f"  Last processed: {results[-1].get('image_path', '').split('/')[-1]}")
EOF
else
    echo "⚠️  Results file not found yet"
fi

echo ""
echo "=== Recent Log Output ==="
tail -20 "$LOG_FILE" 2>/dev/null || echo "Log file empty or not found"
echo ""
echo "To monitor in real-time: tail -f $LOG_FILE"
