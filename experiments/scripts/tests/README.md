# Test and Diagnostic Scripts

Scripts for testing models, debugging, and comparing with previous implementations.

## Scripts

### `test_yolo_world_model.py`

Comprehensive YOLO World model testing:
- Model loading
- Class setting
- Single predictions
- Multiple prompts
- Evaluator integration

**Usage:**
```bash
python experiments/scripts/tests/test_yolo_world_model.py
```

### `test_known_working_prompts.py`

Tests specific prompts that previously yielded detections to verify model functionality.

**Usage:**
```bash
python experiments/scripts/tests/test_known_working_prompts.py
```

### `test_compare_with_previous.py`

Compares current implementation with previous working code to identify differences.

**Usage:**
```bash
python experiments/scripts/tests/test_compare_with_previous.py
```

## When to Use

- **After code changes:** Run tests to verify models still work
- **Debugging issues:** Use diagnostic scripts to isolate problems
- **Verifying fixes:** Test specific scenarios after bug fixes
