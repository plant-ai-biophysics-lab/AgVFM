# Experiment Scripts

Organized scripts for running experiments, monitoring progress, testing, visualization, and examples.

## Directory Structure

```
scripts/
├── experiments/          # Experiment runners
│   └── phase1/         # Phase 1: Factor Analysis
│       ├── run_factor_analysis.py
│       └── README.md
│   └── phase2/         # Phase 2: Combination Tests (future)
│   └── phase3/         # Phase 3: Confidence Threshold Analysis (future)
│
├── monitoring/          # Status and progress checking
│   ├── check_experiment_progress.sh
│   ├── check_experiment_status.sh
│   └── README.md
│
├── tests/              # Test and diagnostic scripts
│   ├── test_yolo_world_model.py
│   ├── test_known_working_prompts.py
│   ├── test_compare_with_previous.py
│   └── README.md
│
├── visualization/      # Visualization scripts
│   ├── visualize_phase1_results.py
│   └── README.md
│
└── examples/           # Example/demo scripts
    ├── example_usage.py
    └── README.md
```

## Quick Start

### Run Phase 1 Experiments

```bash
# Run both models on full test set
python experiments/scripts/experiments/phase1/run_factor_analysis.py --model all --full-test-set

# Run just YOLO World
python experiments/scripts/experiments/phase1/run_factor_analysis.py --model yolo_world --full-test-set
```

### Monitor Progress

```bash
# Quick status check
bash experiments/scripts/monitoring/check_experiment_progress.sh

# Detailed status
bash experiments/scripts/monitoring/check_experiment_status.sh
```

### Generate Visualizations

```bash
# After experiments complete
python experiments/scripts/visualization/visualize_phase1_results.py
```

## Documentation

Each subdirectory has its own README with detailed documentation:
- **experiments/phase1/README.md** - Phase 1 experiment details
- **monitoring/README.md** - Monitoring script usage
- **tests/README.md** - Test script usage
- **visualization/README.md** - Visualization script usage
- **examples/README.md** - Example script usage

## Adding New Experiments

When adding Phase 2, Phase 3, etc.:

1. Create new directory: `experiments/phase2/`
2. Add experiment runner: `experiments/phase2/run_combination_tests.py`
3. Add README: `experiments/phase2/README.md`
4. Update this README with new phase

## Design Principles

- **Scalable:** Each phase in its own directory
- **Organized:** Scripts grouped by purpose
- **Documented:** Each directory has README
- **Maintainable:** Clear structure for future additions
