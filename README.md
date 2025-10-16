# High-Frequency Trading Reinforcement Learning Framework

A modular research codebase for building and evaluating reinforcement learning agents in high-frequency trading environments. The repository now separates core package code, scripts, data artefacts, documentation, and archived experiments so that day-to-day development stays organised.
This repo is cleaned. 

## Repository Layout

```
.
├── rltrader/                 # Core Python package
│   ├── agents/               # Public agent APIs backed by maintained implementations
│   ├── envs/                 # Stable environment interfaces
│   ├── configs/              # Published configuration presets
│   ├── utils/                # Shared utilities
│   └── lib/                  # Original source tree preserved for reference
├── scripts/
│   ├── analysis/             # Offline analytics and reporting helpers
│   ├── applications/         # Flask/Streamlit dashboards
│   ├── monitoring/           # Operational monitoring utilities
│   ├── training/             # Entry points for training workflows
│   ├── utilities/            # Data and infrastructure helpers
│   └── lib/                  # Archived exploratory scripts
├── data/
│   └── raw/                  # Order book datasets (git-ignored; add via Git LFS if needed)
├── runs/
│   ├── logs/                 # Training logs, PID files, TensorBoard runs
│   └── results/              # Aggregated CSV/PNG outputs
├── lib/                      # Archived datasets and miscellaneous resources
├── infra/
│   └── docker/               # Optional container definitions (Dockerfile, compose)
├── tests/
│   ├── manual/               # Interactive validation suites
│   └── test_execution.py     # Pytest-based smoke coverage
├── docs/
│   ├── guides/               # Operational documentation
│   ├── reports/              # Result summaries
│   └── research/             # Academic paper artefacts
├── tools/                    # Order book data preparation utilities
├── research/experiments/     # Archived experimental notebooks and configs
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Prepare data (optional)

Place raw order book CSV files under `data/raw/`. The training scripts default to `data/raw/orderbook_trimmed_small.csv`; adjust as required.

### 3. Run a baseline training session

```bash
python scripts/training/run_two_sided_training.py
```

Configuration parameters can be edited inline (see the `config` dictionary near the top of the script) or by loading JSON from the run directory.

For rebated market experiments:

```bash
python scripts/training/run_rebated_training.py
```


TensorBoard logs are emitted to `runs/logs/<run>/tensorboard`. Launch TensorBoard locally with:

```bash
tensorboard --logdir runs/logs
```

### 5. Analyse results

Utility scripts under `scripts/analysis/` provide quick summaries, e.g.:

- `python scripts/analysis/training_progress_report.py`
- `python scripts/analysis/extract_tensorboard_scalars.py`
- `python scripts/analysis/benchmark_inference_speed.py`

Outputs are written to `runs/results/` by default.


This project is distributed under the MIT License. See `LICENSE` (if provided) for full details.
