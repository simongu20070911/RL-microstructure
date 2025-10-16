from pathlib import Path
import os
import csv
from glob import glob
from tensorboard.backend.event_processing import event_accumulator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / "runs"
LOGS_DIR = RUNS_DIR / "logs"
RESULTS_DIR = RUNS_DIR / "results"
LIB_PREV_DIR = PROJECT_ROOT / "rltrader" / "lib" / "envs" / "previous_ver"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def extract_scalars(event_file, out_csv):
    out_csv = Path(out_csv)
    ea = event_accumulator.EventAccumulator(event_file, size_guidance={'scalars': 0})
    ea.Reload()
    tags = ea.Tags().get('scalars', [])
    # Focus on RL-relevant metrics
    relevant_tags = [t for t in tags if any(x in t.lower() for x in ['reward', 'loss', 'mean', 'min', 'max', 'eval', 'performance'])]
    if not relevant_tags:
        relevant_tags = tags  # fallback: dump all scalars
    with out_csv.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['tag', 'step', 'value'])
        for tag in relevant_tags:
            for event in ea.Scalars(tag):
                writer.writerow([tag, event.step, event.value])

def main():
    # 1. Previous experiment
    prev_event = LIB_PREV_DIR / '20250316-113047' / 'events.out.tfevents.1742124647.gaen-linux.1854622.0'
    out_prev = RESULTS_DIR / 'results_20250316-113047.csv'
    if prev_event.exists():
        extract_scalars(str(prev_event), out_prev)
        print(f"Extracted: {out_prev}")
    else:
        print(f"Event file not found: {prev_event}")

    # 2. Latest log experiment
    if not LOGS_DIR.exists():
        print(f"Log directory not found: {LOGS_DIR}")
        return

    subdirs = [d for d in LOGS_DIR.iterdir() if d.is_dir()]
    if subdirs:
        latest_subdir = max(subdirs, key=lambda d: d.stat().st_mtime)
        event_files = glob(str(latest_subdir / 'events.out.tfevents.*'))
        out_latest = RESULTS_DIR / 'results_latest_log.csv'
        with out_latest.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['tag', 'step', 'value'])
            for event_file in event_files:
                ea = event_accumulator.EventAccumulator(event_file, size_guidance={'scalars': 0})
                ea.Reload()
                tags = ea.Tags().get('scalars', [])
                relevant_tags = [t for t in tags if any(x in t.lower() for x in ['reward', 'loss', 'mean', 'min', 'max', 'eval', 'performance'])]
                if not relevant_tags:
                    relevant_tags = tags
                for tag in relevant_tags:
                    for event in ea.Scalars(tag):
                        writer.writerow([tag, event.step, event.value])
        print(f"Extracted: {out_latest}")
    else:
        print(f"No subdirectories found in {LOGS_DIR}")

if __name__ == '__main__':
    main()
