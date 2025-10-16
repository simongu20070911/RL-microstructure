import os
import csv
from glob import glob
from tensorboard.backend.event_processing import event_accumulator

def extract_scalars(event_file, out_csv):
    ea = event_accumulator.EventAccumulator(event_file, size_guidance={'scalars': 0})
    ea.Reload()
    tags = ea.Tags().get('scalars', [])
    # Focus on RL-relevant metrics
    relevant_tags = [t for t in tags if any(x in t.lower() for x in ['reward', 'loss', 'mean', 'min', 'max', 'eval', 'performance'])]
    if not relevant_tags:
        relevant_tags = tags  # fallback: dump all scalars
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['tag', 'step', 'value'])
        for tag in relevant_tags:
            for event in ea.Scalars(tag):
                writer.writerow([tag, event.step, event.value])

def main():
    # 1. Previous experiment
    prev_event = 'envs/previous_ver/20250316-113047/events.out.tfevents.1742124647.gaen-linux.1854622.0'
    out_prev = 'results_20250316-113047.csv'
    if os.path.exists(prev_event):
        extract_scalars(prev_event, out_prev)
        print(f"Extracted: {out_prev}")
    else:
        print(f"Event file not found: {prev_event}")

    # 2. Latest log experiment
    logs_dir = 'logs'
    subdirs = [os.path.join(logs_dir, d) for d in os.listdir(logs_dir) if os.path.isdir(os.path.join(logs_dir, d))]
    if subdirs:
        latest_subdir = max(subdirs, key=os.path.getmtime)
        event_files = glob(os.path.join(latest_subdir, 'events.out.tfevents.*'))
        out_latest = 'results_latest_log.csv'
        with open(out_latest, 'w', newline='') as f:
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
        print("No subdirectories found in logs/")

if __name__ == '__main__':
    main()