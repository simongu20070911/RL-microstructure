import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_metric(df_prev, df_latest, metric_tag, title, ylabel, filename):
    plt.figure(figsize=(10, 6))

    if df_prev is not None and metric_tag in df_prev['tag'].unique():
        df_prev_metric = df_prev[df_prev['tag'] == metric_tag]
        plt.plot(df_prev_metric['step'], df_prev_metric['value'], label='Previous Run (20250316-113047)')

    if df_latest is not None and metric_tag in df_latest['tag'].unique():
        df_latest_metric = df_latest[df_latest['tag'] == metric_tag]
        plt.plot(df_latest_metric['step'], df_latest_metric['value'], label='Latest Run')

    plt.xlabel('Training Steps')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
    print(f"Generated plot: {filename}")

def main():
    prev_csv_path = 'results_20250316-113047.csv'
    latest_csv_path = 'results_latest_log.csv'

    df_prev = None
    if os.path.exists(prev_csv_path):
        df_prev = pd.read_csv(prev_csv_path)
        print(f"Loaded {prev_csv_path}")
    else:
        print(f"File not found: {prev_csv_path}")

    df_latest = None
    if os.path.exists(latest_csv_path):
        df_latest = pd.read_csv(latest_csv_path)
        print(f"Loaded {latest_csv_path}")
    else:
        print(f"File not found: {latest_csv_path}")

    # Plot training reward mean
    plot_metric(df_prev, df_latest, 'rollout/ep_rew_mean', 'Training Episode Reward Mean', 'Mean Reward', 'training_reward_mean.png')

    # Plot validation PnL mean (only available in previous run)
    if df_prev is not None and 'validation/mean_pnl' in df_prev['tag'].unique():
         plot_metric(df_prev, None, 'validation/mean_pnl', 'Validation Mean PnL (Previous Run)', 'Mean PnL', 'validation_pnl_mean.png')
    else:
        print("Validation PnL data not available in previous run.")


if __name__ == '__main__':
    main()