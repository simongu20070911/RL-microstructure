import pandas as pd
import sys

file_path = sys.argv[1]
metric_tags = sys.argv[2:]

try:
    df = pd.read_csv(file_path)
    print(f"Last recorded values from {file_path}:")
    for tag in metric_tags:
        if tag in df['tag'].unique():
            # Get the last value for the specific tag
            last_value = df[df['tag'] == tag]['value'].iloc[-1]
            print(f"- {tag}: {last_value}")
        else:
            print(f"- {tag}: Not found")

except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
except Exception as e:
    print(f"Error reading CSV: {e}")