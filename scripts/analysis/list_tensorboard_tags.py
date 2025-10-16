import pandas as pd
import sys

file_path = sys.argv[1]

try:
    df = pd.read_csv(file_path)
    if 'tag' in df.columns:
        unique_tags = df['tag'].unique()
        print(f"Unique tags in {file_path}:")
        for tag in unique_tags:
            print(f"- {tag}")
    else:
        print(f"Column 'tag' not found in {file_path}")

except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
except Exception as e:
    print(f"Error reading CSV: {e}")