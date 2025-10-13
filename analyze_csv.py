import pandas as pd
import sys

file_path = sys.argv[1]

try:
    # Read the first 100 lines to get a sense of the data without loading the whole file
    df_head = pd.read_csv(file_path, nrows=100)

    print("File Path:", file_path)
    print("\nFirst 5 rows:")
    print(df_head.head().to_markdown(index=False))

    print("\nColumn Info:")
    df_head.info()

    # Attempt to get basic descriptive statistics if possible with the head
    print("\nBasic Statistics (first 100 rows):")
    print(df_head.describe().to_markdown())

except Exception as e:
    print(f"Error analyzing CSV: {e}")