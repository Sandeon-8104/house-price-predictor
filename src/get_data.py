"""
Step 1: Get the California Housing dataset and save it as a CSV.

Run this from the project folder with:  py src/get_data.py
"""

from pathlib import Path

from sklearn.datasets import fetch_california_housing

# Folder where the CSV will be saved (project_root/data)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def main():
    print("Downloading California Housing dataset...")

    # as_frame=True gives us a pandas DataFrame (a table, like Excel)
    dataset = fetch_california_housing(as_frame=True)
    df = dataset.frame  # features + target in one table

    # Save to CSV so we never need to download again
    csv_path = DATA_DIR / "california_housing.csv"
    df.to_csv(csv_path, index=False)

    # Show a quick summary so we know it worked
    print(f"\nSaved to: {csv_path}")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print("\nColumns in the dataset:")
    for col in df.columns:
        print(f"  - {col}")
    print("\nFirst 5 rows:")
    print(df.head())

if __name__ == "__main__":
    main()
