"""
Step 3: Clean the data.

Based on our EDA findings, we fix:
  1. Capped prices  - rows where MedHouseVal == 5.0 (true price unknown, misleads model)
  2. Extreme outliers - e.g. AveRooms = 141, AveOccup = 1243 (dorms/prisons, not normal homes)
  3. Missing values  - none here, but we check anyway (good habit)

Run from the project folder with:  py src/clean_data.py
Output:  data/california_housing_clean.csv
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_ROOT / "data" / "california_housing.csv"
CLEAN_PATH = PROJECT_ROOT / "data" / "california_housing_clean.csv"


def main():
    df = pd.read_csv(RAW_PATH)
    print(f"Starting rows: {len(df)}")

    # ---------- 1. Missing values ----------
    missing = df.isnull().sum().sum()
    if missing > 0:
        # Fill numeric gaps with the column median (robust to outliers)
        df = df.fillna(df.median(numeric_only=True))
        print(f"Filled {missing} missing values with column medians")
    else:
        print("No missing values - nothing to fill")

    # ---------- 2. Remove capped prices ----------
    capped = (df["MedHouseVal"] >= 5.0).sum()
    df = df[df["MedHouseVal"] < 5.0]
    print(f"Removed {capped} rows with capped price (>= $500k, true value unknown)")

    # ---------- 3. Remove extreme outliers ----------
    # Keep only sensible districts. Thresholds chosen from the describe() stats:
    # 99%+ of rows are far below these values.
    before = len(df)
    df = df[df["AveRooms"] < 15]     # normal homes, not 141-room anomalies
    df = df[df["AveBedrms"] < 5]     # avg bedrooms per household
    df = df[df["AveOccup"] < 10]     # avg people per household
    df = df[df["Population"] < 10000]
    print(f"Removed {before - len(df)} outlier rows (dorms/prisons/data errors)")

    # ---------- Save ----------
    df.to_csv(CLEAN_PATH, index=False)
    print(f"\nFinal rows: {len(df)}  (removed {20640 - len(df)} total)")
    print(f"Saved clean data to: {CLEAN_PATH}")

    print("\nNew statistics (compare max values with before!):")
    print(df.describe().round(2).to_string())


if __name__ == "__main__":
    main()
