"""
Step 2: Exploratory Data Analysis (EDA).

We look at the data BEFORE modeling to understand:
  1. What does each column look like? (histograms)
  2. Are there missing values or weird outliers?
  3. Which features relate to house price? (scatter + correlation)

Run from the project folder with:  py src/explore_data.py
Plots are saved into:  notebooks/plots/
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save plots to files (no popup windows)
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "california_housing.csv"
PLOTS_DIR = PROJECT_ROOT / "notebooks" / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def main():
    df = pd.read_csv(DATA_PATH)

    # ---------- 1. Basic info ----------
    print("=" * 60)
    print("SHAPE (rows, columns):", df.shape)
    print("=" * 60)

    print("\nMISSING VALUES per column:")
    print(df.isnull().sum())

    print("\nSTATISTICS (look at min/max for outliers!):")
    print(df.describe().round(2).to_string())

    # ---------- 2. Histograms of every column ----------
    df.hist(bins=50, figsize=(14, 10))
    plt.suptitle("Distribution of every column", fontsize=16)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "1_histograms.png", dpi=100)
    plt.close()
    print("\nSaved plot: 1_histograms.png")

    # ---------- 3. Income vs Price scatter (the classic plot) ----------
    plt.figure(figsize=(10, 6))
    plt.scatter(df["MedInc"], df["MedHouseVal"], s=2, alpha=0.3, color="green")
    plt.xlabel("Median Income (in $10,000s)")
    plt.ylabel("Median House Value (in $100,000s)")
    plt.title("House Price vs. Income")
    plt.savefig(PLOTS_DIR / "2_income_vs_price.png", dpi=100)
    plt.close()
    print("Saved plot: 2_income_vs_price.png")

    # ---------- 4. Geographic plot: price on the California map ----------
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        df["Longitude"], df["Latitude"],
        c=df["MedHouseVal"], cmap="jet", s=4, alpha=0.4,
    )
    plt.colorbar(scatter, label="Median House Value ($100k)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("House prices across California (red = expensive)")
    plt.savefig(PLOTS_DIR / "3_california_map.png", dpi=100)
    plt.close()
    print("Saved plot: 3_california_map.png")

    # ---------- 5. Correlation heatmap ----------
    plt.figure(figsize=(10, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation between all columns")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "4_correlation_heatmap.png", dpi=100)
    plt.close()
    print("Saved plot: 4_correlation_heatmap.png")

    # ---------- 6. Which features correlate most with price? ----------
    print("\nCORRELATION with house price (1.0 = perfect, 0 = none):")
    print(corr["MedHouseVal"].sort_values(ascending=False).round(3).to_string())


if __name__ == "__main__":
    main()
