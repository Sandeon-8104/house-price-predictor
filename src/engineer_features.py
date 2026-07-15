"""
Step 4: Feature Engineering.

We create NEW columns from existing ones to give the model better hints:
  1. Ratio features   - rooms per person, bedroom ratio (quality-of-home signals)
  2. Location features - distance to San Francisco & Los Angeles
                        (turns raw Lat/Long into something meaningful:
                         "close to a big city = expensive")

Run from the project folder with:  py src/engineer_features.py
Output:  data/california_housing_features.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_PATH = PROJECT_ROOT / "data" / "california_housing_clean.csv"
FEATURES_PATH = PROJECT_ROOT / "data" / "california_housing_features.csv"

# City centers (latitude, longitude)
SAN_FRANCISCO = (37.77, -122.42)
LOS_ANGELES = (34.05, -118.24)


def distance_to(df, city):
    """Straight-line distance (in degrees) from each district to a city."""
    lat, lon = city
    return np.sqrt((df["Latitude"] - lat) ** 2 + (df["Longitude"] - lon) ** 2)


def main():
    df = pd.read_csv(CLEAN_PATH)
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns")

    # ---------- 1. Ratio features ----------
    # How spacious are homes? (rooms per person living there)
    df["RoomsPerPerson"] = df["AveRooms"] / df["AveOccup"]

    # What fraction of rooms are bedrooms?
    # (low ratio = big living areas = usually fancier homes)
    df["BedrmsRatio"] = df["AveBedrms"] / df["AveRooms"]

    # ---------- 2. Location features ----------
    df["DistToSF"] = distance_to(df, SAN_FRANCISCO)
    df["DistToLA"] = distance_to(df, LOS_ANGELES)

    # Distance to whichever big city is closer
    df["DistToCity"] = df[["DistToSF", "DistToLA"]].min(axis=1)

    # ---------- Save ----------
    df.to_csv(FEATURES_PATH, index=False)
    print(f"Added 5 new features. Now {df.shape[1]} columns.")
    print(f"Saved to: {FEATURES_PATH}")

    # ---------- Check: do the new features correlate with price? ----------
    new_features = ["RoomsPerPerson", "BedrmsRatio", "DistToSF", "DistToLA", "DistToCity"]
    print("\nCorrelation of NEW features with house price:")
    print(df[new_features + ["MedHouseVal"]].corr()["MedHouseVal"].drop("MedHouseVal").round(3).to_string())
    print("\n(For comparison: best original feature MedInc = 0.688,")
    print(" Latitude was only -0.144 and Longitude -0.046)")


if __name__ == "__main__":
    main()
