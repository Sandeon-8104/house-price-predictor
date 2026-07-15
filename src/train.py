"""
Step 5 + 6: Split the data, then train and compare 4 models.

Step 5 - SPLIT:
  80% training set  -> model learns from this
  20% test set      -> hidden during training, used to grade the model on
                       data it has NEVER seen (like a real exam)

Step 6 - TRAIN (simple -> powerful):
  1. Linear Regression  - draws a straight line (baseline)
  2. Decision Tree      - learns if/else rules
  3. Random Forest      - 100 trees voting together
  4. Gradient Boosting  - trees that fix each other's mistakes

Run from the project folder with:  py src/train.py
"""

import time
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "california_housing_features.csv"

TARGET = "MedHouseVal"


def main():
    # ---------- STEP 5: Split ----------
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=[TARGET])   # features (inputs)
    y = df[TARGET]                  # target (what we predict)

    # random_state=42 makes the split reproducible (same split every run)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training set: {len(X_train)} rows (model learns from these)")
    print(f"Test set:     {len(X_test)} rows (model never sees these until grading)")

    # ---------- STEP 6: Train & compare ----------
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
    }

    print("\n" + "=" * 68)
    print(f"{'Model':<20} {'R2':>6} {'RMSE':>8} {'Avg Error':>12} {'Train time':>12}")
    print("=" * 68)

    results = {}
    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)          # <-- this IS training
        train_time = time.time() - start

        preds = model.predict(X_test)         # predict on UNSEEN data
        r2 = r2_score(y_test, preds)
        rmse = root_mean_squared_error(y_test, preds)
        mae = mean_absolute_error(y_test, preds)

        results[name] = (model, r2)
        # prices are in $100,000s -> multiply by 100k for dollars
        print(f"{name:<20} {r2:>6.3f} {rmse:>8.3f} {'$' + format(mae * 100000, ',.0f'):>12} {train_time:>10.1f}s")

    print("=" * 68)
    print("R2:  1.0 = perfect, 0 = useless  |  RMSE: lower = better")
    print("Avg Error = how far off a typical prediction is, in dollars")

    best_name = max(results, key=lambda k: results[k][1])
    print(f"\nWINNER: {best_name} (R2 = {results[best_name][1]:.3f})")


if __name__ == "__main__":
    main()
