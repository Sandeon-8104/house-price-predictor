"""
Step 7 + 8: Evaluate the best model deeply, then save it.

Step 7 - EVALUATE:
  1. Predicted vs Actual plot - see WHERE the model is right/wrong
  2. Feature importance       - which columns the model relied on most
  3. Sample predictions       - look at real examples in dollars

Step 8 - SAVE:
  Save the trained model to models/house_price_model.pkl
  so we can reuse it anytime WITHOUT retraining.

Run from the project folder with:  py src/evaluate_and_save.py
"""

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "california_housing_features.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "house_price_model.pkl"
PLOTS_DIR = PROJECT_ROOT / "notebooks" / "plots"

TARGET = "MedHouseVal"


def main():
    # ---------- Recreate the same split (random_state=42 = identical split) ----------
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---------- Train the winner: Random Forest ----------
    print("Training the winning model (Random Forest)...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f"R2 = {r2_score(y_test, preds):.3f}, RMSE = {root_mean_squared_error(y_test, preds):.3f}")

    # ---------- 7a. Predicted vs Actual plot ----------
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, preds, s=3, alpha=0.3, color="green")
    # perfect-prediction line: if pred == actual, dot lands on this line
    lims = [0, 5]
    plt.plot(lims, lims, "r--", label="Perfect prediction")
    plt.xlabel("Actual price ($100,000s)")
    plt.ylabel("Predicted price ($100,000s)")
    plt.title("Predicted vs Actual - Random Forest")
    plt.legend()
    plt.savefig(PLOTS_DIR / "5_predicted_vs_actual.png", dpi=100)
    plt.close()
    print("\nSaved plot: 5_predicted_vs_actual.png")

    # ---------- 7b. Feature importance ----------
    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values()
    plt.figure(figsize=(9, 6))
    importance.plot(kind="barh", color="teal")
    plt.title("Which features does the model rely on?")
    plt.xlabel("Importance (all bars sum to 1.0)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "6_feature_importance.png", dpi=100)
    plt.close()
    print("Saved plot: 6_feature_importance.png")

    print("\nFEATURE IMPORTANCE (what the model actually uses):")
    print(importance.sort_values(ascending=False).round(3).to_string())

    # ---------- 7c. Sample predictions in real dollars ----------
    print("\nSAMPLE PREDICTIONS (5 random test districts):")
    sample = X_test.sample(5, random_state=1)
    sample_preds = model.predict(sample)
    sample_actual = y_test.loc[sample.index]
    for i, (pred, actual) in enumerate(zip(sample_preds, sample_actual), 1):
        diff = abs(pred - actual) * 100000
        print(f"  District {i}: predicted ${pred*100000:>9,.0f} | actual ${actual*100000:>9,.0f} | off by ${diff:,.0f}")

    # ---------- STEP 8: Save the model ----------
    joblib.dump(model, MODEL_PATH)
    size_mb = MODEL_PATH.stat().st_size / 1e6
    print(f"\nModel saved to: {MODEL_PATH} ({size_mb:.1f} MB)")
    print("Reload it anytime with:  model = joblib.load('models/house_price_model.pkl')")


if __name__ == "__main__":
    main()
