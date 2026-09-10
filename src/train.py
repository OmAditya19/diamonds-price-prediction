import sys
from pathlib import Path

import json
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
# ---------------------------------------------------------
# Project path setup
# ---------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.preprocessing import clean_data, prepare_features


DATA_URL = (
    "https://raw.githubusercontent.com/"
    "mwaskom/seaborn-data/master/diamonds.csv"
)

MODEL_PATH = "models/diamond_price_model.pkl"
METADATA_PATH = "models/model_metadata.json"


def load_data():
    return pd.read_csv(DATA_URL)


def train_model():

    # -------------------------
    # 1. Load data
    # -------------------------
    df = load_data()

    # -------------------------
    # 2. Clean data
    # -------------------------
    df = clean_data(df)

    # -------------------------
    # 3. Prepare features
    # -------------------------
    X = prepare_features(df)

    y = np.log(df["price"])

    # -------------------------
    # 4. Train/test split
    # -------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    # -------------------------
    # 5. Add intercept
    # -------------------------
    X_train_const = sm.add_constant(X_train)
    X_test_const = sm.add_constant(X_test)

    # -------------------------
    # 6. Fit Model C
    # -------------------------
    model = sm.OLS(
        y_train,
        X_train_const
    ).fit()

    # -------------------------
    # 7. Predictions
    # -------------------------
    log_predictions = model.predict(X_test_const)

    # -------------------------
    # 8. Smearing correction
    # -------------------------
    correction_factor = np.mean(
        np.exp(model.resid)
    )

    predictions = (
        np.exp(log_predictions)
        * correction_factor
    )

    actual_prices = np.exp(y_test)

    # -------------------------
    # 9. Evaluation
    # -------------------------
    test_r2 = r2_score(
        y_test,
        log_predictions
    )

    test_mae = mean_absolute_error(
        actual_prices,
        predictions
    )

    # ---------------------------------------------------------
    # Visualization 1 — Actual vs Predicted
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        actual_prices,
        predictions,
        alpha=0.15,
        s=10,
        )

    min_price = min(actual_prices.min(), predictions.min())
    max_price = max(actual_prices.max(), predictions.max())

    ax.plot(
        [min_price, max_price],
        [min_price, max_price],
        linestyle="--",
        linewidth=2,
        )

    ax.set_xlabel("Actual Price ($)")
    ax.set_ylabel("Predicted Price ($)")
    ax.set_title("Actual vs Predicted Diamond Prices")

    fig.tight_layout()
    fig.savefig(
        ROOT_DIR / "outputs" / "actual_vs_predicted.png",
        dpi=180,
        bbox_inches="tight",
        )
    plt.close(fig)
    # ---------------------------------------------------------
    # Visualization 2 — Prediction Errors
    # ---------------------------------------------------------
    errors = predictions - actual_prices

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(
        errors,
        bins=60,
        alpha=0.8,
        )

    ax.axvline(
        0,
        linestyle="--",
        linewidth=2,
        )

    ax.set_xlabel("Prediction Error ($)")
    ax.set_ylabel("Number of Diamonds")
    ax.set_title("Distribution of Prediction Errors")

    fig.tight_layout()

    fig.savefig(
        ROOT_DIR / "outputs" / "prediction_errors.png",
        dpi=180,
        bbox_inches="tight",
        )

    plt.close(fig)

    # ---------------------------------------------------------
    # Visualization 3 — Price vs Carat
    # ---------------------------------------------------------
    
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(
        np.exp(X_test["log_carat"]),
        actual_prices,
        alpha=0.12,
        s=10,
        )

    ax.set_xlabel("Carat")
    ax.set_ylabel("Price ($)")
    ax.set_title("Diamond Price vs Carat — Test Set")

    fig.tight_layout()

    fig.savefig(
        ROOT_DIR / "outputs" / "price_vs_carat.png",
        dpi=180,
        bbox_inches="tight",
        )

    plt.close(fig)

    # -------------------------
    # 10. Save model
    # -------------------------
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    # -------------------------
    # 11. Save metadata
    # -------------------------
    metadata = {
        "model_name": "Log-Log Regression",
        "features": [
            "log_carat",
            "cut",
            "color",
            "clarity",
            "depth",
            "table",
        ],
        "training_observations": len(X_train),
        "testing_observations": len(X_test),
        "test_r2": test_r2,
        "test_mae": test_mae,
        "smearing_correction": correction_factor,
        "random_state": 42,
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(
            metadata,
            f,
            indent=4
        )

    # -------------------------
    # 12. Print results
    # -------------------------
    print("=" * 55)
    print("DIAMOND PRICE MODEL")
    print("=" * 55)

    print(
        f"Training observations: "
        f"{len(X_train):,}"
    )

    print(
        f"Testing observations:  "
        f"{len(X_test):,}"
    )

    print(
        f"Test R²: "
        f"{test_r2:.4f}"
    )

    print(
        f"Test MAE: "
        f"${test_mae:,.2f}"
    )

    print(
        f"Smearing correction: "
        f"{correction_factor:.4f}"
    )

    print("\nModel saved:")
    print(MODEL_PATH)

    print("\nMetadata saved:")
    print(METADATA_PATH)


if __name__ == "__main__":
    train_model()
