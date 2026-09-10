import json
import pickle

import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

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
