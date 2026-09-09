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


def load_data():
    return pd.read_csv(DATA_URL)


def train_model():

    # -------------------------
    # 1. Load
    # -------------------------
    df = load_data()

    # -------------------------
    # 2. Clean
    # -------------------------
    df = clean_data(df)

    # -------------------------
    # 3. Features
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
    # 6. Train
    # -------------------------
    model = sm.OLS(
        y_train,
        X_train_const
    ).fit()

    # -------------------------
    # 7. Log-scale prediction
    # -------------------------
    log_predictions = model.predict(X_test_const)

    # -------------------------
    # 8. Smearing correction
    # -------------------------
    correction_factor = np.mean(
        np.exp(model.resid)
    )

    # -------------------------
    # 9. Convert back to dollars
    # -------------------------
    predictions = (
        np.exp(log_predictions)
        * correction_factor
    )

    actual_prices = np.exp(y_test)

    # -------------------------
    # 10. Metrics
    # -------------------------
    mae = mean_absolute_error(
        actual_prices,
        predictions
    )

    log_r2 = r2_score(
        y_test,
        log_predictions
    )

    print("=" * 50)
    print("DIAMOND PRICE MODEL")
    print("=" * 50)

    print(f"Training observations: {len(X_train):,}")
    print(f"Testing observations:  {len(X_test):,}")

    print(f"Log-scale R²: {log_r2:.4f}")
    print(f"Test MAE: ${mae:,.2f}")

    print(
        f"Smearing correction: "
        f"{correction_factor:.4f}"
    )

    return model, correction_factor


if __name__ == "__main__":
    train_model()
