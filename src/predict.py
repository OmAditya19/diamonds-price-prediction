import json
import pickle

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.preprocessing import (
    CUT_MAPPING,
    COLOR_MAPPING,
    CLARITY_MAPPING,
)


MODEL_PATH = "models/diamond_price_model.pkl"
METADATA_PATH = "models/model_metadata.json"


def load_model():

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    return model, metadata


def prepare_input(
    carat,
    cut,
    color,
    clarity,
    depth,
    table,
):

    data = pd.DataFrame({
        "carat": [carat],
        "cut": [cut],
        "color": [color],
        "clarity": [clarity],
        "depth": [depth],
        "table": [table],
    })

    data["cut"] = data["cut"].map(CUT_MAPPING)
    data["color"] = data["color"].map(COLOR_MAPPING)
    data["clarity"] = data["clarity"].map(CLARITY_MAPPING)

    data["log_carat"] = np.log(
        data["carat"]
    )

    features = [
        "log_carat",
        "cut",
        "color",
        "clarity",
        "depth",
        "table",
    ]

    X = data[features]

    return sm.add_constant(X, has_constant = "add")


def predict_price(
    carat,
    cut,
    color,
    clarity,
    depth,
    table,
):

    model, metadata = load_model()

    X = prepare_input(
        carat=carat,
        cut=cut,
        color=color,
        clarity=clarity,
        depth=depth,
        table=table,
    )

    log_prediction = model.predict(X)

    price = (
        np.exp(log_prediction.iloc[0])
        * metadata["smearing_correction"]
    )

    return float(price)
