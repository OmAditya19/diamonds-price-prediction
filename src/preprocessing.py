import numpy as np
import pandas as pd


CUT_MAPPING = {
    "Fair": 0,
    "Good": 1,
    "Very Good": 2,
    "Premium": 3,
    "Ideal": 4,
}

COLOR_MAPPING = {
    "J": 0,
    "I": 1,
    "H": 2,
    "G": 3,
    "F": 4,
    "E": 5,
    "D": 6,
}

CLARITY_MAPPING = {
    "I1": 0,
    "SI2": 1,
    "SI1": 2,
    "VS2": 3,
    "VS1": 4,
    "VVS2": 5,
    "VVS1": 6,
    "IF": 7,
}


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the data-quality rules used in the original analysis.
    """

    cleaned_df = df[
        (df["x"] > 0)
        & (df["y"] > 0)
        & (df["z"] > 0)
        & (df["y"] < 30)
        & (df["z"] < 30)
    ].copy()

    return cleaned_df


def encode_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the ordinal mappings used in the original notebook.
    """

    df = df.copy()

    df["cut"] = df["cut"].map(CUT_MAPPING)
    df["color"] = df["color"].map(COLOR_MAPPING)
    df["clarity"] = df["clarity"].map(CLARITY_MAPPING)

    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the features used by Model C.
    """

    df = encode_categories(df)

    df["log_carat"] = np.log(df["carat"])

    features = [
        "log_carat",
        "cut",
        "color",
        "clarity",
        "depth",
        "table",
    ]

    return df[features]
