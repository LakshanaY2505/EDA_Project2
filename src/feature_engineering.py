"""
feature_engineering.py
-----------------------
Feature engineering pipeline for the Telco churn dataset.
Expects the cleaned DataFrame produced by data_cleaning.clean_data().

Usage:
    from src.feature_engineering import build_features
    df_model = build_features(df_clean)
"""

import pandas as pd
import numpy as np


# ── Constants ──────────────────────────────────────────────────────────────────

SERVICE_COLS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

BINARY_COLS = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]

MULTI_CATEGORY_COLS = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod",
]

BINARY_MAP = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
BUCKET_MAP  = {"new": 0, "mid": 1, "long": 2}


# ── Individual feature functions ───────────────────────────────────────────────

def add_tenure_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bin tenure into new / mid / long segments.
        new  : 0–12 months
        mid  : 13–48 months
        long : 49+ months
    """
    df = df.copy()
    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 48, df["tenure"].max() + 1],
        labels=["new", "mid", "long"],
        right=True,
    )
    df["tenure_bucket"] = df["tenure_bucket"].map(BUCKET_MAP)
    return df


def add_high_charges(df: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
    """
    Binary flag: 1 if MonthlyCharges > threshold, else 0.
    Defaults to the dataset median if threshold is not supplied.
    """
    df = df.copy()
    if threshold is None:
        threshold = df["MonthlyCharges"].median()
    df["high_charges"] = (df["MonthlyCharges"] > threshold).astype(int)
    return df


def add_num_services(df: pd.DataFrame) -> pd.DataFrame:
    """Count of active add-on services per customer (0–6)."""
    df = df.copy()
    df["num_services"] = df[SERVICE_COLS].apply(
        lambda row: (row == "Yes").sum(), axis=1
    )
    return df


# ── Encoding helpers ───────────────────────────────────────────────────────────

def encode_binary_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Map Yes/No/Male/Female binary columns to 0/1."""
    df = df.copy()
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].map(BINARY_MAP)
    return df


def encode_categorical_cols(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode multi-category columns (drop first to avoid multicollinearity)."""
    df = df.copy()
    cols_present = [c for c in MULTI_CATEGORY_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=cols_present, drop_first=True)
    return df


# ── Main pipeline ──────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame, drop_total_charges: bool = True) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    Args:
        df: Cleaned DataFrame from data_cleaning.clean_data().
        drop_total_charges: Drop TotalCharges (highly correlated with tenure).

    Returns:
        Model-ready DataFrame with all features encoded as numeric.
    """
    print(f"[build_features] Input shape: {df.shape}")

    # New features
    df = add_tenure_bucket(df)
    df = add_high_charges(df)
    df = add_num_services(df)

    # Drop redundant column
    if drop_total_charges and "TotalCharges" in df.columns:
        df.drop(columns=["TotalCharges"], inplace=True)
        print("[build_features] Dropped TotalCharges (correlated with tenure)")

    # Encode
    df = encode_binary_cols(df)
    df = encode_categorical_cols(df)

    # Sanity checks
    assert df.isnull().sum().sum() == 0, "Nulls found after feature engineering!"
    assert (df.dtypes == "object").sum() == 0, "Non-numeric columns remain!"

    print(f"[build_features] Output shape: {df.shape}")
    print(f"[build_features] Feature columns: {[c for c in df.columns if c != 'Churn']}")
    return df


def split_X_y(df: pd.DataFrame):
    """Convenience: return (X, y) split from the model-ready DataFrame."""
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return X, y


if __name__ == "__main__":
    from data_cleaning import clean_data

    df_clean = clean_data("data/raw/Telco-Customer-Churn.csv")
    df_model = build_features(df_clean)
    df_model.to_csv("data/processed/featured_churn.csv", index=False)
    print("Saved → data/processed/featured_churn.csv")