"""
Stage 1 — prepare.py
Loads raw CSV, validates schema, splits into train/test, saves stats.
Dataset: Kaggle Credit Card Fraud Detection (public domain)
  → https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
"""

import json
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("prepare")


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path) as f:
        return yaml.safe_load(f)


def validate_schema(df: pd.DataFrame, target: str) -> None:
    """Minimal schema checks — fail fast before any processing."""
    assert target in df.columns, f"Target column '{target}' not found"
    assert len(df) > 0, "Dataset is empty"
    null_pct = df.isnull().mean()
    high_null = null_pct[null_pct > 0.5]
    if not high_null.empty:
        log.warning("Columns with >50%% nulls: %s", high_null.to_dict())
    log.info("Schema OK | rows=%d cols=%d", *df.shape)


def compute_stats(df: pd.DataFrame, target: str) -> dict:
    return {
        "n_rows": int(len(df)),
        "n_features": int(df.shape[1] - 1),
        "n_positive": int(df[target].sum()),
        "n_negative": int((df[target] == 0).sum()),
        "positive_rate": float(df[target].mean()),
        "null_count": int(df.isnull().sum().sum()),
    }


def main():
    params = load_params()
    base = params["base"]
    data = params["data"]
    features = params["features"]

    raw_path = Path(data["raw_path"])
    if not raw_path.exists():
        log.error(
            "Raw data not found at %s\n"
            "Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            "Place creditcard.csv in data/raw/",
            raw_path,
        )
        sys.exit(1)

    log.info("Loading %s ...", raw_path)
    df = pd.read_csv(raw_path)

    validate_schema(df, base["target_column"])

    # Drop configured columns
    drop_cols = [c for c in features.get("drop_columns", []) if c in df.columns]
    if drop_cols:
        log.info("Dropping columns: %s", drop_cols)
        df = df.drop(columns=drop_cols)

    # Compute and save stats before split
    stats = compute_stats(df, base["target_column"])
    log.info("Dataset stats: %s", stats)

    Path("reports").mkdir(exist_ok=True)
    with open("reports/data_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Stratified train/test split
    stratify = df[base["target_column"]] if data.get("stratify") else None
    X = df.drop(columns=[base["target_column"]])
    y = df[base["target_column"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=data["test_size"],
        random_state=base["random_seed"],
        stratify=stratify,
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    train_df.to_csv(data["processed_train"], index=False)
    test_df.to_csv(data["processed_test"], index=False)

    log.info(
        "Split complete | train=%d test=%d positive_rate=%.4f",
        len(train_df),
        len(test_df),
        stats["positive_rate"],
    )


if __name__ == "__main__":
    main()
