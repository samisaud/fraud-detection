"""
Stage 2 — featurize.py
Scaling, interaction features, SMOTE oversampling for class imbalance.
"""

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("featurize")

SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path) as f:
        return yaml.safe_load(f)


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Domain-inspired interaction features for fraud detection.
    Amount is the only interpretable non-anonymised column.
    """
    if "Amount" in df.columns:
        df = df.copy()
        df["Amount_log1p"] = np.log1p(df["Amount"])
        df["Amount_squared"] = df["Amount"] ** 2
        # V14 and V17 are typically the highest-importance PCA components
        if "V14" in df.columns and "V17" in df.columns:
            df["V14_V17_interact"] = df["V14"] * df["V17"]
        if "V10" in df.columns and "V12" in df.columns:
            df["V10_V12_interact"] = df["V10"] * df["V12"]
    return df


def main():
    params = load_params()
    base = params["base"]
    feat = params["features"]
    target = base["target_column"]

    train_df = pd.read_csv(params["data"]["processed_train"])
    test_df = pd.read_csv(params["data"]["processed_test"])

    log.info("Train shape: %s | Test shape: %s", train_df.shape, test_df.shape)

    # Add interaction features
    train_df = add_interaction_features(train_df)
    test_df = add_interaction_features(test_df)

    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    # Scaling — fit on train only, transform both
    scaler_cls = SCALERS.get(feat["scaling_method"], StandardScaler)
    scaler = scaler_cls()
    log.info("Fitting %s scaler on train ...", feat["scaling_method"])
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns
    )

    # Handle class imbalance
    if feat.get("handle_imbalance"):
        method = feat.get("imbalance_method", "smote")
        log.info(
            "Applying %s | before: %s positives",
            method,
            int(y_train.sum()),
        )
        if method == "smote":
            sampler = SMOTE(random_state=base["random_seed"])
        elif method == "undersample":
            sampler = RandomUnderSampler(random_state=base["random_seed"])
        else:
            sampler = None

        if sampler:
            X_train_scaled, y_train = sampler.fit_resample(X_train_scaled, y_train)
            log.info("After resampling: %d rows | %d positives", len(y_train), int(y_train.sum()))

    # Save artefacts
    Path("models").mkdir(exist_ok=True)
    joblib.dump(scaler, "models/scaler.joblib")
    log.info("Scaler saved to models/scaler.joblib")

    train_out = pd.concat(
        [pd.DataFrame(X_train_scaled), pd.Series(y_train, name=target).reset_index(drop=True)],
        axis=1,
    )
    test_out = pd.concat(
        [pd.DataFrame(X_test_scaled), pd.Series(y_test, name=target).reset_index(drop=True)],
        axis=1,
    )

    train_out.to_csv("data/processed/train_features.csv", index=False)
    test_out.to_csv("data/processed/test_features.csv", index=False)
    log.info("Features saved | train=%d test=%d", len(train_out), len(test_out))


if __name__ == "__main__":
    main()
