"""
Stage 3 — train.py
Trains the configured model (XGBoost / LightGBM / RandomForest) with
stratified cross-validation and logs everything to MLflow.
"""

import json
import logging
import sys
import time
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("train")


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path) as f:
        return yaml.safe_load(f)


def build_model(algorithm: str, hyperparams: dict, random_seed: int):
    """Instantiate model and return (model, early_stopping_rounds)."""
    hp = hyperparams.get(algorithm, {}).copy()

    if algorithm == "xgboost":
        from xgboost import XGBClassifier

        early_stopping_rounds = hp.pop("early_stopping_rounds", None)
        model = XGBClassifier(
            **hp,
            random_state=random_seed,
            verbosity=0,
            eval_metric="aucpr",
        )
        return model, early_stopping_rounds

    if algorithm == "lightgbm":
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(**hp, random_state=random_seed, verbose=-1)
        return model, None

    if algorithm == "random_forest":
        model = RandomForestClassifier(**hp, random_state=random_seed, n_jobs=-1)
        return model, None

    raise ValueError(f"Unknown algorithm: {algorithm!r}. Choose xgboost | lightgbm | random_forest")


def cross_val_auprc(model, X: pd.DataFrame, y: pd.Series, cv: int, random_seed: int) -> dict:
    """Return mean ± std CV metrics (AUPRC headline, ROC-AUC secondary)."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_seed)
    results = cross_validate(
        model,
        X,
        y,
        cv=skf,
        scoring={"roc_auc": "roc_auc", "average_precision": "average_precision"},
        return_train_score=False,
        n_jobs=-1,
    )
    return {
        "cv_auprc_mean": float(np.mean(results["test_average_precision"])),
        "cv_auprc_std": float(np.std(results["test_average_precision"])),
        "cv_roc_auc_mean": float(np.mean(results["test_roc_auc"])),
        "cv_roc_auc_std": float(np.std(results["test_roc_auc"])),
    }


def main():
    params = load_params()
    base = params["base"]
    model_cfg = params["model"]
    mlflow_cfg = params.get("mlflow", {})

    algorithm = model_cfg["algorithm"]
    hyperparams = model_cfg["hyperparams"]
    random_seed = base["random_seed"]
    target = base["target_column"]

    train_df = pd.read_csv("data/processed/train_features.csv")
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]

    log.info(
        "Training %s | rows=%d | fraud=%d (%.2f%%)",
        algorithm,
        len(X_train),
        int(y_train.sum()),
        100 * y_train.mean(),
    )

    model, early_stopping_rounds = build_model(algorithm, hyperparams, random_seed)

    # ── Cross-validation (on a 90% slice so we keep 10% as eval set for XGB) ─────
    cv_metrics = cross_val_auprc(model, X_train, y_train, cv=5, random_seed=random_seed)
    log.info(
        "CV AUPRC: %.4f ± %.4f | CV ROC-AUC: %.4f ± %.4f",
        cv_metrics["cv_auprc_mean"],
        cv_metrics["cv_auprc_std"],
        cv_metrics["cv_roc_auc_mean"],
        cv_metrics["cv_roc_auc_std"],
    )

    # ── Final fit on full training set ───────────────────────────────────────────
    t0 = time.time()
    if algorithm == "xgboost" and early_stopping_rounds:
        # Hold out 10% of train for early stopping — never seen by cross-val
        from sklearn.model_selection import train_test_split

        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=random_seed, stratify=y_train
        )
        model.set_params(early_stopping_rounds=early_stopping_rounds)
        model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
        log.info("XGBoost best iteration: %d", model.best_iteration)
    else:
        model.fit(X_train, y_train)

    elapsed = time.time() - t0
    log.info("Fit complete in %.1fs", elapsed)

    # ── Train-set metrics (sanity check — not used for model selection) ──────────
    y_prob_train = model.predict_proba(X_train)[:, 1]
    train_auprc = float(average_precision_score(y_train, y_prob_train))
    train_roc_auc = float(roc_auc_score(y_train, y_prob_train))
    log.info("Train AUPRC=%.4f | Train ROC-AUC=%.4f", train_auprc, train_roc_auc)

    # ── Save artefacts ────────────────────────────────────────────────────────────
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/model.joblib")
    log.info("Model saved → models/model.joblib")

    train_metrics = {
        "algorithm": algorithm,
        "train_auprc": train_auprc,
        "train_roc_auc": train_roc_auc,
        "fit_time_seconds": round(elapsed, 2),
        **cv_metrics,
    }
    Path("reports").mkdir(exist_ok=True)
    with open("reports/train_metrics.json", "w") as f:
        json.dump(train_metrics, f, indent=2)
    log.info("Train metrics saved → reports/train_metrics.json")

    # ── MLflow logging ────────────────────────────────────────────────────────────
    tracking_uri = mlflow_cfg.get("tracking_uri", "mlruns")
    if "{DAGSHUB" in tracking_uri:
        tracking_uri = "mlruns"  # fall back to local when placeholder not filled
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "fraud-detection"))

    with mlflow.start_run(run_name=f"train-{algorithm}"):
        mlflow.log_param("algorithm", algorithm)
        mlflow.log_params(hyperparams.get(algorithm, {}))
        mlflow.log_metrics(train_metrics | {"fit_time_seconds": elapsed})
        mlflow.log_artifact("reports/train_metrics.json")
        mlflow.sklearn.log_model(model, artifact_path="model")

    log.info(
        "Training done. CV AUPRC=%.4f ± %.4f",
        cv_metrics["cv_auprc_mean"],
        cv_metrics["cv_auprc_std"],
    )


if __name__ == "__main__":
    main()
