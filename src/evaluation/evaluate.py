"""
Stage 4 — evaluate.py
Full evaluation: metrics, curves, SHAP explainability, MLflow logging.
"""

import json
import logging
import os
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap
import yaml
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("evaluate")


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path) as f:
        return yaml.safe_load(f)


def save_roc_curve(fpr, tpr, auc_score: float, out_path: str) -> None:
    data = [
        {"fpr": float(f), "tpr": float(t)}
        for f, t in zip(fpr, tpr)
    ]
    with open(out_path, "w") as f:
        json.dump(data, f)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}", linewidth=2, color="#378ADD")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Fraud Detection")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path.replace(".json", ".png"), dpi=150)
    plt.close(fig)


def save_pr_curve(precision, recall, ap_score: float, out_path: str) -> None:
    data = [
        {"precision": float(p), "recall": float(r)}
        for p, r in zip(precision, recall)
    ]
    with open(out_path, "w") as f:
        json.dump(data, f)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, label=f"AP = {ap_score:.4f}", linewidth=2, color="#639922")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — Fraud Detection")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path.replace(".json", ".png"), dpi=150)
    plt.close(fig)


def save_confusion_matrix(cm: np.ndarray, out_path: str) -> None:
    data = cm.tolist()
    with open(out_path, "w") as f:
        json.dump({"confusion_matrix": data}, f)

    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legit", "Fraud"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(out_path.replace(".json", ".png"), dpi=150)
    plt.close(fig)


def compute_shap(model, X_sample: pd.DataFrame, out_dir: str) -> None:
    log.info("Computing SHAP values on %d samples ...", len(X_sample))
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(shap_values, X_sample, show=False, plot_size=None)
        fig = plt.gcf()
        fig.tight_layout()
        fig.savefig(f"{out_dir}/shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close("all")
        log.info("SHAP summary plot saved.")
    except Exception as e:
        log.warning("SHAP computation failed (non-fatal): %s", e)


def setup_mlflow(mlflow_cfg: dict) -> None:
    if not os.getenv("MLFLOW_TRACKING_USERNAME"):
        mlflow.set_tracking_uri("mlruns")
    else:
        mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])


def main():
    params = load_params()
    target = params["base"]["target_column"]
    eval_cfg = params["evaluation"]
    mlflow_cfg = params["mlflow"]

    test_df = pd.read_csv("data/processed/test_features.csv")
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    model = joblib.load("models/model.joblib")
    log.info("Model loaded | test rows: %d", len(X_test))

    # Predictions
    threshold = eval_cfg.get("threshold", 0.5)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # Core metrics
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "average_precision": float(average_precision_score(y_test, y_prob)),
        "f1": float(f1_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "threshold": threshold,
        "n_test_samples": len(y_test),
        "n_fraud_detected": int(y_pred.sum()),
        "n_actual_fraud": int(y_test.sum()),
    }
    log.info("Metrics: %s", metrics)

    Path("reports/figures").mkdir(parents=True, exist_ok=True)

    # Curves
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    save_roc_curve(fpr, tpr, metrics["roc_auc"], "reports/figures/roc_curve.json")

    precision_arr, recall_arr, _ = precision_recall_curve(y_test, y_prob)
    save_pr_curve(precision_arr, recall_arr, metrics["average_precision"], "reports/figures/pr_curve.json")

    cm = confusion_matrix(y_test, y_pred)
    save_confusion_matrix(cm, "reports/figures/confusion_matrix.json")

    # SHAP
    shap_sample = X_test.sample(
        min(eval_cfg.get("shap_sample_size", 500), len(X_test)),
        random_state=params["base"]["random_seed"],
    )
    compute_shap(model, shap_sample, "reports/figures")

    # Classification report
    report = classification_report(y_test, y_pred, target_names=["Legit", "Fraud"])
    log.info("\n%s", report)
    with open("reports/classification_report.txt", "w") as f:
        f.write(report)

    # Save metrics JSON for DVC tracking
    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Log to MLflow
    setup_mlflow(mlflow_cfg)
    with mlflow.start_run(run_name="evaluate") as run:
        mlflow.log_metrics(metrics)
        mlflow.log_artifact("reports/metrics.json")
        mlflow.log_artifact("reports/classification_report.txt")
        mlflow.log_artifact("reports/figures/roc_curve.png")
        mlflow.log_artifact("reports/figures/pr_curve.png")
        mlflow.log_artifact("reports/figures/confusion_matrix.png")
        if Path("reports/figures/shap_summary.png").exists():
            mlflow.log_artifact("reports/figures/shap_summary.png")
        log.info("Logged to MLflow run: %s", run.info.run_id)

    log.info("Evaluation complete. ROC-AUC=%.4f | Avg-Precision=%.4f", metrics["roc_auc"], metrics["average_precision"])


if __name__ == "__main__":
    main()
