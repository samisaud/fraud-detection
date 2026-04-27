"""
Stage 4 — evaluate.py
Full evaluation: AUPRC as headline metric, ROC, SHAP, confusion matrix.
AUPRC (Average Precision) is the correct metric for fraud detection —
not ROC-AUC which is misleading on heavily imbalanced datasets.
"""

import json
import logging
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", stream=sys.stdout)
log = logging.getLogger("evaluate")


def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def find_best_threshold(y_true, y_prob):
    """
    Find threshold that maximises F1 on the PR curve.
    Critical for fraud: default 0.5 threshold is almost always wrong
    on imbalanced data — this finds the optimal operating point.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    log.info(
        "Optimal threshold: %.4f (F1=%.4f P=%.4f R=%.4f)",
        best_threshold, f1_scores[best_idx],
        precision[best_idx], recall[best_idx],
    )
    return float(best_threshold)


def main():
    params = load_params()
    target = params["base"]["target_column"]

    test_df = pd.read_csv("data/processed/test_features.csv")
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    model = joblib.load("models/model.joblib")
    log.info("Model loaded | test rows: %d | fraud cases: %d", len(X_test), int(y_test.sum()))

    y_prob = model.predict_proba(X_test)[:, 1]

    # --- Find optimal threshold (not hardcoded 0.5) ---
    threshold = find_best_threshold(y_test, y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    # --- AUPRC is the headline metric for fraud ---
    auprc = float(average_precision_score(y_test, y_prob))
    roc_auc = float(roc_auc_score(y_test, y_prob))

    metrics = {
        # AUPRC first — this is what matters for imbalanced fraud data
        "auprc": auprc,
        "roc_auc": roc_auc,
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "optimal_threshold": threshold,
        "n_test_samples": int(len(y_test)),
        "n_actual_fraud": int(y_test.sum()),
        "n_fraud_detected": int(y_pred.sum()),
        "n_false_negatives": int(((y_test == 1) & (y_pred == 0)).sum()),
        "n_false_positives": int(((y_test == 0) & (y_pred == 1)).sum()),
    }

    log.info("=" * 50)
    log.info("HEADLINE: AUPRC = %.4f  (ROC-AUC = %.4f)", auprc, roc_auc)
    log.info("F1=%.4f  Precision=%.4f  Recall=%.4f", metrics["f1"], metrics["precision"], metrics["recall"])
    log.info("Fraud detected: %d / %d | False negatives: %d",
             metrics["n_fraud_detected"], metrics["n_actual_fraud"], metrics["n_false_negatives"])
    log.info("=" * 50)

    Path("reports/figures").mkdir(parents=True, exist_ok=True)

    # --- PR Curve (more informative than ROC for fraud) ---
    precision_arr, recall_arr, _ = precision_recall_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall_arr, precision_arr, color="#00d4aa", linewidth=2.5,
            label=f"AUPRC = {auprc:.4f}  ← headline metric")
    ax.axvline(x=metrics["recall"], color="white", linestyle="--", alpha=0.5,
               label=f"Operating point (threshold={threshold:.3f})")
    ax.set_xlabel("Recall (Fraud Caught Rate)")
    ax.set_ylabel("Precision (Alert Accuracy)")
    ax.set_title("Precision-Recall Curve\n(AUPRC is the correct metric for imbalanced fraud data)")
    ax.legend()
    ax.set_facecolor("#0e1117")
    fig.patch.set_facecolor("#0e1117")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    fig.tight_layout()
    fig.savefig("reports/figures/pr_curve.png", dpi=150, facecolor="#0e1117")
    plt.close(fig)

    # Save PR curve data for Streamlit
    pr_data = [{"precision": float(p), "recall": float(r)}
               for p, r in zip(precision_arr, recall_arr, strict=False)]
    with open("reports/figures/pr_curve.json", "w") as f:
        json.dump(pr_data, f)

    # --- ROC Curve ---
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#378ADD", linewidth=2.5, label=f"ROC-AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "w--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve\n(Note: misleading for imbalanced data — use AUPRC instead)")
    ax.legend()
    ax.set_facecolor("#0e1117")
    fig.patch.set_facecolor("#0e1117")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    fig.tight_layout()
    fig.savefig("reports/figures/roc_curve.png", dpi=150, facecolor="#0e1117")
    plt.close(fig)

    roc_data = [{"fpr": float(f), "tpr": float(t)} for f, t in zip(fpr, tpr, strict=False)]
    with open("reports/figures/roc_curve.json", "w") as f:
        json.dump(roc_data, f)

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legit", "Fraud"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig("reports/figures/confusion_matrix.png", dpi=150)
    plt.close(fig)
    with open("reports/figures/confusion_matrix.json", "w") as f:
        json.dump({"confusion_matrix": cm.tolist()}, f)

    # --- SHAP (optional — skip if not installed) ---
    try:
        import shap
        shap_n = min(300, len(X_test))
        X_sample = X_test.sample(shap_n, random_state=42)
        log.info("Computing SHAP on %d samples...", shap_n)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        shap.summary_plot(shap_values[1] if isinstance(shap_values, list) else shap_values,
                          X_sample, show=False)
        plt.savefig("reports/figures/shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close("all")
        log.info("SHAP saved.")
    except Exception as e:
        log.warning("SHAP skipped: %s", e)

    # --- Classification report ---
    report = classification_report(y_test, y_pred, target_names=["Legit", "Fraud"])
    log.info("\n%s", report)
    with open("reports/classification_report.txt", "w") as f:
        f.write(f"Optimal threshold: {threshold:.4f}\n\n")
        f.write(report)

    # --- Save metrics ---
    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # --- Log to MLflow ---
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("fraud-detection")
    with mlflow.start_run(run_name="evaluate"):
        mlflow.log_metrics(metrics)
        mlflow.log_artifact("reports/metrics.json")
        mlflow.log_artifact("reports/classification_report.txt")
        for fig_file in Path("reports/figures").glob("*.png"):
            mlflow.log_artifact(str(fig_file))

    log.info("Evaluation complete.")
    log.info("AUPRC=%.4f | ROC-AUC=%.4f | F1=%.4f | Recall=%.4f",
             auprc, roc_auc, metrics["f1"], metrics["recall"])


if __name__ == "__main__":
    main()
