"""
Stage 5 — drift.py
Generates an Evidently AI data drift + model performance report.
Runs locally, outputs a standalone HTML file — zero cloud needed.
"""

import logging
import sys
from pathlib import Path

import joblib
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("drift")


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path) as f:
        return yaml.safe_load(f)


def main():
    params = load_params()
    target = params["base"]["target_column"]
    ref_size = params["evaluation"].get("drift_reference_size", 5000)

    try:
        from evidently import ColumnMapping
        from evidently.metric_preset import (
            ClassificationPreset,
            DataDriftPreset,
            DataQualityPreset,
        )
        from evidently.report import Report
    except ImportError:
        log.error("evidently not installed. Run: pip install evidently")
        sys.exit(1)

    train_df = pd.read_csv("data/processed/train_features.csv")
    test_df = pd.read_csv("data/processed/test_features.csv")

    # Use a sample as reference baseline
    reference = train_df.sample(
        min(ref_size, len(train_df)), random_state=params["base"]["random_seed"]
    )
    current = test_df.copy()

    model = joblib.load("models/model.joblib")

    # Add prediction columns for model performance report
    feature_cols = [c for c in reference.columns if c != target]
    reference["prediction"] = model.predict_proba(reference[feature_cols])[:, 1]
    current["prediction"] = model.predict_proba(current[feature_cols])[:, 1]

    column_mapping = ColumnMapping(
        target=target,
        prediction="prediction",
        numerical_features=feature_cols,
    )

    report = Report(
        metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
            ClassificationPreset(),
        ]
    )

    log.info("Running Evidently report (reference=%d rows, current=%d rows) ...", len(reference), len(current))
    report.run(
        reference_data=reference,
        current_data=current,
        column_mapping=column_mapping,
    )

    Path("reports").mkdir(exist_ok=True)
    out_path = "reports/drift_report.html"
    report.save_html(out_path)
    log.info("Drift report saved to %s", out_path)


if __name__ == "__main__":
    main()
