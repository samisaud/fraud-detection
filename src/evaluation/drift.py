import logging
import sys
from pathlib import Path

import joblib
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", stream=sys.stdout)
log = logging.getLogger("drift")

params = yaml.safe_load(open("params.yaml", encoding="utf-8"))  # noqa: SIM115
target = params["base"]["target_column"]

train_df = pd.read_csv("data/processed/train_features.csv")
test_df = pd.read_csv("data/processed/test_features.csv")
model = joblib.load("models/model.joblib")

ref_size = params["evaluation"].get("drift_reference_size", 5000)
reference = train_df.sample(min(ref_size, len(train_df)), random_state=42)
current = test_df.copy()
feature_cols = [c for c in reference.columns if c != target]
reference["prediction"] = model.predict_proba(reference[feature_cols])[:, 1]
current["prediction"] = model.predict_proba(current[feature_cols])[:, 1]

Path("reports").mkdir(exist_ok=True)

try:
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset
    from evidently.pipeline.column_mapping import ColumnMapping
    from evidently.report import Report

    col_map = ColumnMapping(
        target=target,
        prediction="prediction",
        numerical_features=feature_cols,
    )
    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    report.run(reference_data=reference, current_data=current, column_mapping=col_map)
    report.save_html("reports/drift_report.html")
    log.info("Drift report saved to reports/drift_report.html")
except Exception as e:
    log.warning("Evidently failed: %s — writing placeholder", e)
    Path("reports/drift_report.html").write_text(
        "<html><body><h1>Drift Report</h1><p>Run locally with compatible evidently version.</p></body></html>"
    )
    log.info("Placeholder saved.")
