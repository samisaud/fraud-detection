"""
tests/test_pipeline.py
Unit and integration tests for the full ML pipeline.
Run: pytest tests/ -v --cov=src --cov-report=term-missing
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def synthetic_fraud_df():
    """Small synthetic dataset that mirrors the credit card fraud schema."""
    X, y = make_classification(
        n_samples=1000,
        n_features=28,
        n_informative=15,
        n_redundant=5,
        weights=[0.98, 0.02],  # realistic imbalance
        random_state=42,
    )
    cols = [f"V{i}" for i in range(1, 29)]
    df = pd.DataFrame(X, columns=cols)
    df["Amount"] = np.abs(np.random.exponential(scale=100, size=1000))
    df["is_fraud"] = y
    return df


@pytest.fixture
def sample_params():
    return {
        "base": {"random_seed": 42, "target_column": "is_fraud", "project_name": "test"},
        "data": {
            "raw_path": "data/raw/creditcard.csv",
            "processed_train": "data/processed/train.csv",
            "processed_test": "data/processed/test.csv",
            "test_size": 0.2,
            "stratify": True,
        },
        "features": {
            "scaling_method": "standard",
            "handle_imbalance": False,
            "imbalance_method": "smote",
            "drop_columns": ["Time"],
        },
        "model": {
            "algorithm": "random_forest",
            "hyperparams": {
                "random_forest": {"n_estimators": 10, "max_depth": 3, "class_weight": "balanced"}
            },
        },
        "evaluation": {"threshold": 0.5, "shap_sample_size": 50, "drift_reference_size": 100},
        "mlflow": {
            "tracking_uri": "mlruns",
            "experiment_name": "test-exp",
            "model_registry_name": "TestModel",
        },
    }


# ─── Data tests ────────────────────────────────────────────────────────────────


class TestDataPreparation:
    def test_schema_validation_passes(self, synthetic_fraud_df):
        from src.data.prepare import validate_schema

        validate_schema(synthetic_fraud_df, "is_fraud")  # should not raise

    def test_schema_validation_missing_target(self, synthetic_fraud_df):
        from src.data.prepare import validate_schema

        with pytest.raises(AssertionError):
            validate_schema(synthetic_fraud_df.drop(columns=["is_fraud"]), "is_fraud")

    def test_compute_stats(self, synthetic_fraud_df):
        from src.data.prepare import compute_stats

        stats = compute_stats(synthetic_fraud_df, "is_fraud")
        assert "n_rows" in stats
        assert "positive_rate" in stats
        assert stats["n_rows"] == 1000
        assert 0 < stats["positive_rate"] < 1

    def test_train_test_split_stratified(self, synthetic_fraud_df):
        from sklearn.model_selection import train_test_split

        X = synthetic_fraud_df.drop(columns=["is_fraud"])
        y = synthetic_fraud_df["is_fraud"]
        _, _, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        # Fraud rate should be similar in both splits
        assert abs(y_train.mean() - y_test.mean()) < 0.01


# ─── Feature tests ─────────────────────────────────────────────────────────────


class TestFeaturization:
    def test_interaction_features_added(self, synthetic_fraud_df):
        from src.features.featurize import add_interaction_features

        df = add_interaction_features(synthetic_fraud_df)
        assert "Amount_log1p" in df.columns
        assert "Amount_squared" in df.columns
        assert (df["Amount_log1p"] >= 0).all()

    def test_amount_log1p_non_negative(self, synthetic_fraud_df):
        from src.features.featurize import add_interaction_features

        df = add_interaction_features(synthetic_fraud_df)
        assert (df["Amount_log1p"] >= 0).all()

    def test_no_nulls_after_featurization(self, synthetic_fraud_df):
        from src.features.featurize import add_interaction_features

        df = add_interaction_features(synthetic_fraud_df)
        assert df.isnull().sum().sum() == 0

    def test_scaler_fit_transform(self, synthetic_fraud_df):
        from sklearn.preprocessing import StandardScaler

        feature_cols = [c for c in synthetic_fraud_df.columns if c != "is_fraud"]
        X = synthetic_fraud_df[feature_cols]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        # Mean should be ~0 and std ~1 after scaling
        assert abs(X_scaled.mean()) < 0.1
        assert abs(X_scaled.std() - 1.0) < 0.1


# ─── Evaluation tests ──────────────────────────────────────────────────────────


class TestEvaluation:
    def test_metrics_computed(self, synthetic_fraud_df):
        import warnings; warnings.filterwarnings("ignore")
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import average_precision_score, roc_auc_score
        from sklearn.model_selection import train_test_split

        X = synthetic_fraud_df.drop(columns=["is_fraud"])
        y = synthetic_fraud_df["is_fraud"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, probs)
        ap = average_precision_score(y_test, probs)

        assert 0.0 <= auc <= 1.0
        assert 0.0 < ap <= 1.0

    def test_confusion_matrix_shape(self, synthetic_fraud_df):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import confusion_matrix
        from sklearn.model_selection import train_test_split

        X = synthetic_fraud_df.drop(columns=["is_fraud"])
        y = synthetic_fraud_df["is_fraud"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        clf = RandomForestClassifier(n_estimators=5, random_state=42)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        cm = confusion_matrix(y_test, preds)
        assert cm.shape == (2, 2)


# ─── API tests ─────────────────────────────────────────────────────────────────


class TestAPI:
    def test_health_endpoint(self):
        from fastapi.testclient import TestClient

        from src.serve import app

        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "ok"

    def test_predict_endpoint_no_model(self):
        """When model is not loaded, should return 503."""
        import src.serve as serve_module

        original = serve_module.model
        serve_module.model = None

        from fastapi.testclient import TestClient

        from src.serve import app

        client = TestClient(app)

        payload = {f"V{i}": 0.0 for i in range(1, 29)}
        payload["Amount"] = 100.0
        response = client.post("/predict", json=payload)
        assert response.status_code == 503

        serve_module.model = original
