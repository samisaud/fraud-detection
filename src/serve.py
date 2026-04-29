"""
serve.py — FastAPI inference server
Run: uvicorn src.serve:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

log = logging.getLogger("serve")
logging.basicConfig(level=logging.INFO)

MODEL_PATH = Path("models/model.joblib")
SCALER_PATH = Path("models/scaler.joblib")
METRICS_PATH = Path("reports/metrics.json")

DEFAULT_THRESHOLD = 0.5

model = None
scaler = None
optimal_threshold: float = DEFAULT_THRESHOLD


def _load_threshold() -> float:
    """Read the PR-curve-optimal threshold produced by evaluate.py."""
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
        threshold = float(metrics.get("optimal_threshold", DEFAULT_THRESHOLD))
        log.info("Loaded optimal threshold %.4f from %s", threshold, METRICS_PATH)
        return threshold
    log.warning("metrics.json not found — using default threshold %.2f", DEFAULT_THRESHOLD)
    return DEFAULT_THRESHOLD


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler, optimal_threshold
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        log.info("Model loaded from %s", MODEL_PATH)
    if SCALER_PATH.exists():
        scaler = joblib.load(SCALER_PATH)
        log.info("Scaler loaded from %s", SCALER_PATH)
    optimal_threshold = _load_threshold()
    yield


app = FastAPI(
    title="Fraud Detection API",
    description="End-to-end ML pipeline demo",
    version="1.0.0",
    lifespan=lifespan,
)


class TransactionFeatures(BaseModel):
    """Input features — V1-V28 are PCA-anonymised. Amount is transaction value."""

    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(..., ge=0, description="Transaction amount in USD")


class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    threshold: float
    model_version: str


class BatchPredictionRequest(BaseModel):
    transactions: list[TransactionFeatures]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run DVC pipeline first.")

    threshold = optimal_threshold
    df = pd.DataFrame([transaction.model_dump()])

    df["Amount_log1p"] = np.log1p(df["Amount"])
    df["Amount_squared"] = df["Amount"] ** 2
    if "V14" in df.columns and "V17" in df.columns:
        df["V14_V17_interact"] = df["V14"] * df["V17"]
    if "V10" in df.columns and "V12" in df.columns:
        df["V10_V12_interact"] = df["V10"] * df["V12"]

    if scaler is not None:
        try:
            df_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns)
        except Exception:
            df_scaled = df
    else:
        df_scaled = df

    prob = float(model.predict_proba(df_scaled)[:, 1][0])

    return PredictionResponse(
        fraud_probability=round(prob, 6),
        is_fraud=prob >= threshold,
        threshold=threshold,
        model_version="1.0.0",
    )


@app.post("/predict/batch")
def predict_batch(request: BatchPredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return {
        "predictions": [predict(txn) for txn in request.transactions],
        "count": len(request.transactions),
    }
