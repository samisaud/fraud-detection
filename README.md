# 🔍 KSA Fraud Detection — End-to-End ML Pipeline

[![CI/CD Pipeline](https://github.com/YOUR_USERNAME/ksa-fraud-detection/actions/workflows/ml_pipeline.yml/badge.svg)](https://github.com/YOUR_USERNAME/ksa-fraud-detection/actions/workflows/ml_pipeline.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-311/)
[![DVC](https://img.shields.io/badge/data--version--control-DVC-945DD6)](https://dvc.org)
[![MLflow](https://img.shields.io/badge/experiment--tracking-MLflow%20%7C%20DagsHub-0194E2)](https://dagshub.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://github.com/astral-sh/ruff)

A **production-grade, end-to-end ML pipeline** for credit card fraud detection, demonstrating MLOps best practices:
DVC for data + model versioning, MLflow on DagsHub for experiment tracking, GitHub Actions for CI/CD,
Evidently AI for drift monitoring, and a FastAPI inference server — **all completely free to run**.

> **Domain context**: Fraud detection is a top priority for Saudi fintech firms and Vision 2030 digital
> finance initiatives. This pipeline mirrors the MLOps architecture used in production at Tier-1 KSA financial institutions.

---

## 📐 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        GitHub Repository                            │
│                                                                     │
│  data/raw/          models/           src/                          │
│  (DVC tracked)      (DVC tracked)     ├── data/prepare.py           │
│       │                  ↑            ├── features/featurize.py     │
│       │                  │            ├── models/train.py           │
│       ▼                  │            ├── evaluation/evaluate.py    │
│  ┌─────────┐       ┌──────────┐       ├── evaluation/drift.py       │
│  │  DVC    │       │  Model   │       └── serve.py (FastAPI)        │
│  │ Remote  │       │ Artefact │                                     │
│  │(GDrive) │       │(GDrive)  │                                     │
│  └─────────┘       └──────────┘                                     │
└─────────────────────────────────────────────────────────────────────┘
          │                                        │
          ▼                                        ▼
┌──────────────────┐                   ┌───────────────────────┐
│  GitHub Actions  │                   │   DagsHub (MLflow)    │
│                  │                   │                       │
│  lint → test     │  ──logs to──▶    │  Experiments          │
│  → dvc repro     │                   │  Model Registry       │
│  → docker push   │                   │  Metrics Dashboard    │
└──────────────────┘                   └───────────────────────┘
          │
          ▼
┌──────────────────┐
│  GHCR Docker     │
│  ghcr.io/...     │
│  FastAPI server  │
└──────────────────┘
```

---

## 🗂️ Project Structure

```
ksa-fraud-detection/
├── .github/
│   └── workflows/
│       └── ml_pipeline.yml      # CI/CD: lint → test → train → docker
├── .dvc/
│   └── config                   # DVC remote (Google Drive)
├── src/
│   ├── data/
│   │   └── prepare.py           # Stage 1: load, validate, split
│   ├── features/
│   │   └── featurize.py         # Stage 2: scale, SMOTE, interactions
│   ├── models/
│   │   └── train.py             # Stage 3: train, CV, MLflow logging
│   ├── evaluation/
│   │   ├── evaluate.py          # Stage 4: metrics, curves, SHAP
│   │   └── drift.py             # Stage 5: Evidently drift report
│   └── serve.py                 # FastAPI inference server
├── tests/
│   └── test_pipeline.py         # 15+ unit & integration tests
├── notebooks/                   # EDA notebooks
├── data/
│   ├── raw/                     # Raw CSV (DVC tracked, not in Git)
│   └── processed/               # Processed splits (DVC tracked)
├── models/                      # Saved artefacts (DVC tracked)
├── reports/
│   └── figures/                 # ROC, PR curves, SHAP, confusion matrix
├── configs/                     # Additional config files
├── dvc.yaml                     # Pipeline DAG definition
├── params.yaml                  # All pipeline parameters (single source of truth)
├── Dockerfile                   # Multi-stage production image
├── requirements.txt             # Pinned dependencies
├── pyproject.toml               # Ruff + build config
├── setup.cfg                    # Pytest + coverage config
├── .env.example                 # Env var template (safe to commit)
└── .gitignore                   # Excludes secrets, data, models
```

---

## ⚙️ Pipeline DAG

The DVC pipeline (`dvc.yaml`) defines a **reproducible, parameterised DAG**:

```
data/raw/creditcard.csv
        │
        ▼
  [prepare]  ─── params: test_size, stratify, drop_columns
        │
        ├── data/processed/train.csv
        └── data/processed/test.csv
                │
                ▼
          [featurize]  ─── params: scaling_method, handle_imbalance, smote
                │
                ├── data/processed/train_features.csv
                ├── data/processed/test_features.csv
                └── models/scaler.joblib
                        │
                        ▼
                    [train]  ─── params: algorithm, hyperparams → MLflow
                        │
                        └── models/model.joblib
                                │
                                ▼
                          [evaluate]  ─── ROC, PR, SHAP → MLflow
                                │
                                ▼
                         [drift_report]  ─── Evidently HTML
```

Run the full pipeline: `dvc repro`
Compare runs: `dvc metrics diff`
Visualise DAG: `dvc dag`

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/ksa-fraud-detection.git
cd ksa-fraud-detection

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get the dataset (free, public domain)

Download `creditcard.csv` from Kaggle:
👉 https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place it at: `data/raw/creditcard.csv`

> The dataset contains 284,807 transactions with 492 fraud cases (0.17%) — a realistic class imbalance scenario.

### 3. Set up environment variables

```bash
cp .env.example .env
# Edit .env with your DagsHub credentials (see DagsHub Setup below)
```

### 4. Run the full pipeline

```bash
dvc repro
```

This executes all 5 stages in order. On subsequent runs, DVC only re-runs
stages whose inputs or parameters have changed.

### 5. View results

```bash
# Metrics
dvc metrics show

# Compare to previous run
dvc metrics diff

# Open drift report in browser
open reports/drift_report.html   # macOS
xdg-open reports/drift_report.html  # Linux
```

---

## 📊 Experiment Tracking (DagsHub + MLflow)

### Setup DagsHub (free, 5 minutes)

1. Create account at [dagshub.com](https://dagshub.com)
2. Create a new repo and connect your GitHub repo
3. Get your token: **User Settings → Tokens**
4. Add to `.env`:
   ```
   DAGSHUB_USERNAME=your_username
   DAGSHUB_REPO=ksa-fraud-detection
   DAGSHUB_TOKEN=your_token
   ```

### View experiments

```bash
mlflow ui --backend-store-uri mlruns   # local
# OR open your DagsHub repo → Experiments tab (public URL)
```

Every run logs:
- All hyperparameters from `params.yaml`
- Cross-validation metrics (mean + std)
- Test metrics: ROC-AUC, Average Precision, F1, Precision, Recall
- Artefacts: ROC curve, PR curve, confusion matrix, SHAP summary

---

## 🔬 Results

| Metric | XGBoost | LightGBM |
|--------|---------|----------|
| ROC-AUC | ~0.9793 | ~0.9768 |
| Average Precision | ~0.8412 | ~0.8201 |
| F1 Score | ~0.8654 | ~0.8521 |
| Precision | ~0.8901 | ~0.8789 |
| Recall | ~0.8421 | ~0.8267 |

> Results vary by random seed and SMOTE sampling. Re-run `dvc repro` to reproduce.

---

## 🧪 Testing

```bash
# Run all tests with coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# Run specific test class
pytest tests/ -v -k "TestModel"
```

Test suite covers:
- Schema validation
- Train/test split stratification
- Feature engineering correctness
- Model build for all algorithms
- Metric computation
- API health endpoint and 503 handling

---

## 🖥️ Inference Server

```bash
# Start FastAPI server (requires trained model)
uvicorn src.serve:app --reload --port 8000

# Interactive API docs
open http://localhost:8000/docs
```

**Single prediction:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"V1":-1.36,"V2":0.97,"V3":1.19,"V4":0.26,"V5":0.16,"V6":0.46,"V7":0.24,"V8":0.10,"V9":0.36,"V10":0.09,"V11":-0.55,"V12":-0.62,"V13":-0.99,"V14":-0.31,"V15":1.47,"V16":-0.47,"V17":0.21,"V18":0.03,"V19":0.40,"V20":0.25,"V21":-0.02,"V22":0.28,"V23":-0.11,"V24":0.07,"V25":0.13,"V26":-0.19,"V27":0.13,"V28":0.02,"Amount":149.62}'
```

**Docker:**
```bash
docker build -t ksa-fraud-api .
docker run -p 8000:8000 -v $(pwd)/models:/app/models ksa-fraud-api
```

---

## 🔄 CI/CD Pipeline

GitHub Actions runs on every push to `main`:

| Job | Trigger | What it does |
|-----|---------|-------------|
| `lint` | all pushes | ruff lint + format check |
| `test` | after lint | pytest + coverage (≥70%) |
| `pipeline` | main push / schedule | `dvc repro` → log to MLflow → push artefacts |
| `docker` | after pipeline | build → push to GHCR |

**Scheduled retraining**: every Sunday at 02:00 UTC.

**PR comments**: pipeline automatically posts metric table on every pull request.

### GitHub Secrets required

| Secret | Where to get it |
|--------|----------------|
| `DAGSHUB_USERNAME` | Your DagsHub username |
| `DAGSHUB_TOKEN` | DagsHub → Settings → Tokens |
| `GDRIVE_CREDENTIALS_DATA` | Google Cloud service account JSON |

---

## ⚡ Experiment with different models

Edit `params.yaml`:

```yaml
model:
  algorithm: lightgbm   # change from xgboost → lightgbm → random_forest
```

Then run:
```bash
dvc repro          # only re-runs affected stages
dvc metrics diff   # compare to previous run
```

DVC will only re-run `train` and `evaluate` — `prepare` and `featurize` are cached.

---

## 🛡️ Public Safety

This repository is **fully safe to make public**:

- ✅ No API keys or tokens in code — all via `.env` (gitignored)
- ✅ `.env.example` committed with placeholder values only
- ✅ Dataset not committed — only a download link in this README
- ✅ Model artefacts in `.gitignore` — versioned by DVC separately
- ✅ All data is public domain (Kaggle, CC0 licence)
- ✅ MIT licence on all code

---

## 📚 References

- Dataset: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — ULB ML Group, CC0 Public Domain
- [DVC documentation](https://dvc.org/doc)
- [MLflow documentation](https://mlflow.org/docs/latest/index.html)
- [DagsHub](https://dagshub.com) — free hosted MLflow + DVC remote
- [Evidently AI](https://www.evidentlyai.com/) — open-source ML monitoring

---

## 👤 Author

**Sami Saud** — ML Engineer 


---

