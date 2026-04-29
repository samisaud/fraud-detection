.PHONY: help setup lint test train serve dashboard docker-build clean

PYTHON := python
VENV   := .venv

help:
	@echo ""
	@echo "  Fraud Detection — MLOps Pipeline"
	@echo ""
	@echo "  Setup"
	@echo "    make setup        Create venv and install dependencies"
	@echo ""
	@echo "  Quality"
	@echo "    make lint         Run ruff linter"
	@echo "    make test         Run test suite with coverage"
	@echo ""
	@echo "  Pipeline (requires DVC + raw data)"
	@echo "    make pipeline     Run full DVC pipeline (prepare → train → evaluate)"
	@echo "    make prepare      Stage 1: prepare raw data"
	@echo "    make featurize    Stage 2: feature engineering"
	@echo "    make train        Stage 3: train model"
	@echo "    make evaluate     Stage 4: evaluate + SHAP"
	@echo "    make drift        Stage 5: drift report"
	@echo ""
	@echo "  Serving"
	@echo "    make serve        Start FastAPI inference server (port 8000)"
	@echo "    make dashboard    Start Streamlit dashboard (port 8501)"
	@echo ""
	@echo "  Docker"
	@echo "    make docker-build  Build all Docker images"
	@echo "    make docker-up     Start full stack (API + dashboard)"
	@echo "    make docker-down   Stop stack"
	@echo ""
	@echo "  Housekeeping"
	@echo "    make clean        Remove generated artefacts"

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt
	@echo "Done. Activate with: source $(VENV)/bin/activate"

lint:
	ruff check src/ tests/

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

# ── Pipeline stages ────────────────────────────────────────────────────────────

pipeline:
	dvc repro

prepare:
	$(PYTHON) src/data/prepare.py

featurize:
	$(PYTHON) src/features/featurize.py

train:
	$(PYTHON) src/models/train.py

evaluate:
	$(PYTHON) src/evaluation/evaluate.py

drift:
	$(PYTHON) src/evaluation/drift.py

# ── Serving ───────────────────────────────────────────────────────────────────

serve:
	uvicorn src.serve:app --reload --port 8000

dashboard:
	streamlit run app.py --server.port 8501

# ── Docker ────────────────────────────────────────────────────────────────────

docker-build:
	docker build -t fraud-api:latest -f Dockerfile .
	docker build -t fraud-dashboard:latest -f Dockerfile.streamlit .

docker-up:
	docker compose up -d

docker-down:
	docker compose down

# ── Housekeeping ──────────────────────────────────────────────────────────────

clean:
	rm -rf reports/figures/*.png reports/figures/*.json
	rm -f reports/metrics.json reports/train_metrics.json reports/data_stats.json
	rm -f reports/classification_report.txt reports/drift_report.html
	rm -rf mlruns/ __pycache__ .pytest_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
