# Diabetes Risk Prediction — MLOps Project

MLOps pipeline on the CDC BRFSS Diabetes Health Indicators dataset.
Target: predict diabetes risk (binary) from 21 health indicators.

---

## Dataset

- **Source:** CDC BRFSS Diabetes Health Indicators
- **File:** `data/diabetes.csv` (~253,680 rows, 22 columns)
- **Target:** `Diabetes_012` collapsed to binary — 0 (no diabetes) vs 1 (pre/diabetes)
- **Class imbalance:** ~84% / 16%
- **Processed:** `data/diabetes_processed.csv` (cleaned, BMI_cat added, binary target)

---

## Project structure

```
diabetes_risk_prediction/
├── api/
│   ├── app.py                # FastAPI /predict endpoint — loads model from MLflow registry
│   └── Dockerfile            # API container — copies src/ and configs/ for imports
├── frontend/
│   ├── frontend.py           # Streamlit prediction form — calls FastAPI /predict
│   └── Dockerfile            # Frontend container — installs streamlit + requests
├── configs/
│   └── config.yaml           # model hyperparams, MLflow settings, data paths
├── data/
│   ├── diabetes.csv          # raw dataset (DVC-tracked, not in Git)
│   ├── diabetes.csv.dvc      # DVC pointer file (in Git)
│   ├── diabetes_processed.csv# cleaned + feature-engineered (DVC-tracked)
│   └── diabetes_processed.csv.dvc # DVC pointer file (in Git)
├── documents/                # reference material (gitignored)
├── notebooks/
│   ├── test_notebook.ipynb   # EDA + 4-model comparison
│   └── test2_notebook.ipynb  # additional experiments
├── src/
│   ├── __init__.py           # makes src/ a Python package
│   ├── config.py             # YAML loader
│   ├── data.py               # build_processed_data() + load_data()
│   ├── preprocess.py         # clean_data() + make_target()
│   ├── features.py           # engineer_features() — adds BMI_cat
│   └── evaluate.py           # compute_metrics() — accuracy, F1, ROC AUC
├── tests/
│   ├── __init__.py
│   ├── test_features.py      # unit tests for bmi_category + engineer_features
│   ├── test_preprocess.py    # unit tests for clean_data + make_target
│   └── test_api.py           # API contract tests via FastAPI TestClient (MLflow mocked)
├── train.py                  # config-driven training + MLflow tracking + auto-registration
├── .dvc/                     # DVC config and cache
├── docker-compose.yml        # all services: MLflow + FastAPI + Streamlit
├── mlflow_data/              # MLflow DB + artifacts (gitignored, persists across restarts)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

```bash
# create and activate virtual environment
python3 -m venv env
source env/bin/activate

# install dependencies
pip install -r requirements.txt
```

---

## Running commands

### Start the full stack
All three services (MLflow, FastAPI, Streamlit) run via Docker Compose:
```bash
docker compose up --build  # first run — builds images and starts all services
docker compose up          # subsequent runs — reuses cached images
docker compose down        # stop and remove containers
docker compose ps          # check running containers
docker compose logs api    # view logs for a specific service
docker compose restart api # restart a single service without full teardown
```

Once running:
- MLflow UI → `http://localhost:5001` (Experiments + Model Registry)
- FastAPI docs → `http://localhost:8000/docs`
- Streamlit app → `http://localhost:8501`

Data and model artifacts persist in `mlflow_data/` across container restarts.

> **Note:** Train the model locally at least once (`python3 train.py`) before starting the stack.
> The API container loads the model from the MLflow registry on startup and will crash if no
> `@production` model is registered yet.

### Start MLflow only (for local training)
If running `train.py` locally without the full Docker stack:
```bash
docker compose up mlflow -d   # start MLflow container only
docker compose down           # stop when done
```
MLflow UI: `http://localhost:5001`
- **Experiments tab** — all runs, metrics, params, artifacts
- **Models tab** — registered model versions and their aliases (e.g. `@production`)

Data persists in `mlflow_data/` even after the container stops.

### DVC — data versioning
Data files are tracked by DVC, not Git. The `.dvc` pointer files go to Git; the actual CSVs go to DVC remote storage.

```bash
# download data files (after cloning the repo)
dvc pull

# push data to remote after changes
dvc push

# check status
dvc status
```

Local remote is stored in `.dvc_remote/` (gitignored). To switch to sotorage for CI/CD, update `.dvc/config`.

### Run tests
```bash
python -m pytest tests/ -v
```
13 tests across three files — no MLflow connection or data files required:
- `test_features.py` — unit tests for BMI categorisation and feature engineering
- `test_preprocess.py` — unit tests for data cleaning and target creation
- `test_api.py` — API contract tests using FastAPI TestClient (MLflow is mocked)

### Build processed dataset
Only needed once, or when raw data changes:
```bash
python3 -c "from src.data import build_processed_data; build_processed_data()"
```

### Train a model
Model type is set in `configs/config.yaml` under `model.type`:
```bash
python3 train.py
```
Logs params, metrics, and the model artifact to the MLflow container.
Also registers the model to the MLflow Model Registry as `DiabetesRiskModel`.

### Start the API (local, without Docker)
```bash
uvicorn api.app:app --reload
```
Open `http://127.0.0.1:8000/docs` for the interactive Swagger UI. Requires the MLflow container running first.

### Start the Streamlit frontend (local, without Docker)
```bash
streamlit run frontend/frontend.py
```
Open `http://localhost:8501` in your browser. Requires the FastAPI server and MLflow container running first.

---

## Model results (from notebook comparison)

| Model               | Accuracy | F1     | ROC AUC       |
|---------------------|----------|--------|---------------|
| Logistic Regression | 0.7145   | 0.4766 | 0.8026        |
| Decision Tree       | 0.6995   | 0.4655 | 0.7965        |
| **Random Forest**   | **0.7250**| **0.4811** | **0.8063** |
| XGBoost             | 0.7072   | 0.4742 | 0.8056        |

All models trained with `class_weight='balanced'` to handle the 84/16 class imbalance.
Primary metric: **ROC AUC** (accuracy misleading on imbalanced data).

---

## MLOps roadmap

| Stage | What it adds                                      | Tool                      | Status      |
|------:|---------------------------------------------------|---------------------------|-------------|
| 1     | Data cleaning + feature engineering               | pandas, scikit-learn      |    Done     |
| 2     | Baseline model + evaluation                       | scikit-learn              |    Done     |
| 3     | Experiment tracking (params, metrics, artifacts)  | MLflow tracking           |    Done     |
| 4     | Model Registry (version + stage management)       | MLflow registry           |    Done     |
| 5     | Data versioning                                   | DVC + local/S3 remote     |    Done     |
| 6     | Model serving — `/predict` endpoint               | FastAPI                   |    Done     |
| 7     | Frontend — prediction form + result display       | Streamlit → FastAPI       |    Done     |
| 8     | MLflow container (local tracking server)          | Docker Compose            |    Done     |
| 9     | Containerise all services (FastAPI + Streamlit)   | Docker Compose            |    Done     |
| 10    | CI/CD pipeline                                    | GitHub Actions            |    Pending  |
| 11    | Feature store                                     | Feast                     |    Pending  |
| 12    | Monitoring — metrics push                         | Prometheus Pushgateway    |    Pending  |
| 13    | Monitoring — dashboards                           | Grafana                   |    Pending  |

---
