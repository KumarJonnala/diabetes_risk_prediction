# Diabetes Risk Prediction — MLOps Project

A step-by-step MLOps build around the Diabetes Health Indicators dataset.
Goal: train a model, track experiments, register the best version, and serve it behind a FastAPI endpoint running in Docker.

## Project layout (planned)

```
diabetes_risk_prediction/
├── configs/
│   └── config.yaml          # Single source of paths & hyperparams
├── dataset/
│   └── diabetes.csv         # Raw dataset (~253k rows)
├── documents/               # Project brief PDFs (reference)
├── src/
│   ├── __init__.py
│   ├── config.py            # YAML loader + path helpers
│   └── data.py              # Data loading + train/test split
├── notebooks/               # EDA and modeling experiments
├── models/                  # Local joblib artifacts (gitignored)
├── scripts/                 # CLI entry points (train.py, evaluate.py — coming)
├── tests/                   # pytest suite
├── main.py                  # FastAPI app (will load the model)
├── requirements.txt
├── .gitignore
└── README.md
```

## How the MLOps pieces will fit together

Roadmap 

| Stage | What it adds                                     | Tool             |
|------:|--------------------------------------------------|------------------|
| 1     | Reproducible data loading + config               | pandas + YAML    |
| 2     | Baseline model + evaluation                      | scikit-learn     |
| 3     | Experiment tracking (params, metrics, artifacts) | MLflow tracking  |
| 4     | Model registry (Staging / Production stages)     | MLflow registry  |
| 5     | Serving the registered model                     | FastAPI          |
| 6     | Containerized + reproducible runtime             | Docker           |
| 7     | Tests + CI                                       | pytest + GH Act. |
| 8     | Monitoring (inputs / predictions / drift)        | logging + later  |


## Setup

```bash
# create venv
python3 -m venv env
# activate the existing venv
source env/bin/activate

# install / refresh deps (adds mlflow, pytest, pyyaml, joblib)
pip install -r requirements.txt
```
