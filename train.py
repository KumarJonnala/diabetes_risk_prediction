import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.data import load_data
from src.evaluate import compute_metrics
from src.config import load_config

# Config
cfg = load_config()
MODEL = cfg["model"]["type"]

# Data 
X_train, X_test, y_train, y_test = load_data(cfg["data"]["processed_path"])

# Model definitions
models = {
    "logistic_regression": LogisticRegression(**cfg["model"]["logistic_regression"]),
    "decision_tree":       DecisionTreeClassifier(**cfg["model"]["decision_tree"]),
    "random_forest":       RandomForestClassifier(**cfg["model"]["random_forest"], n_jobs=-1),
    "xgboost":             XGBClassifier(**cfg["model"]["xgboost"],
                               use_label_encoder=False, eval_metric="logloss"),
}

# MLflow
mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

with mlflow.start_run(run_name=MODEL):
    model = models[MODEL]
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_proba)

    mlflow.log_param("model_type", MODEL)
    mlflow.log_params(model.get_params())
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name=cfg["mlflow"]["registered_model_name"],
    )

    print(f"Model    : {MODEL}")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"F1       : {metrics['f1']:.4f}")
    print(f"ROC AUC  : {metrics['roc_auc']:.4f}")