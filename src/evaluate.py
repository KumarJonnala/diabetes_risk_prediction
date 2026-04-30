from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def compute_metrics(y_test, y_pred, y_proba) -> dict:
    """Return accuracy, F1, and ROC AUC as a dictionary."""
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1":       f1_score(y_test, y_pred),
        "roc_auc":  roc_auc_score(y_test, y_proba),
    }