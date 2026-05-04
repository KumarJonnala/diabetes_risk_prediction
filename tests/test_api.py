from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# Patch mlflow.sklearn.load_model before importing app.py as it calls MLflow in CI and fails without it 

_mock_model = MagicMock()
_mock_model.predict_proba.return_value = [[0.7, 0.3]]  # default: low risk

with patch("mlflow.sklearn.load_model", return_value=_mock_model):
    from api.app import app

client = TestClient(app)

# valid payload used across multiple tests

VALID_PAYLOAD = {
    "HighBP": 0, "HighChol": 0, "CholCheck": 1,
    "Smoker": 0, "Stroke": 0, "HeartDiseaseorAttack": 0,
    "PhysActivity": 1, "Fruits": 1, "Veggies": 1,
    "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0,
    "DiffWalk": 0, "Sex": 0,
    "BMI": 24.0, "GenHlth": 2, "MentHlth": 0,
    "PhysHlth": 0, "Age": 5, "Education": 4, "Income": 5,
}

# utility endpoints

def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_about_returns_message():
    response = client.get("/about")
    assert response.status_code == 200
    assert "message" in response.json()

# /predict — happy path

def test_predict_returns_200_with_valid_input():
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200

def test_predict_response_shape():
    """Response must contain diabetes_risk_score and risk_label."""
    response = client.post("/predict", json=VALID_PAYLOAD)
    body = response.json()
    assert "diabetes_risk_score" in body
    assert "risk_label" in body

def test_predict_low_risk_label():
    """Model returns probability 0.3 for class 1 → Low Risk."""
    _mock_model.predict_proba.return_value = [[0.7, 0.3]]
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.json()["risk_label"] == "Low Risk"

def test_predict_high_risk_label():
    """Model returns probability 0.8 for class 1 → High Risk."""
    _mock_model.predict_proba.return_value = [[0.2, 0.8]]
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.json()["risk_label"] == "High Risk"

def test_predict_score_is_percentage():
    """Risk score must be between 0 and 100."""
    _mock_model.predict_proba.return_value = [[0.6, 0.4]]
    response = client.post("/predict", json=VALID_PAYLOAD)
    score = response.json()["diabetes_risk_score"]
    assert 0 <= score <= 100

# /predict — validation errors

def test_predict_rejects_bmi_too_high():
    payload = {**VALID_PAYLOAD, "BMI": 200.0}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_predict_rejects_bmi_too_low():
    payload = {**VALID_PAYLOAD, "BMI": 5.0}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_predict_rejects_missing_field():
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "BMI"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_predict_rejects_invalid_binary_field():
    """Binary fields only accept 0 or 1 — value 2 should be rejected."""
    payload = {**VALID_PAYLOAD, "HighBP": 2}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
