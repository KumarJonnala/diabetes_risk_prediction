import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, computed_field
from typing import Annotated
import pandas as pd
import mlflow
import mlflow.sklearn

# pydantic model to validate input data
from src.config import load_config
from src.features import bmi_category

# import model from MLflow registry
cfg = load_config()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"]))
model = mlflow.sklearn.load_model(f"models:/{cfg['mlflow']['registered_model_name']}@production")

app = FastAPI()

FEATURE_COLUMNS = [
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
    'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
    'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
    'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education',
    'Income', 'BMI_cat'
]

class UserInput(BaseModel):
    # binary fields (0/1)
    HighBP:               Annotated[int, Field(..., ge=0, le=1)]
    HighChol:             Annotated[int, Field(..., ge=0, le=1)]
    CholCheck:            Annotated[int, Field(..., ge=0, le=1)]
    Smoker:               Annotated[int, Field(..., ge=0, le=1)]
    Stroke:               Annotated[int, Field(..., ge=0, le=1)]
    HeartDiseaseorAttack: Annotated[int, Field(..., ge=0, le=1)]
    PhysActivity:         Annotated[int, Field(..., ge=0, le=1)]
    Fruits:               Annotated[int, Field(..., ge=0, le=1)]
    Veggies:              Annotated[int, Field(..., ge=0, le=1)]
    HvyAlcoholConsump:    Annotated[int, Field(..., ge=0, le=1)]
    AnyHealthcare:        Annotated[int, Field(..., ge=0, le=1)]
    NoDocbcCost:          Annotated[int, Field(..., ge=0, le=1)]
    DiffWalk:             Annotated[int, Field(..., ge=0, le=1)]
    Sex:                  Annotated[int, Field(..., ge=0, le=1)]

    # numeric fields
    BMI:      Annotated[float, Field(..., ge=10.0, le=100.0)]
    GenHlth:  Annotated[int, Field(..., ge=1, le=5)]
    MentHlth: Annotated[int, Field(..., ge=0, le=30)]
    PhysHlth: Annotated[int, Field(..., ge=0, le=30)]
    Age:      Annotated[int, Field(..., ge=1, le=13)]
    Education:Annotated[int, Field(..., ge=1, le=6)]
    Income:   Annotated[int, Field(..., ge=1, le=8)]

    # engineered features
    @computed_field
    def BMI_cat(self) -> int:
        return bmi_category(self.BMI)
    

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/about")
def about():
    return {"message": "This API predicts the risk of diabetes based on user input features."}

@app.post("/predict")
def predict(user_input: UserInput):
    df = pd.DataFrame([user_input.model_dump()])[FEATURE_COLUMNS]
    probability = model.predict_proba(df)[0][1]  # probability of class 1
    risk_percent = round(probability * 100, 2)
    label = "High Risk" if probability >= 0.5 else "Low Risk"

    return JSONResponse(status_code=200, content={
        "diabetes_risk_score": risk_percent,
        "risk_label": label,
    })