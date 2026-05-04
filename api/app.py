import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, computed_field
from typing import Annotated
import pandas as pd
import mlflow
import mlflow.sklearn

from src.config import load_config
from src.features import bmi_category
from src.db import Base, Prediction, get_engine, get_session_factory

# startup / shutdown

engine = None
SessionLocal = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create DB tables on startup. Graceful if DB is unavailable (e.g. tests)."""
    global engine, SessionLocal
    try:
        engine = get_engine()
        SessionLocal = get_session_factory(engine)
        Base.metadata.create_all(engine)
        print("Database connected — predictions table ready")
    except Exception as e:
        print(f"Warning: Database unavailable — prediction logging disabled. ({e})")
        engine = None
        SessionLocal = None
    yield
    # shutdown: nothing to clean up (engine pool closes automatically)


# app + model

cfg = load_config()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"]))
model = mlflow.sklearn.load_model(f"models:/{cfg['mlflow']['registered_model_name']}@production")

app = FastAPI(lifespan=lifespan)

FEATURE_COLUMNS = [
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
    'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
    'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
    'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education',
    'Income', 'BMI_cat'
]


# request schema

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

    # engineered feature — computed from BMI, not sent by the caller
    @computed_field
    def BMI_cat(self) -> int:
        return bmi_category(self.BMI)


# endpoints

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/about")
def about():
    return {"message": "This API predicts the risk of diabetes based on user input features."}

@app.post("/predict")
def predict(user_input: UserInput):
    df = pd.DataFrame([user_input.model_dump()])[FEATURE_COLUMNS]
    probability = model.predict_proba(df)[0][1]
    risk_percent = round(probability * 100, 2)
    label = "High Risk" if probability >= 0.5 else "Low Risk"

    # log prediction to database (non-blocking — DB failure does not affect response)
    if SessionLocal is not None:
        try:
            session = SessionLocal()
            session.add(Prediction(
                **user_input.model_dump(),
                risk_score=risk_percent,
                risk_label=label,
            ))
            session.commit()
            session.close()
        except Exception as e:
            print(f"DB logging failed: {e}")

    return JSONResponse(status_code=200, content={
        "diabetes_risk_score": risk_percent,
        "risk_label": label,
    })
