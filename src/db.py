import os
from datetime import datetime

from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

# connection

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://diabetes:diabetes@localhost:5432/diabetes_db"
)

def get_engine():
    """Return a SQLAlchemy engine. Call once at app startup."""
    return create_engine(DATABASE_URL)


# ORM models

Base = declarative_base()

class Prediction(Base):
    """One row per /predict call — input features + model output + timestamp."""
    __tablename__ = "predictions"

    id        = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.now, nullable=False)

    # binary input features (0/1)
    HighBP               = Column(Integer, nullable=False)
    HighChol             = Column(Integer, nullable=False)
    CholCheck            = Column(Integer, nullable=False)
    Smoker               = Column(Integer, nullable=False)
    Stroke               = Column(Integer, nullable=False)
    HeartDiseaseorAttack = Column(Integer, nullable=False)
    PhysActivity         = Column(Integer, nullable=False)
    Fruits               = Column(Integer, nullable=False)
    Veggies              = Column(Integer, nullable=False)
    HvyAlcoholConsump    = Column(Integer, nullable=False)
    AnyHealthcare        = Column(Integer, nullable=False)
    NoDocbcCost          = Column(Integer, nullable=False)
    DiffWalk             = Column(Integer, nullable=False)
    Sex                  = Column(Integer, nullable=False)

    # numeric input features
    BMI       = Column(Float,   nullable=False)
    GenHlth   = Column(Integer, nullable=False)
    MentHlth  = Column(Integer, nullable=False)
    PhysHlth  = Column(Integer, nullable=False)
    Age       = Column(Integer, nullable=False)
    Education = Column(Integer, nullable=False)
    Income    = Column(Integer, nullable=False)

    # engineered feature (stored so queries don't need to recompute it)
    BMI_cat   = Column(Integer, nullable=False)

    # model output
    risk_score = Column(Float,  nullable=False)
    risk_label = Column(String, nullable=False)


# session factory

def get_session_factory(engine):
    """Return a session class bound to the given engine."""
    return sessionmaker(bind=engine)
