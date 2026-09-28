"""FastAPI inference service for the trained churn model."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from churn_prediction.data import FEATURE_COLUMNS


DEFAULT_MODEL_PATH = Path("artifacts/churn_model.joblib")


class CustomerFeatures(BaseModel):
    """Validated feature payload for one customer."""

    model_config = ConfigDict(extra="forbid")

    tenure_months: int = Field(ge=0, le=120)
    monthly_charges: float = Field(ge=0, le=1000)
    total_charges: float = Field(ge=0, le=100_000)
    contract_type: Literal["month_to_month", "one_year", "two_year"]
    payment_method: Literal[
        "electronic_check", "credit_card", "bank_transfer", "mailed_check"
    ]
    internet_service: Literal["fiber", "dsl", "none"]
    support_tickets: int = Field(ge=0, le=100)
    late_payments: int = Field(ge=0, le=100)
    usage_gb: float = Field(ge=0, le=10_000)
    senior_citizen: bool
    dependents: bool
    paperless_billing: bool


class ChurnPrediction(BaseModel):
    churn_probability: float
    predicted_churn: bool


app = FastAPI(
    title="Customer Churn Prediction API",
    version="0.1.0",
    description="Predict customer churn probability from subscription and service features.",
)


@lru_cache(maxsize=1)
def load_model():
    model_path = Path(os.environ.get("CHURN_MODEL_PATH", DEFAULT_MODEL_PATH))
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model artifact not found at {model_path}. Run "
            "`python -m churn_prediction.train` after generating the dataset."
        )
    return joblib.load(model_path)


@app.get("/health")
def health() -> dict[str, str]:
    model_path = Path(os.environ.get("CHURN_MODEL_PATH", DEFAULT_MODEL_PATH))
    if not model_path.is_file():
        return {"status": "model_not_ready", "model_path": str(model_path)}
    return {"status": "ready"}


@app.post("/predict", response_model=ChurnPrediction)
def predict(customer: CustomerFeatures) -> ChurnPrediction:
    try:
        model = load_model()
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    features = pd.DataFrame(
        [customer.model_dump(include=set(FEATURE_COLUMNS))],
        columns=FEATURE_COLUMNS,
    )
    probability = float(model.predict_proba(features)[0, 1])
    return ChurnPrediction(
        churn_probability=probability,
        predicted_churn=probability >= 0.5,
    )
