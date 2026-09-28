# Customer churn prediction engine

A small, reproducible Python project that generates synthetic customer data,
trains a baseline XGBoost classifier, and serves churn predictions over HTTP.
The generated data is for development and demonstration only; it is not
representative of real customers.

## Setup

Requires Python 3.10 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Generate data and train

```powershell
python -m churn_prediction.data --output data/churn.csv --rows 10000 --seed 42
python -m churn_prediction.train --input data/churn.csv --model-output artifacts/churn_model.joblib
```

The training command reports holdout accuracy, ROC AUC, and a classification
report. It saves the complete preprocessing/model pipeline so inference uses
the same categorical encoding as training.

## Run the inference API

```powershell
uvicorn churn_prediction.api:app --app-dir src --reload
```

Open <http://127.0.0.1:8000/docs> for the interactive API documentation.
`GET /health` reports whether a trained model artifact is available.
`POST /predict` accepts a single customer and returns its churn probability
and predicted class. Set `CHURN_MODEL_PATH` to use a model saved elsewhere.

Example request:

```json
{
  "tenure_months": 4,
  "monthly_charges": 89.5,
  "total_charges": 358.0,
  "contract_type": "month_to_month",
  "payment_method": "electronic_check",
  "internet_service": "fiber",
  "support_tickets": 3,
  "late_payments": 2,
  "usage_gb": 75.0,
  "senior_citizen": false,
  "dependents": false,
  "paperless_billing": true
}
```

## Tests

```powershell
python -m pytest
```
