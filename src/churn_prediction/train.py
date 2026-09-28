"""Train and evaluate the baseline churn classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from churn_prediction.data import FEATURE_COLUMNS, TARGET_COLUMN


CATEGORICAL_COLUMNS = ["contract_type", "payment_method", "internet_service"]
NUMERIC_COLUMNS = [column for column in FEATURE_COLUMNS if column not in CATEGORICAL_COLUMNS]


def build_pipeline(seed: int = 42) -> Pipeline:
    """Build preprocessing and XGBoost as one serializable pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_COLUMNS,
            ),
            (
                "numeric",
                SimpleImputer(strategy="median"),
                NUMERIC_COLUMNS,
            ),
        ]
    )
    classifier = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def train_model(
    data_path: Path, model_path: Path, seed: int = 42
) -> dict[str, float]:
    """Train on a stratified split, print holdout metrics, and save the pipeline."""
    data = pd.read_csv(data_path)
    required_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing_columns = sorted(required_columns - set(data.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing_columns)}")
    if data[TARGET_COLUMN].nunique() != 2:
        raise ValueError("The churn target must contain both classes (0 and 1)")

    features = data[FEATURE_COLUMNS]
    target = data[TARGET_COLUMN].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=seed,
        stratify=target,
    )
    model = build_pipeline(seed=seed)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"ROC AUC:  {metrics['roc_auc']:.3f}")
    print(classification_report(y_test, predictions, zero_division=0))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Saved model pipeline to {model_path}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/churn.csv"))
    parser.add_argument(
        "--model-output", type=Path, default=Path("artifacts/churn_model.joblib")
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_model(args.input, args.model_output, seed=args.seed)


if __name__ == "__main__":
    main()
