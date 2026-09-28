"""Generate a deterministic, synthetic customer churn dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "contract_type",
    "payment_method",
    "internet_service",
    "support_tickets",
    "late_payments",
    "usage_gb",
    "senior_citizen",
    "dependents",
    "paperless_billing",
]
TARGET_COLUMN = "churn"


def generate_dataset(rows: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """Create customer features and a probabilistic churn label."""
    if rows < 2:
        raise ValueError("rows must be at least 2")

    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 73, size=rows)
    monthly_charges = np.clip(rng.normal(70, 22, size=rows), 18, 130).round(2)
    total_charges = np.maximum(
        0, tenure * monthly_charges + rng.normal(0, 90, size=rows)
    ).round(2)
    contract = rng.choice(
        ["month_to_month", "one_year", "two_year"],
        size=rows,
        p=[0.55, 0.25, 0.20],
    )
    payment = rng.choice(
        ["electronic_check", "credit_card", "bank_transfer", "mailed_check"],
        size=rows,
        p=[0.36, 0.25, 0.24, 0.15],
    )
    internet = rng.choice(
        ["fiber", "dsl", "none"], size=rows, p=[0.48, 0.38, 0.14]
    )
    support_tickets = rng.poisson(1.2, size=rows).clip(0, 8)
    late_payments = rng.poisson(0.45, size=rows).clip(0, 5)
    usage_gb = np.clip(rng.gamma(2.2, 24, size=rows), 0, 250).round(1)
    senior = rng.binomial(1, 0.18, size=rows).astype(bool)
    dependents = rng.binomial(1, 0.30, size=rows).astype(bool)
    paperless = rng.binomial(1, 0.62, size=rows).astype(bool)

    # A transparent synthetic signal: short tenure, flexible contracts, service
    # issues, and payment friction increase the probability of churn.
    log_odds = (
        -1.35
        - 0.035 * (tenure - 24)
        + 1.0 * (contract == "month_to_month")
        + 0.25 * (contract == "one_year")
        + 0.35 * (payment == "electronic_check")
        + 0.20 * (internet == "fiber")
        + 0.32 * support_tickets
        + 0.48 * late_payments
        + 0.25 * senior
        - 0.35 * dependents
        + 0.20 * paperless
    )
    probability = 1 / (1 + np.exp(-log_odds))
    churn = rng.binomial(1, probability, size=rows)

    return pd.DataFrame(
        {
            "tenure_months": tenure,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "contract_type": contract,
            "payment_method": payment,
            "internet_service": internet,
            "support_tickets": support_tickets,
            "late_payments": late_payments,
            "usage_gb": usage_gb,
            "senior_citizen": senior,
            "dependents": dependents,
            "paperless_billing": paperless,
            TARGET_COLUMN: churn,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/churn.csv"))
    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = generate_dataset(rows=args.rows, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output, index=False)
    print(f"Wrote {len(dataset):,} rows to {args.output}")
    print(f"Churn rate: {dataset[TARGET_COLUMN].mean():.1%}")


if __name__ == "__main__":
    main()
