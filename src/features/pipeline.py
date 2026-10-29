"""Scikit-learn preprocessing pipelines for Credit Risk Prediction.

Ensures strict prevention of data leakage:
- Fit transformations on training data only.
- Transform validation, test, and production payloads identically.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.engineering import CreditRiskFeatureEngineer

logger = logging.getLogger(__name__)

# Feature group classifications
CATEGORICAL_FEATURES = ["SEX", "EDUCATION", "MARRIAGE"]

BASE_NUMERICAL_FEATURES = (
    ["LIMIT_BAL", "AGE"]
    + [f"PAY_{i}" for i in range(1, 7)]
    + [f"BILL_AMT{i}" for i in range(1, 7)]
    + [f"PAY_AMT{i}" for i in range(1, 7)]
)

ENGINEERED_NUMERICAL_FEATURES = [
    "credit_utilization_ratio",
    "avg_bill_amt",
    "avg_pay_amt",
    "bill_to_limit_avg",
    "payment_to_bill_ratio",
    "max_delay_months",
    "months_delayed_count",
    "pay_trend",
]

ALL_NUMERICAL_FEATURES = BASE_NUMERICAL_FEATURES + ENGINEERED_NUMERICAL_FEATURES


def build_column_transformer(scale_numeric: bool = True) -> ColumnTransformer:
    """Build ColumnTransformer for one-hot encoding categories and scaling numerics."""
    numeric_transformer = (
        StandardScaler() if scale_numeric else "passthrough"
    )

    categorical_transformer = OneHotEncoder(
        drop="first",
        handle_unknown="ignore",
        sparse_output=False,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, ALL_NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def build_preprocessing_pipeline(scale_numeric: bool = True) -> Pipeline:
    """Create complete preprocessing pipeline: Feature Engineering -> Encoding & Scaling."""
    pipeline = Pipeline(
        steps=[
            ("feature_engineer", CreditRiskFeatureEngineer()),
            ("column_preprocessor", build_column_transformer(scale_numeric=scale_numeric)),
        ]
    )
    return pipeline


def get_preprocessed_feature_names(fitted_pipeline: Pipeline) -> List[str]:
    """Extract human-readable feature names from the fitted preprocessing pipeline."""
    try:
        column_preprocessor: ColumnTransformer = fitted_pipeline.named_steps["column_preprocessor"]
        names = column_preprocessor.get_feature_names_out()
        return list(names)
    except Exception as exc:
        logger.warning(f"Could not extract feature names dynamically: {exc}")
        return ALL_NUMERICAL_FEATURES + [f"cat_{c}" for c in CATEGORICAL_FEATURES]
