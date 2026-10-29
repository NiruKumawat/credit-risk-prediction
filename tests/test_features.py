"""Unit tests for feature engineering, scikit-learn pipeline, and leakage checks."""

import pytest
import numpy as np
import pandas as pd
from src.features.engineering import CreditRiskFeatureEngineer
from src.features.pipeline import (
    build_preprocessing_pipeline,
    get_preprocessed_feature_names,
    CATEGORICAL_FEATURES,
    ALL_NUMERICAL_FEATURES,
)


def test_credit_risk_feature_engineer(sample_applicants_df):
    """Test domain feature creation and correctness."""
    engineer = CreditRiskFeatureEngineer()
    engineered_df = engineer.fit_transform(sample_applicants_df)

    # Check generated feature columns exist
    expected_new = [
        "credit_utilization_ratio",
        "avg_bill_amt",
        "avg_pay_amt",
        "bill_to_limit_avg",
        "payment_to_bill_ratio",
        "max_delay_months",
        "months_delayed_count",
        "pay_trend",
    ]
    for feat in expected_new:
        assert feat in engineered_df.columns, f"Missing engineered feature: {feat}"

    # Verify credit utilization formula
    expected_util = sample_applicants_df["BILL_AMT1"] / sample_applicants_df["LIMIT_BAL"]
    np.testing.assert_allclose(engineered_df["credit_utilization_ratio"], expected_util, rtol=1e-5)

    # Verify max_delay_months calculation
    status_cols = [f"PAY_{i}" for i in range(1, 7)]
    expected_max = sample_applicants_df[status_cols].max(axis=1)
    np.testing.assert_array_equal(engineered_df["max_delay_months"], expected_max)


def test_preprocessing_pipeline_transform(sample_applicants_df):
    """Test that preprocessing pipeline fits on training and transforms identically."""
    pipeline = build_preprocessing_pipeline(scale_numeric=True)

    # Fit and transform
    X_trans = pipeline.fit_transform(sample_applicants_df)
    feature_names = get_preprocessed_feature_names(pipeline)

    assert X_trans.shape[0] == len(sample_applicants_df)
    assert X_trans.shape[1] == len(feature_names)
    assert not np.isnan(X_trans).any()

    # Test transform on unseen single record without refitting
    single_trans = pipeline.transform(sample_applicants_df.iloc[[0]])
    assert single_trans.shape == (1, len(feature_names))


def test_no_target_in_features(sample_applicants_df):
    """Data leakage check: Verify target is never required or present in feature pipeline."""
    pipeline = build_preprocessing_pipeline()
    assert "default_payment_next_month" not in sample_applicants_df.columns
    # Should fit and transform cleanly without any target column
    X_trans = pipeline.fit_transform(sample_applicants_df)
    assert X_trans is not None
