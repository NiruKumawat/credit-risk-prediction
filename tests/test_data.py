"""Unit tests for data loading, schema validation, and cleaning."""

import pytest
import pandas as pd
from src.data.loader import (
    load_credit_card_data,
    standardize_columns,
    clean_categorical_categories,
    ALL_RAW_FEATURES,
    TARGET_COL,
)
from src.data.validator import (
    validate_credit_data,
    validate_applicant_record,
    DataValidationError,
)


def test_standardize_columns():
    """Test column whitespace stripping, PAY_0 renaming, and ID dropping."""
    raw_df = pd.DataFrame({
        " ID ": [1, 2],
        " PAY_0 ": [0, 1],
        "default payment next month": [0, 1],
        "LIMIT_BAL": [50000, 100000],
    })
    cleaned = standardize_columns(raw_df)
    assert "ID" not in cleaned.columns
    assert "PAY_1" in cleaned.columns
    assert "PAY_0" not in cleaned.columns
    assert TARGET_COL in cleaned.columns


def test_clean_categorical_categories():
    """Test undocumented category consolidation for EDUCATION and MARRIAGE."""
    df = pd.DataFrame({
        "EDUCATION": [0, 1, 2, 3, 4, 5, 6],
        "MARRIAGE": [0, 1, 2, 3, 0, 1, 2],
    })
    cleaned = clean_categorical_categories(df)
    assert set(cleaned["EDUCATION"].unique()).issubset({1, 2, 3, 4})
    assert set(cleaned["MARRIAGE"].unique()).issubset({1, 2, 3})


def test_data_loader_structure():
    """Test full data loading pipeline returns expected shape and columns."""
    df = load_credit_card_data()
    assert len(df) > 1000
    for col in ALL_RAW_FEATURES:
        assert col in df.columns, f"Missing feature: {col}"
    assert TARGET_COL in df.columns
    assert df[TARGET_COL].isin([0, 1]).all()


def test_validate_credit_data_report():
    """Test schema and statistical validation function."""
    df = load_credit_card_data()
    report = validate_credit_data(df, is_training=True)
    assert report["validation_passed"] is True
    assert report["missing_values_total"] == 0
    assert 0.15 < report["positive_class_ratio"] < 0.30


def test_validate_applicant_record_valid(valid_applicant_dict):
    """Test single applicant validation on valid payload."""
    is_valid, errors = validate_applicant_record(valid_applicant_dict)
    assert is_valid is True
    assert len(errors) == 0


def test_validate_applicant_record_invalid():
    """Test single applicant validation catching out-of-range domain inputs."""
    invalid = {
        "LIMIT_BAL": -1000,  # Invalid limit
        "SEX": 5,           # Invalid sex
        "AGE": 12,          # Invalid age < 18
        "PAY_1": 15,        # Invalid pay status > 8
    }
    is_valid, errors = validate_applicant_record(invalid)
    assert is_valid is False
    assert len(errors) > 0
