"""Data validation and schema integrity checks for Credit Risk Prediction."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.data.loader import ALL_RAW_FEATURES, TARGET_COL

logger = logging.getLogger(__name__)

# Business and domain validation constraints
VALID_SEX_VALUES = {1, 2}  # 1: Male, 2: Female
VALID_EDUCATION_VALUES = {1, 2, 3, 4}  # 1: Grad School, 2: University, 3: High School, 4: Others
VALID_MARRIAGE_VALUES = {1, 2, 3}  # 1: Married, 2: Single, 3: Others
VALID_PAY_STATUS_RANGE = (-2, 8)  # Repayment status range
MIN_AGE = 18
MAX_AGE = 100
MIN_LIMIT_BAL = 0


class DataValidationError(ValueError):
    """Custom exception raised when credit card data fails schema or domain validation."""
    pass


def validate_credit_data(df: pd.DataFrame, is_training: bool = True) -> Dict[str, Any]:
    """Perform comprehensive schema and data quality validation.

    Args:
        df: Input DataFrame to validate.
        is_training: If True, requires target column to be present.

    Returns:
        Dictionary with validation report summary.

    Raises:
        DataValidationError: If critical validation checks fail.
    """
    report: Dict[str, Any] = {
        "num_records": len(df),
        "num_columns": len(df.columns),
        "missing_values_total": int(df.isnull().sum().sum()),
        "missing_values_by_col": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "validation_passed": True,
        "warnings": [],
    }

    # 1. Check required feature columns
    missing_cols = [col for col in ALL_RAW_FEATURES if col not in df.columns]
    if missing_cols:
        raise DataValidationError(f"Missing required feature columns: {missing_cols}")

    # 2. Check target column if training
    if is_training:
        if TARGET_COL not in df.columns:
            raise DataValidationError(
                f"Target column '{TARGET_COL}' is missing from training dataset."
            )
        target_counts = df[TARGET_COL].value_counts(dropna=False).to_dict()
        report["target_distribution"] = target_counts
        unique_targets = set(df[TARGET_COL].dropna().unique())
        if not unique_targets.issubset({0, 1}):
            raise DataValidationError(
                f"Target column must contain only binary values {{0, 1}}, found: {unique_targets}"
            )
        report["positive_class_ratio"] = float(df[TARGET_COL].mean())

    # 3. Check for null values
    if report["missing_values_total"] > 0:
        cols_with_nulls = {k: v for k, v in report["missing_values_by_col"].items() if v > 0}
        report["warnings"].append(f"Columns with null values detected: {cols_with_nulls}")

    # 4. Check domain constraints
    invalid_age = df[(df["AGE"] < MIN_AGE) | (df["AGE"] > MAX_AGE)]
    if len(invalid_age) > 0:
        report["warnings"].append(
            f"Found {len(invalid_age)} records with age outside [{MIN_AGE}, {MAX_AGE}]."
        )

    invalid_limit = df[df["LIMIT_BAL"] < MIN_LIMIT_BAL]
    if len(invalid_limit) > 0:
        report["warnings"].append(
            f"Found {len(invalid_limit)} records with negative credit limit."
        )

    logger.info(
        f"Validation completed successfully: {len(df)} records. "
        f"Missing values: {report['missing_values_total']}. "
        f"Warnings: {len(report['warnings'])}"
    )
    return report


def validate_applicant_record(record: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate a single applicant record (e.g. from API or Dashboard payload).

    Args:
        record: Dictionary with feature name -> value mappings.

    Returns:
        Tuple of (is_valid: bool, error_messages: list of strings)
    """
    errors: List[str] = []

    # Check presence of required features
    for col in ALL_RAW_FEATURES:
        if col not in record:
            errors.append(f"Missing required field: '{col}'")

    if errors:
        return False, errors

    # Numerical type checks and ranges
    try:
        limit_bal = float(record["LIMIT_BAL"])
        if limit_bal <= 0:
            errors.append("LIMIT_BAL must be greater than 0.")
    except (ValueError, TypeError):
        errors.append("LIMIT_BAL must be a valid number.")

    try:
        age = int(record["AGE"])
        if not (MIN_AGE <= age <= MAX_AGE):
            errors.append(f"AGE must be an integer between {MIN_AGE} and {MAX_AGE}.")
    except (ValueError, TypeError):
        errors.append("AGE must be a valid integer.")

    try:
        sex = int(record["SEX"])
        if sex not in VALID_SEX_VALUES:
            errors.append(f"SEX must be 1 (Male) or 2 (Female), received: {sex}.")
    except (ValueError, TypeError):
        errors.append("SEX must be an integer (1 or 2).")

    try:
        education = int(record["EDUCATION"])
        if education not in VALID_EDUCATION_VALUES:
            errors.append(
                f"EDUCATION must be in {{1, 2, 3, 4}} (Grad School, University, High School, Others), received: {education}."
            )
    except (ValueError, TypeError):
        errors.append("EDUCATION must be an integer in {1, 2, 3, 4}.")

    try:
        marriage = int(record["MARRIAGE"])
        if marriage not in VALID_MARRIAGE_VALUES:
            errors.append(
                f"MARRIAGE must be in {{1, 2, 3}} (Married, Single, Others), received: {marriage}."
            )
    except (ValueError, TypeError):
        errors.append("MARRIAGE must be an integer in {1, 2, 3}.")

    # Validate payment status ranges
    for i in range(1, 7):
        col = f"PAY_{i}"
        try:
            status = int(record[col])
            if status < VALID_PAY_STATUS_RANGE[0] or status > VALID_PAY_STATUS_RANGE[1]:
                errors.append(
                    f"{col} must be between {VALID_PAY_STATUS_RANGE[0]} and {VALID_PAY_STATUS_RANGE[1]}, received: {status}."
                )
        except (ValueError, TypeError):
            errors.append(f"{col} must be a valid integer.")

    # Validate bill and payment amounts
    for i in range(1, 7):
        for prefix in ["BILL_AMT", "PAY_AMT"]:
            col = f"{prefix}{i}"
            try:
                float(record[col])
            except (ValueError, TypeError):
                errors.append(f"{col} must be a valid number.")

    return len(errors) == 0, errors
