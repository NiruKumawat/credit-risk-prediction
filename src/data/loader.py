"""Data loader and ingestion pipeline for the Credit Risk Prediction System."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

TARGET_COL = "default_payment_next_month"
RAW_TARGET_COL = "default payment next month"

# Standardized feature definitions
DEMOGRAPHIC_FEATURES = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
PAYMENT_STATUS_FEATURES = [f"PAY_{i}" for i in range(1, 7)]
BILL_AMOUNT_FEATURES = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMOUNT_FEATURES = [f"PAY_AMT{i}" for i in range(1, 7)]

ALL_RAW_FEATURES = (
    DEMOGRAPHIC_FEATURES
    + PAYMENT_STATUS_FEATURES
    + BILL_AMOUNT_FEATURES
    + PAY_AMOUNT_FEATURES
)


def get_feature_column_names() -> List[str]:
    """Return the ordered list of standard input feature names."""
    return list(ALL_RAW_FEATURES)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names: strip whitespace, rename PAY_0 -> PAY_1, and format target."""
    df = df.copy()
    # Strip any leading/trailing spaces from column headers
    df.columns = [str(c).strip() for c in df.columns]

    # Standardize PAY_0 -> PAY_1 for consistent 1-to-6 indexing
    if "PAY_0" in df.columns and "PAY_1" not in df.columns:
        df = df.rename(columns={"PAY_0": "PAY_1"})

    # Standardize target column
    if RAW_TARGET_COL in df.columns:
        df = df.rename(columns={RAW_TARGET_COL: TARGET_COL})

    # Drop client ID if present (should not be used as a predictive feature)
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    return df


def clean_categorical_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Consolidate undocumented categorical codes according to UCI dataset documentation.

    EDUCATION: 1=graduate school, 2=university, 3=high school, 4=others.
               Codes 0, 5, 6 are undocumented and consolidated into 4 (others).
    MARRIAGE: 1=married, 2=single, 3=others.
              Code 0 is undocumented and consolidated into 3 (others).
    """
    df = df.copy()
    if "EDUCATION" in df.columns:
        df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    if "MARRIAGE" in df.columns:
        df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    return df


def load_credit_card_data(
    file_path: Optional[str | Path] = None,
    drop_duplicates: bool = True,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load, standardize, clean, and deduplicate credit card client dataset.

    Args:
        file_path: Path to dataset file (.xls, .csv). If None, searches standard locations.
        drop_duplicates: Whether to drop duplicate rows across feature columns.
        use_cache: Whether to load from/save to cleaned CSV cache for fast performance.

    Returns:
        Cleaned Pandas DataFrame ready for splitting and modeling.
    """
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data"

    cached_csv = data_dir / "credit_card_clients_cleaned.csv"

    if file_path is None:
        if use_cache and cached_csv.exists():
            logger.info(f"Loading cleaned data from cache: {cached_csv}")
            return pd.read_csv(cached_csv)

        raw_xls = data_dir / "default of credit card clients.xls"
        raw_csv = data_dir / "credit_card_clients.csv"
        synthetic_csv = data_dir / "synthetic_sample_for_testing.csv"

        if raw_xls.exists():
            file_path = raw_xls
        elif raw_csv.exists():
            file_path = raw_csv
        elif synthetic_csv.exists():
            file_path = synthetic_csv
            logger.warning(
                f"Using synthetic test sample from {synthetic_csv}. (For testing only!)"
            )
        else:
            raise FileNotFoundError(
                f"No credit card dataset found in {data_dir}. "
                "Run `python -m src.data.download --download` first."
            )

    file_path = Path(file_path)
    logger.info(f"Loading dataset from: {file_path}")

    if file_path.suffix.lower() in [".xls", ".xlsx"]:
        # The UCI Excel file contains metadata in row 0, header is on row 1
        df = pd.read_excel(file_path, header=1)
    else:
        df = pd.read_csv(file_path)

    # Standardize column naming
    df = standardize_columns(df)

    # Clean undocumented categorical codes
    df = clean_categorical_categories(df)

    # Check and handle duplicates
    initial_len = len(df)
    if drop_duplicates:
        feature_cols = [c for c in df.columns if c != TARGET_COL]
        duplicates_count = df.duplicated(subset=feature_cols).sum()
        if duplicates_count > 0:
            logger.info(f"Dropping {duplicates_count} duplicate records...")
            df = df.drop_duplicates(subset=feature_cols).reset_index(drop=True)
            logger.info(f"Dataset reduced from {initial_len} to {len(df)} records.")

    # Save to cache for ultra-fast subsequent loads
    if use_cache and not cached_csv.exists() and file_path != synthetic_csv:
        try:
            df.to_csv(cached_csv, index=False)
            logger.info(f"Cached cleaned dataset to {cached_csv}")
        except Exception as e:
            logger.warning(f"Could not cache cleaned dataset: {e}")

    return df
