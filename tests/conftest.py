"""Pytest fixtures and test configurations for Credit Risk Prediction suite."""

import pytest
import pandas as pd
from fastapi.testclient import TestClient

from api.main import app
from src.data.loader import ALL_RAW_FEATURES


@pytest.fixture(scope="session")
def api_client() -> TestClient:
    """Provide a FastAPI TestClient instance for API tests."""
    return TestClient(app)


@pytest.fixture
def valid_applicant_dict() -> dict:
    """Provide a valid credit applicant payload."""
    return {
        "LIMIT_BAL": 120000.0,
        "SEX": 1,
        "EDUCATION": 2,
        "MARRIAGE": 2,
        "AGE": 35,
        "PAY_1": 0,
        "PAY_2": 0,
        "PAY_3": 0,
        "PAY_4": 0,
        "PAY_5": 0,
        "PAY_6": 0,
        "BILL_AMT1": 25000.0,
        "BILL_AMT2": 24000.0,
        "BILL_AMT3": 23000.0,
        "BILL_AMT4": 22000.0,
        "BILL_AMT5": 21000.0,
        "BILL_AMT6": 20000.0,
        "PAY_AMT1": 2500.0,
        "PAY_AMT2": 2500.0,
        "PAY_AMT3": 2500.0,
        "PAY_AMT4": 2500.0,
        "PAY_AMT5": 2500.0,
        "PAY_AMT6": 2500.0,
    }


@pytest.fixture
def sample_applicants_df(valid_applicant_dict) -> pd.DataFrame:
    """Provide a small DataFrame of applicant records for pipeline testing."""
    records = []
    for i in range(5):
        rec = valid_applicant_dict.copy()
        rec["LIMIT_BAL"] += i * 20000
        rec["AGE"] += i
        rec["PAY_1"] = i - 1  # -1, 0, 1, 2, 3
        records.append(rec)
    return pd.DataFrame(records)
