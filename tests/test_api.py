"""Integration tests for FastAPI endpoints, request schema validation, and error handling."""

import pytest
from fastapi.testclient import TestClient


def test_health_endpoint(api_client: TestClient):
    """Test GET /health returns 200 and healthy status."""
    response = api_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "model_name" in data
    assert "server_time" in data


def test_model_info_endpoint(api_client: TestClient):
    """Test GET /model-info returns 200 and valid metadata."""
    response = api_client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "champion_model_name" in data
    assert "selected_decision_threshold" in data
    assert "test_metrics" in data
    assert "disclaimer" in data


def test_predict_endpoint_valid(api_client: TestClient, valid_applicant_dict: dict):
    """Test POST /predict with a valid applicant payload."""
    payload = {"applicant": valid_applicant_dict}
    response = api_client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "default_probability" in data
    assert 0.0 <= data["default_probability"] <= 1.0
    assert data["predicted_class"] in [0, 1]
    assert data["risk_category"] in ["Low", "Medium", "High"]
    assert "threshold_applied" in data
    assert "recommendation" in data
    assert "key_risk_drivers" in data
    assert "disclaimer" in data


def test_predict_endpoint_with_custom_threshold(api_client: TestClient, valid_applicant_dict: dict):
    """Test POST /predict supports custom threshold override."""
    payload = {
        "applicant": valid_applicant_dict,
        "custom_threshold": 0.25,
    }
    response = api_client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["threshold_applied"] == 0.25


def test_predict_batch_endpoint(api_client: TestClient, valid_applicant_dict: dict):
    """Test POST /predict-batch screens multiple applicants."""
    app1 = valid_applicant_dict.copy()
    app2 = valid_applicant_dict.copy()
    app2["PAY_1"] = 2  # Delinquent borrower

    payload = {"applicants": [app1, app2]}
    response = api_client.post("/predict-batch", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["total_records"] == 2
    assert "default_count" in data
    assert len(data["predictions"]) == 2


def test_predict_invalid_schema_negative_limit(api_client: TestClient, valid_applicant_dict: dict):
    """Test POST /predict rejects invalid negative credit limit with HTTP 422."""
    bad_payload = {"applicant": valid_applicant_dict.copy()}
    bad_payload["applicant"]["LIMIT_BAL"] = -500.0  # Invalid gt=0
    response = api_client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_invalid_schema_underage(api_client: TestClient, valid_applicant_dict: dict):
    """Test POST /predict rejects applicant under 18 with HTTP 422."""
    bad_payload = {"applicant": valid_applicant_dict.copy()}
    bad_payload["applicant"]["AGE"] = 15  # Invalid ge=18
    response = api_client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_invalid_schema_missing_fields(api_client: TestClient):
    """Test POST /predict rejects missing required fields with HTTP 422."""
    bad_payload = {"applicant": {"LIMIT_BAL": 50000.0}}
    response = api_client.post("/predict", json=bad_payload)
    assert response.status_code == 422
