"""Pydantic schemas and data validation models for Credit Risk API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator


class ApplicantFeatures(BaseModel):
    """Credit card loan applicant features for risk evaluation."""

    LIMIT_BAL: float = Field(..., gt=0, description="Amount of given credit in NT dollars (e.g. 50000)")
    SEX: int = Field(..., ge=1, le=2, description="Gender (1 = male, 2 = female)")
    EDUCATION: int = Field(..., ge=1, le=4, description="Education level (1=graduate school, 2=university, 3=high school, 4=others)")
    MARRIAGE: int = Field(..., ge=1, le=3, description="Marital status (1=married, 2=single, 3=others)")
    AGE: int = Field(..., ge=18, le=100, description="Age in years (18-100)")

    # Repayment status for past 6 months (April to Sept 2005)
    PAY_1: int = Field(..., ge=-2, le=8, description="Repayment status in Sept 2005 (-2=no consumption, -1=paid in full, 0=revolving, 1-8=months delay)")
    PAY_2: int = Field(0, ge=-2, le=8, description="Repayment status in Aug 2005")
    PAY_3: int = Field(0, ge=-2, le=8, description="Repayment status in July 2005")
    PAY_4: int = Field(0, ge=-2, le=8, description="Repayment status in June 2005")
    PAY_5: int = Field(0, ge=-2, le=8, description="Repayment status in May 2005")
    PAY_6: int = Field(0, ge=-2, le=8, description="Repayment status in April 2005")

    # Bill statement amounts
    BILL_AMT1: float = Field(0.0, description="Bill statement amount in Sept 2005")
    BILL_AMT2: float = Field(0.0, description="Bill statement amount in Aug 2005")
    BILL_AMT3: float = Field(0.0, description="Bill statement amount in July 2005")
    BILL_AMT4: float = Field(0.0, description="Bill statement amount in June 2005")
    BILL_AMT5: float = Field(0.0, description="Bill statement amount in May 2005")
    BILL_AMT6: float = Field(0.0, description="Bill statement amount in April 2005")

    # Previous payment amounts
    PAY_AMT1: float = Field(0.0, ge=0.0, description="Previous payment amount in Sept 2005")
    PAY_AMT2: float = Field(0.0, ge=0.0, description="Previous payment amount in Aug 2005")
    PAY_AMT3: float = Field(0.0, ge=0.0, description="Previous payment amount in July 2005")
    PAY_AMT4: float = Field(0.0, ge=0.0, description="Previous payment amount in June 2005")
    PAY_AMT5: float = Field(0.0, ge=0.0, description="Previous payment amount in May 2005")
    PAY_AMT6: float = Field(0.0, ge=0.0, description="Previous payment amount in April 2005")

    @model_validator(mode="before")
    @classmethod
    def handle_pay_0_alias(cls, data: Any) -> Any:
        """Allow clients to pass either PAY_0 or PAY_1 interchangeably."""
        if isinstance(data, dict):
            if "PAY_0" in data and "PAY_1" not in data:
                data["PAY_1"] = data["PAY_0"]
        return data

    model_config = {
        "json_schema_extra": {
            "example": {
                "LIMIT_BAL": 100000.0,
                "SEX": 1,
                "EDUCATION": 2,
                "MARRIAGE": 2,
                "AGE": 32,
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
                "PAY_AMT1": 2000.0,
                "PAY_AMT2": 2000.0,
                "PAY_AMT3": 2000.0,
                "PAY_AMT4": 2000.0,
                "PAY_AMT5": 2000.0,
                "PAY_AMT6": 2000.0,
            }
        }
    }


class PredictionRequest(BaseModel):
    """Single prediction request with optional custom decision threshold."""
    applicant: ApplicantFeatures
    custom_threshold: Optional[float] = Field(
        None,
        ge=0.01,
        le=0.99,
        description="Optional custom decision threshold override. Defaults to trained cost-optimal threshold.",
    )


class BatchPredictionRequest(BaseModel):
    """Batch prediction request for screening applicant cohorts."""
    applicants: List[ApplicantFeatures] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of applicant records to screen.",
    )
    custom_threshold: Optional[float] = Field(
        None,
        ge=0.01,
        le=0.99,
        description="Optional custom decision threshold override.",
    )


class TopRiskFactor(BaseModel):
    feature: str
    impact: float
    description: Optional[str] = None


class SinglePredictionResponse(BaseModel):
    """Structured response for risk prediction."""
    default_probability: float = Field(..., ge=0.0, le=1.0, description="Predicted probability of default in next month.")
    predicted_class: int = Field(..., description="0 = Non-Default (Creditworthy), 1 = Default Risk.")
    risk_category: str = Field(..., description="Risk category: Low, Medium, or High.")
    threshold_applied: float = Field(..., description="Decision threshold applied for classification.")
    recommendation: str = Field(..., description="Decision support recommendation.")
    model_name: str
    model_version: str
    key_risk_drivers: List[Dict[str, Any]]
    disclaimer: str


class BatchPredictionResponse(BaseModel):
    """Structured response for batch applicant screening."""
    total_records: int
    default_count: int
    non_default_count: int
    predicted_default_rate: float
    threshold_applied: float
    predictions: List[SinglePredictionResponse]
    disclaimer: str


class HealthResponse(BaseModel):
    """API health status and system telemetry."""
    status: str
    model_loaded: bool
    model_name: str
    server_time: str
    version: str


class ModelInfoResponse(BaseModel):
    """Detailed model metadata, versioning, threshold parameters, and benchmark metrics."""
    project: str
    company: str
    champion_model_name: str
    champion_model_type: str
    selected_decision_threshold: float
    f1_optimal_threshold: float
    dataset_split: Dict[str, Any]
    test_metrics: Dict[str, Any]
    business_cost_optimization: Dict[str, Any]
    top_shap_features: List[Dict[str, Any]]
    disclaimer: str
