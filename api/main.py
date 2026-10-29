"""FastAPI application for Credit Risk and Loan Default Prediction.

Ignosis FinTech Portfolio Service.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    SinglePredictionResponse,
)
from api.service import credit_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Ignosis Credit Risk & Loan Default Prediction API",
    description=(
        "Production-style Machine Learning REST API for credit risk evaluation, "
        "default probability scoring, and local explainability. Built for Ignosis FinTech."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable Cross-Origin Resource Sharing for frontend/dashboard access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def root_redirect():
    """Redirect to API documentation."""
    return {
        "service": "Ignosis Credit Risk API",
        "status": "online",
        "docs": "/docs",
        "health": "/health",
        "model_info": "/model-info",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Check service health and model readiness",
    tags=["System"],
)
def get_health() -> HealthResponse:
    """Return API health status, loaded model identity, and server timestamp."""
    return HealthResponse(
        status="healthy" if credit_service.is_ready else "degraded",
        model_loaded=credit_service.is_ready,
        model_name=credit_service.model_name,
        server_time=datetime.now().isoformat(),
        version=credit_service.model_version,
    )


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Get model architecture, metadata, and benchmark metrics",
    tags=["Model Telemetry"],
)
def get_model_info() -> ModelInfoResponse:
    """Return model architecture, training partition statistics, decision threshold, and evaluation metrics."""
    if not credit_service.metadata:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model metadata is not loaded. Train the model first.",
        )

    meta = credit_service.metadata
    return ModelInfoResponse(
        project=meta.get("project", "Credit Risk Prediction"),
        company=meta.get("company", "Ignosis FinTech"),
        champion_model_name=meta.get("champion_model_name", "RandomForest"),
        champion_model_type=meta.get("champion_model_type", "RandomForestClassifier"),
        selected_decision_threshold=meta.get("selected_decision_threshold", 0.38),
        f1_optimal_threshold=meta.get("f1_optimal_threshold", 0.53),
        dataset_split=meta.get("dataset_split", {}),
        test_metrics=meta.get("test_metrics_optimal_threshold", {}),
        business_cost_optimization=meta.get("business_cost_optimization", {}),
        top_shap_features=meta.get("top_shap_features", []),
        disclaimer=meta.get("disclaimer", ""),
    )


@app.post(
    "/predict",
    response_model=SinglePredictionResponse,
    summary="Evaluate single loan applicant credit risk",
    tags=["Inference"],
)
def predict_single_applicant(request: PredictionRequest) -> SinglePredictionResponse:
    """Predict default probability and risk decision for a single credit applicant.

    Returns:
    - Default probability (0.0 to 1.0)
    - Binary decision (0 = Non-Default, 1 = Default Risk)
    - Applied decision threshold
    - Risk category (Low, Medium, High)
    - Top risk driver attributions
    """
    if not credit_service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready for inference.",
        )

    try:
        response = credit_service.predict_single(
            applicant=request.applicant,
            custom_threshold=request.custom_threshold,
        )
        return response
    except Exception as exc:
        logger.error(f"Prediction failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(exc)}",
        )


@app.post(
    "/predict-batch",
    response_model=BatchPredictionResponse,
    summary="Screen a batch cohort of loan applicants",
    tags=["Inference"],
)
def predict_batch_applicants(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """Screen multiple credit card applicants in a batch for portfolio risk analysis."""
    if not credit_service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready for inference.",
        )

    try:
        response = credit_service.predict_batch(
            applicants=request.applicants,
            custom_threshold=request.custom_threshold,
        )
        return response
    except Exception as exc:
        logger.error(f"Batch prediction failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference error: {str(exc)}",
        )
