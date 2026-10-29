"""Unit tests for model training, metrics calculation, calibration, and threshold optimization."""

import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.features.pipeline import build_preprocessing_pipeline
from src.models.baseline import build_dummy_baseline
from src.models.logistic_regression import build_logistic_regression
from src.models.tree_models import build_decision_tree, build_random_forest
from src.evaluation.metrics import compute_classification_metrics
from src.evaluation.calibration import compute_calibration_curve
from src.evaluation.threshold import find_optimal_business_threshold


@pytest.fixture
def tiny_training_data(sample_applicants_df):
    """Provide a minimal dataset with features and target for quick model tests."""
    df = pd.concat([sample_applicants_df, sample_applicants_df], ignore_index=True)
    y = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 1])
    return df, y


def test_dummy_baseline(tiny_training_data):
    """Test DummyClassifier baseline outputs valid probabilities."""
    X, y = tiny_training_data
    pipe = Pipeline([
        ("preprocessor", build_preprocessing_pipeline()),
        ("classifier", build_dummy_baseline()),
    ])
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)[:, 1]
    assert len(proba) == len(y)
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)


def test_logistic_regression_training(tiny_training_data):
    """Test LogisticRegression fits and produces valid probabilities."""
    X, y = tiny_training_data
    pipe = Pipeline([
        ("preprocessor", build_preprocessing_pipeline(scale_numeric=True)),
        ("classifier", build_logistic_regression()),
    ])
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)[:, 1]
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)


def test_random_forest_training(tiny_training_data):
    """Test RandomForest classifier fits and outputs probabilities."""
    X, y = tiny_training_data
    pipe = Pipeline([
        ("preprocessor", build_preprocessing_pipeline()),
        ("classifier", build_random_forest(n_estimators=10, max_depth=3)),
    ])
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)[:, 1]
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)


def test_compute_classification_metrics():
    """Test metrics dictionary output format and bounds."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.4, 0.7, 0.8, 0.9])
    metrics = compute_classification_metrics(y_true, y_proba, threshold=0.5)

    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert "brier_score" in metrics
    assert "confusion_matrix" in metrics
    assert metrics["roc_auc"] == 1.0  # Perfect separation
    assert metrics["recall"] == 1.0


def test_threshold_optimization():
    """Test business cost optimization finds threshold that reduces financial loss."""
    rng = np.random.RandomState(42)
    y_true = rng.choice([0, 1], size=100, p=[0.75, 0.25])
    y_proba = np.clip(y_true * 0.5 + rng.uniform(0.1, 0.4, size=100), 0.01, 0.99)

    res = find_optimal_business_threshold(y_true, y_proba, cost_fn=50000, cost_fp=8000)
    assert "cost_optimal_threshold" in res
    assert 0.05 <= res["cost_optimal_threshold"] <= 0.95
    assert res["loss_at_optimal_threshold"] <= res["loss_at_default_0_5"]


def test_calibration_curve():
    """Test calibration metrics and ECE computation."""
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.2, 0.3, 0.7, 0.8])
    calib = compute_calibration_curve(y_true, y_proba, n_bins=5)
    assert "brier_score" in calib
    assert "expected_calibration_error" in calib
    assert calib["brier_score"] >= 0.0
