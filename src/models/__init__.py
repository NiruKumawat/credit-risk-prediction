"""Model architectures, training, and explainability package."""

from src.models.baseline import build_dummy_baseline
from src.models.logistic_regression import build_logistic_regression
from src.models.tree_models import (
    build_decision_tree,
    build_random_forest,
    build_xgboost_classifier,
)
from src.models.explainability import compute_shap_explanations

__all__ = [
    "build_dummy_baseline",
    "build_logistic_regression",
    "build_decision_tree",
    "build_random_forest",
    "build_xgboost_classifier",
    "compute_shap_explanations",
]
