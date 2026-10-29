"""Tree-based models: Decision Tree, Random Forest, and XGBoost classifiers.

Mathematical Foundations & Theory:
----------------------------------
1. Decision Trees:
   - Split Criterion (Gini Impurity):
       G = 1 - sum_{k=0}^1 (p_k)^2
     At each internal node, the algorithm greedily searches across all candidate features
     and thresholds to maximize Impurity Reduction:
       Delta G = G_parent - (N_left / N_parent) * G_left - (N_right / N_parent) * G_right.
   - Overfitting & Pruning:
     An unconstrained tree will partition data until every leaf contains a single sample,
     memorizing noise. Regularization via `max_depth`, `min_samples_split`, and
     `min_samples_leaf` bounds model capacity, trading minor bias for major variance reduction.

2. Random Forest (Bagging & Feature Subspacing):
   - Trains an ensemble of B decorrelated trees using bootstrap sampling with replacement.
   - At each split, a random subset of m = sqrt(d) features is considered.
   - Variance Reduction Theorem:
     If each individual tree has variance sigma^2 and average pairwise correlation rho:
       Var(ensemble) = rho * sigma^2 + ((1 - rho) / B) * sigma^2
     As B grows large, variance decreases to rho * sigma^2. Decorrelating trees via random
     feature subsampling lowers rho, achieving lower generalization error than individual trees.

3. XGBoost (Gradient Boosted Trees):
   - Sequentially minimizes a regularized objective using second-order Taylor approximation:
       obj^(t) approx sum_{i=1}^N [ g_i * f_t(x_i) + 0.5 * h_i * (f_t(x_i))^2 ] + Omega(f_t)
     where g_i = partial_yhat l(y_i, yhat^(t-1)) is gradient, h_i is Hessian,
     and Omega(f_t) = gamma * T + 0.5 * lambda * sum(w_j^2) penalizes tree complexity.
   - Imbalance Handling:
     `scale_pos_weight = count(negative) / count(positive)` scales the gradients of positive
     cases (defaulters) so the model focuses equally on detecting the minority class.

4. Feature Importance:
   - Mean Decrease in Impurity (MDI): Total impurity reduction accumulated by a feature across splits.
   - Permutation Importance: Decrease in validation metric when values of a feature are randomly shuffled.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.info("XGBoost is not installed. Tree suite will use Decision Tree & Random Forest.")


def build_decision_tree(
    max_depth: int = 5,
    min_samples_split: int = 20,
    min_samples_leaf: int = 10,
    class_weight: str | Dict[int, float] = "balanced",
    random_state: int = 42,
) -> DecisionTreeClassifier:
    """Build a regularized Decision Tree classifier.

    Controlled max_depth prevents overfitting on financial noise.
    """
    return DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
    )


def build_random_forest(
    n_estimators: int = 200,
    max_depth: int = 10,
    min_samples_split: int = 15,
    min_samples_leaf: int = 5,
    class_weight: str | Dict[int, float] = "balanced",
    n_jobs: int = -1,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Build a Random Forest ensemble with balanced class weights."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        n_jobs=n_jobs,
        random_state=random_state,
    )


def build_xgboost_classifier(
    n_estimators: int = 200,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    scale_pos_weight: float = 3.5,  # ~78% non-default / ~22% default
    random_state: int = 42,
) -> Optional[Any]:
    """Build an XGBoost classifier if installed; otherwise returns None."""
    if not XGBOOST_AVAILABLE:
        return None

    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        eval_metric="logloss",
        n_jobs=-1,
    )


def extract_tree_feature_importance(
    model: Any,
    feature_names: List[str],
    top_n: int = 15,
) -> List[Dict[str, Any]]:
    """Extract and rank MDI feature importances from a fitted tree or ensemble model."""
    if not hasattr(model, "feature_importances_"):
        return []

    importances = model.feature_importances_
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    return df.head(top_n).to_dict(orient="records")
