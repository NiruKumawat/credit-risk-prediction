"""Logistic Regression classifier with mathematical explanations and odds ratio analysis.

Mathematical Foundation:
------------------------
1. Sigmoid Function:
   Maps any real-valued linear score z in (-inf, +inf) to probability p in (0, 1):
       sigma(z) = 1 / (1 + exp(-z))
   where z = beta_0 + sum_{j=1}^d (beta_j * x_j).

2. Log-Odds (Logit Transformation):
   The linear model estimates the log-odds of defaulting versus repaying:
       logit(p) = ln( p / (1 - p) ) = beta_0 + sum_{j=1}^d (beta_j * x_j)

3. Odds Ratio Interpretation:
   Holding all other features constant, a one-standard-deviation increase in feature x_j
   multiplies the baseline odds of default by exp(beta_j):
       - If beta_j > 0: exp(beta_j) > 1 -> Feature increases risk of default.
       - If beta_j < 0: exp(beta_j) < 1 -> Feature is protective (decreases risk of default).

4. Objective Function (Binary Cross-Entropy / Log Loss with L2 Regularization):
   L(beta) = - (1/N) * sum_{i=1}^N [ y_i * ln(p_i) + (1 - y_i) * ln(1 - p_i) ] + (1 / 2C) * ||beta||_2^2
   The penalty (1 / 2C) controls model complexity, preventing extreme coefficient values
   caused by multicollinearity among correlated monthly financial statements.

5. Class Imbalance Weighting:
   To prevent the majority non-defaulters from dominating the gradients, `class_weight='balanced'`
   applies weight w_c = N / (2 * N_c) to each class c in the loss function.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def build_logistic_regression(
    C: float = 1.0,
    solver: str = "lbfgs",
    max_iter: int = 1000,
    class_weight: str | Dict[int, float] = "balanced",
    random_state: int = 42,
) -> LogisticRegression:
    """Build a regularized Logistic Regression classifier with class weighting."""
    return LogisticRegression(
        C=C,
        solver=solver,
        max_iter=max_iter,
        class_weight=class_weight,
        random_state=random_state,
    )


def extract_logistic_regression_explanations(
    fitted_pipeline: Pipeline,
    feature_names: List[str],
    top_n: int = 10,
) -> Dict[str, Any]:
    """Extract learned coefficients, convert to Odds Ratios, and rank top risk factors.

    Args:
        fitted_pipeline: Trained Pipeline containing a LogisticRegression final step.
        feature_names: List of preprocessed feature names.
        top_n: Number of top positive and negative features to return.

    Returns:
        Dictionary containing intercept, odds ratios, and top risk/protective drivers.
    """
    model: LogisticRegression = fitted_pipeline.named_steps["classifier"]
    coefficients = model.coef_[0]
    intercept = float(model.intercept_[0])

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients,
        "odds_ratio": np.exp(coefficients),
        "abs_impact": np.abs(coefficients),
    }).sort_values("abs_impact", ascending=False)

    top_risk_drivers = (
        coef_df[coef_df["coefficient"] > 0]
        .head(top_n)
        .to_dict(orient="records")
    )

    top_protective_factors = (
        coef_df[coef_df["coefficient"] < 0]
        .head(top_n)
        .to_dict(orient="records")
    )

    return {
        "model_type": "LogisticRegression",
        "intercept": intercept,
        "baseline_odds": float(np.exp(intercept)),
        "baseline_probability": float(1.0 / (1.0 + np.exp(-intercept))),
        "top_risk_drivers": top_risk_drivers,
        "top_protective_factors": top_protective_factors,
        "all_coefficients": coef_df.to_dict(orient="records"),
    }
