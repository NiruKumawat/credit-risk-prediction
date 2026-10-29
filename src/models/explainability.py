"""Model explainability using SHAP (SHapley Additive exPlanations) and fallback methods.

Ethical & Domain Notice:
------------------------
SHAP values quantify how features contribute to this specific model's mathematical output
relative to its baseline expectation. These are explanations of MODEL BEHAVIOR based on
statistical associations learned from training data, NOT proof of causality or real-world creditworthiness.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

EXPLAINABILITY_DISCLAIMER = (
    "NOTICE: SHAP values represent statistical feature attribution within this predictive model "
    "relative to its baseline training prior. They reflect historical dataset correlations and "
    "MUST NOT be construed as causal proof, counterfactual guarantees, or automated credit decisions."
)


def compute_shap_explanations(
    fitted_pipeline: Pipeline,
    X_val: pd.DataFrame,
    feature_names: List[str],
    output_dir: Optional[Path] = None,
    max_samples: int = 500,
) -> Dict[str, Any]:
    """Calculate global and local SHAP values with robust fallback if SHAP encounters errors.

    Args:
        fitted_pipeline: Trained Scikit-learn Pipeline with preprocessor and classifier.
        X_val: Raw validation DataFrame.
        feature_names: List of preprocessed feature names.
        output_dir: Optional directory to save SHAP summary plot.
        max_samples: Maximum validation instances to evaluate for performance.

    Returns:
        Dictionary with mean absolute SHAP values, ranking, and explanation metadata.
    """
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    classifier = fitted_pipeline.named_steps["classifier"]

    # Transform validation features
    X_val_trans = preprocessor.transform(X_val)
    if hasattr(X_val_trans, "toarray"):
        X_val_trans = X_val_trans.toarray()

    sample_size = min(len(X_val_trans), max_samples)
    X_sample = X_val_trans[:sample_size]

    shap_results: Dict[str, Any] = {
        "disclaimer": EXPLAINABILITY_DISCLAIMER,
        "fallback_used": False,
        "sample_size": sample_size,
        "top_features": [],
    }

    try:
        import shap

        logger.info(f"Computing SHAP explanations using {classifier.__class__.__name__}...")

        # Select appropriate explainer based on model family
        if hasattr(classifier, "feature_importances_"):
            explainer = shap.TreeExplainer(classifier)
            raw_shap = explainer.shap_values(X_sample)
            # Binary classification handling in shap (can be list [class0, class1] or 3D array)
            if isinstance(raw_shap, list):
                shap_values = raw_shap[1]  # positive class (default)
            elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 3:
                shap_values = raw_shap[:, :, 1]
            else:
                shap_values = raw_shap
        else:
            # Linear model or fallback explainer
            background = X_sample[:min(50, len(X_sample))]
            explainer = shap.LinearExplainer(classifier, background)
            shap_values = explainer.shap_values(X_sample)

        # Calculate mean absolute SHAP importance per feature
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

        importance_df = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": mean_abs_shap,
        }).sort_values("mean_abs_shap", ascending=False)

        shap_results["top_features"] = importance_df.head(15).to_dict(orient="records")

        # Generate and save SHAP summary beeswarm/bar plot if output_dir provided
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            plot_path = output_dir / "shap_summary_plot.png"

            plt.figure(figsize=(10, 6), dpi=300)
            shap.summary_plot(
                shap_values,
                X_sample,
                feature_names=feature_names,
                max_display=12,
                show=False,
            )
            plt.title("SHAP Feature Attribution (Impact on Default Risk)", fontsize=12, fontweight="bold")
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"Saved SHAP summary plot to {plot_path}")

    except Exception as exc:
        logger.warning(f"SHAP explanation computation failed ({exc}). Engaging fallback importance...")
        shap_results["fallback_used"] = True
        shap_results["error_message"] = str(exc)

        # Fallback to model's native feature importances or coefficients
        if hasattr(classifier, "feature_importances_"):
            importances = classifier.feature_importances_
        elif hasattr(classifier, "coef_"):
            importances = np.abs(classifier.coef_[0])
        else:
            importances = np.ones(len(feature_names)) / len(feature_names)

        importance_df = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": importances,
        }).sort_values("mean_abs_shap", ascending=False)

        shap_results["top_features"] = importance_df.head(15).to_dict(orient="records")

    return shap_results


def explain_single_applicant(
    fitted_pipeline: Pipeline,
    applicant_df: pd.DataFrame,
    feature_names: List[str],
    top_k: int = 5,
) -> Dict[str, Any]:
    """Generate local feature attributions for a single applicant (for API and Dashboard).

    Args:
        fitted_pipeline: Trained pipeline.
        applicant_df: 1-row raw DataFrame with applicant features.
        feature_names: Preprocessed feature names.
        top_k: Number of highest risk and protective factors to return.

    Returns:
        Structured breakdown of top risk-increasing and risk-decreasing features.
    """
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    classifier = fitted_pipeline.named_steps["classifier"]

    X_trans = preprocessor.transform(applicant_df)
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()

    try:
        import shap
        if hasattr(classifier, "feature_importances_"):
            explainer = shap.TreeExplainer(classifier)
            raw = explainer.shap_values(X_trans)
            if isinstance(raw, list):
                vals = raw[1][0]
            elif isinstance(raw, np.ndarray) and raw.ndim == 3:
                vals = raw[0, :, 1]
            else:
                vals = raw[0]
        else:
            # Linear coefficients approximation
            vals = (X_trans[0] * classifier.coef_[0])
    except Exception:
        # Fallback approximation using feature values * importances
        if hasattr(classifier, "feature_importances_"):
            vals = X_trans[0] * classifier.feature_importances_
        else:
            vals = np.zeros(len(feature_names))

    df_attr = pd.DataFrame({
        "feature": feature_names,
        "attribution": vals,
        "abs_attribution": np.abs(vals),
    }).sort_values("abs_attribution", ascending=False)

    risk_drivers = df_attr[df_attr["attribution"] > 0].head(top_k).to_dict(orient="records")
    protective_drivers = df_attr[df_attr["attribution"] < 0].head(top_k).to_dict(orient="records")

    return {
        "disclaimer": EXPLAINABILITY_DISCLAIMER,
        "risk_increasing_factors": risk_drivers,
        "risk_decreasing_factors": protective_drivers,
    }
