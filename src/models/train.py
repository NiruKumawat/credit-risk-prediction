"""End-to-end model training, comparison, evaluation, and serialization pipeline.

Executes reproducible, stratified train/validation/test splits, prevents data leakage,
evaluates 5 distinct models, optimizes decision thresholds, computes SHAP attributions,
and serializes artifacts with versioned metadata.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data.loader import load_credit_card_data, TARGET_COL
from src.features.pipeline import build_preprocessing_pipeline, get_preprocessed_feature_names
from src.models.baseline import build_dummy_baseline
from src.models.logistic_regression import build_logistic_regression
from src.models.tree_models import (
    build_decision_tree,
    build_random_forest,
    build_xgboost_classifier,
    XGBOOST_AVAILABLE,
)
from src.models.explainability import compute_shap_explanations
from src.evaluation.metrics import compute_classification_metrics
from src.evaluation.threshold import find_optimal_business_threshold
from src.evaluation.plots import generate_all_evaluation_plots

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RANDOM_SEED = 42


def get_artifacts_dirs() -> Tuple[Path, Path, Path]:
    """Return artifact directory paths for models, metrics, and figures."""
    project_root = Path(__file__).resolve().parents[2]
    artifacts_dir = project_root / "artifacts"
    models_dir = artifacts_dir / "models"
    figures_dir = artifacts_dir / "figures"
    metrics_dir = artifacts_dir / "metrics"

    for d in [models_dir, figures_dir, metrics_dir]:
        d.mkdir(parents=True, exist_ok=True)

    return models_dir, figures_dir, metrics_dir


def create_stratified_splits(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Perform a reproducible, stratified 60/20/20 train/validation/test split.

    Strict Leakage Prevention:
    - Target column is completely isolated before any splitting.
    - Splitting is performed before any scaling or feature transformations.
    - Class distributions are preserved via stratification across all 3 partitions.
    """
    logger.info("Performing stratified train/validation/test partition...")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # First split: 80% dev (train+val), 20% test
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Second split: From 80% dev, take 25% for validation -> 0.80 * 0.25 = 20% total validation
    val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=val_ratio, stratify=y_dev, random_state=random_state
    )

    logger.info(
        f"Splits created: Train={len(X_train)} (pos={y_train.mean():.3f}), "
        f"Val={len(X_val)} (pos={y_val.mean():.3f}), "
        f"Test={len(X_test)} (pos={y_test.mean():.3f})"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_candidate_models() -> Dict[str, Any]:
    """Instantiate candidate estimators for comparison."""
    models: Dict[str, Any] = {
        "DummyBaseline": build_dummy_baseline(strategy="prior", random_state=RANDOM_SEED),
        "LogisticRegression": build_logistic_regression(C=1.0, class_weight="balanced", random_state=RANDOM_SEED),
        "DecisionTree": build_decision_tree(max_depth=5, min_samples_leaf=15, class_weight="balanced", random_state=RANDOM_SEED),
        "RandomForest": build_random_forest(n_estimators=150, max_depth=8, min_samples_leaf=10, class_weight="balanced", random_state=RANDOM_SEED),
    }

    if XGBOOST_AVAILABLE:
        xgb_model = build_xgboost_classifier(n_estimators=150, max_depth=4, scale_pos_weight=3.5, random_state=RANDOM_SEED)
        if xgb_model is not None:
            models["XGBoost"] = xgb_model

    return models


def train_and_evaluate_all_models() -> Dict[str, Any]:
    """Train all models, evaluate metrics, select champion, and save artifacts."""
    models_dir, figures_dir, metrics_dir = get_artifacts_dirs()

    # 1. Load data
    df = load_credit_card_data()

    # 2. Split with stratification (train/val/test)
    X_train, X_val, X_test, y_train, y_val, y_test = create_stratified_splits(df)

    candidate_models = get_candidate_models()
    trained_pipelines: Dict[str, Pipeline] = {}
    val_predictions_proba: Dict[str, np.ndarray] = {}
    val_metrics_summary: Dict[str, Dict[str, Any]] = {}

    best_score = -1.0
    champion_name = ""

    # Fit feature names template
    template_pipe = build_preprocessing_pipeline()
    template_pipe.fit(X_train)
    feature_names = get_preprocessed_feature_names(template_pipe)

    logger.info(f"Beginning training and evaluation across {len(candidate_models)} models...")

    for name, estimator in candidate_models.items():
        t0 = time.time()
        logger.info(f"Training {name}...")

        pipe = Pipeline(
            steps=[
                ("preprocessor", build_preprocessing_pipeline(scale_numeric=True)),
                ("classifier", estimator),
            ]
        )

        # Fit exclusively on training partition
        pipe.fit(X_train, y_train)
        train_duration = time.time() - t0

        # Predict default probability on validation partition
        y_val_proba = pipe.predict_proba(X_val)[:, 1]

        # Compute validation metrics at default 0.5 threshold
        metrics = compute_classification_metrics(y_val, y_val_proba, threshold=0.5)
        metrics["training_time_seconds"] = round(train_duration, 3)

        trained_pipelines[name] = pipe
        val_predictions_proba[name] = y_val_proba
        val_metrics_summary[name] = metrics

        logger.info(
            f"{name} -> ROC-AUC: {metrics['roc_auc']:.4f}, PR-AUC: {metrics['pr_auc']:.4f}, "
            f"F1: {metrics['f1_score']:.4f}, Recall: {metrics['recall']:.4f}"
        )

        # Selection criterion: PR-AUC (gold standard for imbalanced fraud/credit default)
        selection_metric = metrics["pr_auc"]
        if selection_metric > best_score:
            best_score = selection_metric
            champion_name = name

    logger.info(f"Champion Model selected: {champion_name} (Val PR-AUC = {best_score:.4f})")
    champion_pipeline = trained_pipelines[champion_name]

    # 3. Decision Threshold Optimization for Champion Model
    logger.info(f"Optimizing decision threshold for {champion_name}...")
    threshold_results = find_optimal_business_threshold(
        y_val.values,
        val_predictions_proba[champion_name],
    )
    cost_optimal_threshold = threshold_results["cost_optimal_threshold"]
    f1_optimal_threshold = threshold_results["f1_optimal_threshold"]

    logger.info(
        f"Optimal Decision Threshold: {cost_optimal_threshold:.3f} "
        f"(Estimated cost savings vs 0.5: {threshold_results['percent_cost_savings']:.1f}%)"
    )

    # Re-evaluate champion on validation set at cost-optimal threshold
    champion_val_optimal_metrics = compute_classification_metrics(
        y_val, val_predictions_proba[champion_name], threshold=cost_optimal_threshold
    )

    # 4. Final Evaluation on Strictly Unseen TEST Partition
    logger.info(f"Evaluating Champion Model ({champion_name}) on held-out TEST set...")
    test_proba = champion_pipeline.predict_proba(X_test)[:, 1]

    test_metrics_default = compute_classification_metrics(
        y_test, test_proba, threshold=0.5
    )
    test_metrics_optimal = compute_classification_metrics(
        y_test, test_proba, threshold=cost_optimal_threshold
    )

    logger.info(
        f"Test Set Performance ({champion_name} at threshold {cost_optimal_threshold:.2f}): "
        f"ROC-AUC: {test_metrics_optimal['roc_auc']:.4f}, PR-AUC: {test_metrics_optimal['pr_auc']:.4f}, "
        f"Recall: {test_metrics_optimal['recall']:.4f}, Precision: {test_metrics_optimal['precision']:.4f}, "
        f"F1: {test_metrics_optimal['f1_score']:.4f}"
    )

    # 5. SHAP Feature Attribution Analysis
    logger.info("Computing SHAP explanations for Champion Model...")
    shap_results = compute_shap_explanations(
        champion_pipeline,
        X_val,
        feature_names,
        output_dir=figures_dir,
    )

    # 6. Generate and Save All Evaluation Plots
    logger.info("Generating publication-quality evaluation figures...")
    generate_all_evaluation_plots(
        val_predictions_proba,
        y_val.values,
        champion_name,
        cost_optimal_threshold,
        threshold_results,
        figures_dir,
    )

    # 7. Serialize Best Model Pipeline
    model_save_path = models_dir / "best_model.joblib"
    joblib.dump(champion_pipeline, model_save_path)
    logger.info(f"Serialized champion pipeline to: {model_save_path}")

    # 8. Save Comprehensive Metadata
    metadata = {
        "project": "Credit Risk and Loan Default Prediction System",
        "company": "Ignosis FinTech Portfolio",
        "created_at": datetime.now().isoformat(),
        "random_seed": RANDOM_SEED,
        "champion_model_name": champion_name,
        "champion_model_type": str(champion_pipeline.named_steps["classifier"].__class__.__name__),
        "selected_decision_threshold": cost_optimal_threshold,
        "f1_optimal_threshold": f1_optimal_threshold,
        "dataset_split": {
            "total_samples": len(df),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "train_default_rate": round(float(y_train.mean()), 4),
            "val_default_rate": round(float(y_val.mean()), 4),
            "test_default_rate": round(float(y_test.mean()), 4),
        },
        "validation_benchmark_summary": val_metrics_summary,
        "champion_val_metrics_at_optimal_threshold": champion_val_optimal_metrics,
        "test_metrics_default_threshold_0_5": test_metrics_default,
        "test_metrics_optimal_threshold": test_metrics_optimal,
        "business_cost_optimization": {
            "cost_false_negative": threshold_results["cost_fn"],
            "cost_false_positive": threshold_results["cost_fp"],
            "loss_at_default_0_5": threshold_results["loss_at_default_0_5"],
            "loss_at_optimal_threshold": threshold_results["loss_at_optimal_threshold"],
            "cost_savings": threshold_results["estimated_cost_savings"],
            "percent_cost_savings": threshold_results["percent_cost_savings"],
        },
        "raw_input_features": list(X_train.columns),
        "preprocessed_feature_names": feature_names,
        "top_shap_features": shap_results["top_features"][:10],
        "shap_fallback_used": shap_results["fallback_used"],
        "disclaimer": (
            "Decision-support prototype for risk scoring. Not intended for fully automated "
            "adverse lending decisions without human review in compliance with Fair Lending regulations."
        ),
    }

    metadata_path = models_dir / "model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved model metadata to: {metadata_path}")

    # Also save benchmark metrics in metrics directory
    benchmark_path = metrics_dir / "models_benchmark.json"
    with open(benchmark_path, "w") as f:
        json.dump({
            "champion": champion_name,
            "threshold": cost_optimal_threshold,
            "validation_comparison": val_metrics_summary,
            "test_performance": test_metrics_optimal,
        }, f, indent=2)

    return metadata


if __name__ == "__main__":
    train_and_evaluate_all_models()
