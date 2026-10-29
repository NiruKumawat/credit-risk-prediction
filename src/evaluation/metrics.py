"""Comprehensive evaluation metrics calculation for credit risk default prediction."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def compute_classification_metrics(
    y_true: np.ndarray | list,
    y_pred_proba: np.ndarray | list,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Calculate key FinTech classification metrics at a given decision threshold.

    In credit risk, accuracy alone is deceptive due to ~78/22 class imbalance.
    We prioritize PR-AUC, ROC-AUC, Recall (detecting defaults), Precision, and Calibration.

    Args:
        y_true: Ground truth binary target (0 or 1).
        y_pred_proba: Predicted probability of positive class (default, class 1).
        threshold: Decision threshold for converting probabilities to binary classes.

    Returns:
        Dictionary of formatted metrics.
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)

    # Convert continuous probabilities to binary decisions based on threshold
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Confusion matrix elements: tn, fp, fn, tp
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Calculate metrics safely
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # Probability-based metrics (threshold-independent)
    try:
        roc_auc = float(roc_auc_score(y_true, y_pred_proba))
    except ValueError:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_true, y_pred_proba))
    except ValueError:
        pr_auc = 0.0

    brier = float(brier_score_loss(y_true, y_pred_proba))

    return {
        "threshold": float(threshold),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "specificity": round(specificity, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "brier_score": round(brier, 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
    }
