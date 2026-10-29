"""Decision threshold optimization based on FinTech business cost trade-offs."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)

# Typical FinTech credit loss parameters (in NT$ or normalized currency units)
# Default cost (False Negative): loss of unpaid balance / recovery collection cost
DEFAULT_FN_COST = 50000.0  # Approx $1,500 USD equivalent in NTD
# Rejection cost (False Positive): lost net interest margin & customer acquisition cost
DEFAULT_FP_COST = 8000.0   # Approx $250 USD equivalent in NTD


def find_optimal_business_threshold(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    cost_fn: float = DEFAULT_FN_COST,
    cost_fp: float = DEFAULT_FP_COST,
    num_steps: int = 91,
) -> Dict[str, Any]:
    """Find the optimal decision threshold that minimizes business expected financial loss.

    In credit lending, the cost of default (FN) is typically 5x to 10x higher than the
    lost interest margin of rejecting a creditworthy borrower (FP). Therefore, a naive
    threshold of 0.5 leads to severe financial losses.

    Args:
        y_true: Ground truth binary target (0 or 1).
        y_pred_proba: Model predicted probabilities of default.
        cost_fn: Financial penalty for False Negative (unidentified defaulter).
        cost_fp: Financial penalty for False Positive (rejected creditworthy borrower).
        num_steps: Number of threshold evaluation points between 0.05 and 0.95.

    Returns:
        Dictionary containing optimal threshold, baseline threshold comparison, and curve data.
    """
    thresholds = np.linspace(0.05, 0.95, num_steps)
    results: List[Dict[str, float]] = []

    best_loss = float("inf")
    best_threshold_cost = 0.5

    best_f1 = -1.0
    best_threshold_f1 = 0.5

    for t in thresholds:
        y_pred = (y_pred_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        total_cost = (fn * cost_fn) + (fp * cost_fp)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        youden_j = (tp / (tp + fn)) + (tn / (tn + fp)) - 1.0

        if total_cost < best_loss:
            best_loss = total_cost
            best_threshold_cost = float(t)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold_f1 = float(t)

        results.append({
            "threshold": round(float(t), 3),
            "total_cost": round(float(total_cost), 2),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "youden_j": round(float(youden_j), 4),
            "fp_count": int(fp),
            "fn_count": int(fn),
            "tp_count": int(tp),
            "tn_count": int(tn),
        })

    # Compare 0.5 default vs Cost-Optimal threshold
    default_record = next(r for r in results if np.isclose(r["threshold"], 0.5, atol=0.015))
    optimal_cost_record = next(r for r in results if np.isclose(r["threshold"], best_threshold_cost, atol=0.005))

    cost_savings = default_record["total_cost"] - optimal_cost_record["total_cost"]
    percent_savings = (cost_savings / default_record["total_cost"]) * 100 if default_record["total_cost"] > 0 else 0.0

    return {
        "cost_optimal_threshold": round(best_threshold_cost, 3),
        "f1_optimal_threshold": round(best_threshold_f1, 3),
        "cost_fn": cost_fn,
        "cost_fp": cost_fp,
        "loss_at_optimal_threshold": optimal_cost_record["total_cost"],
        "loss_at_default_0_5": default_record["total_cost"],
        "estimated_cost_savings": round(cost_savings, 2),
        "percent_cost_savings": round(percent_savings, 2),
        "threshold_curve": results,
    }
