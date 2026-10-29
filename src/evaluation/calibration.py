"""Probability calibration evaluation and reliability analysis for credit scoring."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

logger = logging.getLogger(__name__)


def compute_calibration_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> Dict[str, Any]:
    """Calculate calibration curve, Brier score, and Expected Calibration Error (ECE).

    In FinTech, accurate Probability of Default (PD) is required for Expected Loss
    calculations (EL = PD * LGD * EAD) and capital reserve requirements (Basel II/III).

    Args:
        y_true: Ground truth binary targets (0 or 1).
        y_pred_proba: Predicted default probabilities.
        n_bins: Number of probability bins.
        strategy: 'uniform' (equal width) or 'quantile' (equal frequency).

    Returns:
        Dictionary containing prob_true, prob_pred, Brier score, and ECE.
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)

    prob_true, prob_pred = calibration_curve(
        y_true, y_pred_proba, n_bins=n_bins, strategy=strategy
    )

    brier = float(brier_score_loss(y_true, y_pred_proba))

    # Calculate Expected Calibration Error (ECE)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_assignments = np.digitize(y_pred_proba, bin_edges) - 1
    bin_assignments = np.clip(bin_assignments, 0, n_bins - 1)

    ece = 0.0
    total_samples = len(y_true)
    for b in range(n_bins):
        bin_mask = bin_assignments == b
        bin_size = np.sum(bin_mask)
        if bin_size > 0:
            bin_acc = np.mean(y_true[bin_mask])
            bin_conf = np.mean(y_pred_proba[bin_mask])
            ece += (bin_size / total_samples) * np.abs(bin_acc - bin_conf)

    return {
        "brier_score": round(brier, 4),
        "expected_calibration_error": round(float(ece), 4),
        "prob_true": [round(float(p), 4) for p in prob_true],
        "prob_pred": [round(float(p), 4) for p in prob_pred],
        "n_bins": n_bins,
        "strategy": strategy,
    }
