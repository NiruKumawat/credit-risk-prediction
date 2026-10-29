"""Plotting utilities for credit risk model evaluation and reporting."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

logger = logging.getLogger(__name__)

# Apply modern aesthetic styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
PALETTE = ["#2563EB", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6"]


def plot_roc_curves(
    models_dict: Dict[str, np.ndarray],
    y_true: np.ndarray,
    output_path: Path,
) -> None:
    """Plot multi-model ROC Curves."""
    plt.figure(figsize=(8, 6), dpi=300)

    for idx, (name, y_pred_proba) in enumerate(models_dict.items()):
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_true, y_pred_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", color=PALETTE[idx % len(PALETTE)], linewidth=2)

    plt.plot([0, 1], [0, 1], "k--", label="Random Chance (AUC = 0.500)", alpha=0.6)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11, fontweight="bold")
    plt.title("Receiver Operating Characteristic (ROC) Comparison", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved ROC curve to {output_path}")


def plot_pr_curves(
    models_dict: Dict[str, np.ndarray],
    y_true: np.ndarray,
    output_path: Path,
) -> None:
    """Plot multi-model Precision-Recall Curves."""
    plt.figure(figsize=(8, 6), dpi=300)
    baseline_rate = np.mean(y_true)

    for idx, (name, y_pred_proba) in enumerate(models_dict.items()):
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        from sklearn.metrics import average_precision_score
        pr_auc = average_precision_score(y_true, y_pred_proba)
        plt.plot(recall, precision, label=f"{name} (PR-AUC = {pr_auc:.3f})", color=PALETTE[idx % len(PALETTE)], linewidth=2)

    plt.axhline(y=baseline_rate, color="k", linestyle="--", label=f"No-Skill Baseline ({baseline_rate:.3f})", alpha=0.6)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall (Default Detection Rate)", fontsize=11, fontweight="bold")
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=11, fontweight="bold")
    plt.title("Precision-Recall (PR) Curve Comparison (Imbalanced Target)", fontsize=13, fontweight="bold")
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved PR curve to {output_path}")


def plot_calibration_curves(
    models_dict: Dict[str, np.ndarray],
    y_true: np.ndarray,
    output_path: Path,
) -> None:
    """Plot multi-model Reliability Diagram (Calibration Curve)."""
    from sklearn.calibration import calibration_curve

    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", alpha=0.6)

    for idx, (name, y_pred_proba) in enumerate(models_dict.items()):
        prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10, strategy="uniform")
        plt.plot(prob_pred, prob_true, marker="o", label=name, color=PALETTE[idx % len(PALETTE)], linewidth=2)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("Mean Predicted Default Probability", fontsize=11, fontweight="bold")
    plt.ylabel("Observed Fraction of Defaults", fontsize=11, fontweight="bold")
    plt.title("Probability Calibration (Reliability Diagram)", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved calibration curve to {output_path}")


def plot_confusion_matrix_heatmap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    title: str = "Confusion Matrix",
) -> None:
    """Plot confusion matrix with counts and percentages."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    labels = [
        [f"{cm[0, 0]}\n({cm_norm[0, 0]:.1%})\n[TN: Re-paid]", f"{cm[0, 1]}\n({cm_norm[0, 1]:.1%})\n[FP: Rejected Good]"],
        [f"{cm[1, 0]}\n({cm_norm[1, 0]:.1%})\n[FN: Defaulted]", f"{cm[1, 1]}\n({cm_norm[1, 1]:.1%})\n[TP: Caught Default]"],
    ]

    plt.figure(figsize=(7, 5.5), dpi=300)
    sns.heatmap(
        cm,
        annot=np.array(labels),
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predicted Non-Default", "Predicted Default"],
        yticklabels=["Actual Non-Default", "Actual Default"],
        annot_kws={"fontsize": 11, "fontweight": "bold"},
    )
    plt.title(title, fontsize=13, fontweight="bold")
    plt.ylabel("True Borrower Outcome", fontsize=11, fontweight="bold")
    plt.xlabel("Model Risk Decision", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved confusion matrix to {output_path}")


def plot_threshold_cost_curve(
    threshold_data: List[Dict[str, float]],
    optimal_threshold: float,
    output_path: Path,
) -> None:
    """Plot business expected loss curve across decision thresholds."""
    thresholds = [r["threshold"] for r in threshold_data]
    costs = [r["total_cost"] for r in threshold_data]
    recalls = [r["recall"] for r in threshold_data]
    precisions = [r["precision"] for r in threshold_data]

    fig, ax1 = plt.subplots(figsize=(9, 5.5), dpi=300)

    color_cost = "#DC2626"
    ax1.set_xlabel("Decision Threshold", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Total Business Expected Loss (NT$)", color=color_cost, fontsize=11, fontweight="bold")
    ax1.plot(thresholds, costs, color=color_cost, linewidth=2.5, label="Total Loss (FN*50k + FP*8k)")
    ax1.axvline(x=optimal_threshold, color="#16A34A", linestyle="--", linewidth=2, label=f"Optimal Threshold ({optimal_threshold:.2f})")
    ax1.axvline(x=0.5, color="gray", linestyle=":", linewidth=1.5, label="Default Threshold (0.50)")
    ax1.tick_params(axis="y", labelcolor=color_cost)

    ax2 = ax1.twinx()
    color_metrics = "#2563EB"
    ax2.set_ylabel("Metric Rate", color=color_metrics, fontsize=11, fontweight="bold")
    ax2.plot(thresholds, recalls, color="#059669", linestyle="-.", linewidth=1.8, label="Recall (Catching Defaults)")
    ax2.plot(thresholds, precisions, color="#2563EB", linestyle="--", linewidth=1.8, label="Precision")
    ax2.tick_params(axis="y", labelcolor=color_metrics)

    plt.title("Decision Threshold vs. Business Expected Loss Trade-Off", fontsize=13, fontweight="bold")
    fig.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved threshold cost curve to {output_path}")


def generate_all_evaluation_plots(
    models_proba: Dict[str, np.ndarray],
    y_true: np.ndarray,
    champion_name: str,
    champion_threshold: float,
    threshold_results: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Generate and save all publication-quality evaluation figures."""
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_roc_curves(models_proba, y_true, output_dir / "roc_curve_comparison.png")
    plot_pr_curves(models_proba, y_true, output_dir / "pr_curve_comparison.png")
    plot_calibration_curves(models_proba, y_true, output_dir / "calibration_curve_comparison.png")

    champ_proba = models_proba[champion_name]
    champ_pred = (champ_proba >= champion_threshold).astype(int)
    plot_confusion_matrix_heatmap(
        y_true,
        champ_pred,
        output_dir / "confusion_matrix_champion.png",
        title=f"Confusion Matrix: {champion_name} (Threshold = {champion_threshold:.2f})",
    )

    plot_threshold_cost_curve(
        threshold_results["threshold_curve"],
        champion_threshold,
        output_dir / "threshold_cost_tradeoff.png",
    )
