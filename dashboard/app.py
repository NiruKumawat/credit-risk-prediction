"""Ignosis Credit Risk & Loan Default Prediction Dashboard.

Interactive decision-support system for credit underwriting, portfolio risk analysis,
model benchmarking, threshold optimization, and SHAP explainability.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is on sys.path so joblib can unpickle custom transformers from 'src'
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import streamlit as st

# Set page config
st.set_page_config(
    page_title="Ignosis | Credit Risk Prediction System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern FinTech aesthetic
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .disclaimer-box {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 0.8rem 1.2rem;
        border-radius: 4px;
        color: #92400E;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
        font-weight: 500;
    }
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .risk-badge-low {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 0.3rem 0.8rem;
        border-radius: 9999px;
        font-weight: 600;
        display: inline-block;
    }
    .risk-badge-medium {
        background-color: #FEF08A;
        color: #854D0E;
        padding: 0.3rem 0.8rem;
        border-radius: 9999px;
        font-weight: 600;
        display: inline-block;
    }
    .risk-badge-high {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 0.3rem 0.8rem;
        border-radius: 9999px;
        font-weight: 600;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model_and_metadata():
    """Load model pipeline and metadata from artifacts."""
    project_root = Path(__file__).resolve().parents[1]
    metadata_path = project_root / "artifacts" / "models" / "model_metadata.json"
    model_path = project_root / "artifacts" / "models" / "best_model.joblib"

    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

    import joblib
    pipeline = None
    if model_path.exists():
        pipeline = joblib.load(model_path)

    return pipeline, metadata, project_root


pipeline, metadata, project_root = load_model_and_metadata()

# Header & Disclaimer
st.markdown('<div class="main-header">Ignosis | Credit Risk & Default Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Production-Style Machine Learning Decision-Support Prototype for Consumer Lending</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="disclaimer-box">
        ⚠️ <strong>REGULATORY & FAIR LENDING NOTICE:</strong> This application is an experimental decision-support
        prototype for underwriter guidance and risk scoring. It is <strong>NOT</strong> an automated credit denial system.
        All adverse decisions require human underwriter adjudication in compliance with ECOA, FCRA, and Fair Lending standards.
    </div>
    """,
    unsafe_allow_html=True,
)

if pipeline is None:
    st.error("Model artifacts not found! Please run the training pipeline first: `python -m src.models.train`.")
    st.stop()

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/fluency/96/bank-building.png", width=64)
st.sidebar.title("Ignosis Risk Engine")
st.sidebar.markdown(f"**Champion Model:** `{metadata.get('champion_model_name', 'RandomForest')}`")
default_threshold = float(metadata.get("selected_decision_threshold", 0.38))
st.sidebar.markdown(f"**Cost-Optimal Threshold:** `{default_threshold:.2f}`")
st.sidebar.markdown(f"**Training Default Rate:** `{metadata.get('dataset_split', {}).get('train_default_rate', 0.22):.1%}`")
st.sidebar.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Single Loan Calculator",
    "📊 Portfolio Batch Screening",
    "🏆 Model Benchmarks",
    "⚖️ Threshold & Business ROI",
    "🔍 Explainability & SHAP",
])


# -------------------------------------------------------------
# TAB 1: SINGLE LOAN CALCULATOR
# -------------------------------------------------------------
with tab1:
    st.subheader("Applicant Credit Risk Calculator")
    st.markdown("Enter applicant attributes or choose a pre-configured profile to calculate default risk.")

    preset = st.selectbox(
        "Choose Applicant Preset Profile:",
        [
            "Custom Input",
            "Prime Low-Risk Borrower (Steady Repayment, High Limit)",
            "Moderate-Risk Borrower (Revolving Debt, High Utilization)",
            "Delinquent High-Risk Borrower (Recent 2-Month Delinquency)",
        ],
    )

    # Preset profiles
    if preset == "Prime Low-Risk Borrower (Steady Repayment, High Limit)":
        p_limit = 250000.0
        p_sex = 2
        p_edu = 1
        p_mrg = 2
        p_age = 34
        p_p1, p_p2, p_p3, p_p4, p_p5, p_p6 = -1, -1, -1, -1, -1, -1
        p_b1, p_b2, p_b3, p_b4, p_b5, p_b6 = 12000.0, 11000.0, 9000.0, 8000.0, 5000.0, 3000.0
        p_a1, p_a2, p_a3, p_a4, p_a5, p_a6 = 12000.0, 11000.0, 9000.0, 8000.0, 5000.0, 3000.0
    elif preset == "Moderate-Risk Borrower (Revolving Debt, High Utilization)":
        p_limit = 60000.0
        p_sex = 1
        p_edu = 2
        p_mrg = 1
        p_age = 29
        p_p1, p_p2, p_p3, p_p4, p_p5, p_p6 = 0, 0, 0, 0, 0, 0
        p_b1, p_b2, p_b3, p_b4, p_b5, p_b6 = 55000.0, 54000.0, 53000.0, 52000.0, 50000.0, 48000.0
        p_a1, p_a2, p_a3, p_a4, p_a5, p_a6 = 2500.0, 2500.0, 2500.0, 2500.0, 2500.0, 2500.0
    elif preset == "Delinquent High-Risk Borrower (Recent 2-Month Delinquency)":
        p_limit = 40000.0
        p_sex = 1
        p_edu = 3
        p_mrg = 1
        p_age = 45
        p_p1, p_p2, p_p3, p_p4, p_p5, p_p6 = 2, 2, 0, 0, 0, 0
        p_b1, p_b2, p_b3, p_b4, p_b5, p_b6 = 38000.0, 37000.0, 36000.0, 35000.0, 34000.0, 33000.0
        p_a1, p_a2, p_a3, p_a4, p_a5, p_a6 = 0.0, 1500.0, 1500.0, 1500.0, 1500.0, 1500.0
    else:
        p_limit = 100000.0
        p_sex = 1
        p_edu = 2
        p_mrg = 2
        p_age = 30
        p_p1, p_p2, p_p3, p_p4, p_p5, p_p6 = 0, 0, 0, 0, 0, 0
        p_b1, p_b2, p_b3, p_b4, p_b5, p_b6 = 20000.0, 19000.0, 18000.0, 17000.0, 16000.0, 15000.0
        p_a1, p_a2, p_a3, p_a4, p_a5, p_a6 = 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Demographics & Financial Capacity**")
        limit_bal = st.number_input("Credit Limit (NT$)", min_value=10000.0, max_value=1000000.0, value=float(p_limit), step=10000.0)
        age = st.slider("Age (Years)", min_value=18, max_value=85, value=int(p_age))
        sex = st.selectbox("Gender", options=[1, 2], index=0 if p_sex == 1 else 1, format_func=lambda x: "Male (1)" if x == 1 else "Female (2)")
        education = st.selectbox("Education Level", options=[1, 2, 3, 4], index=p_edu - 1, format_func=lambda x: {1: "Graduate School", 2: "University", 3: "High School", 4: "Others"}[x])
        marriage = st.selectbox("Marital Status", options=[1, 2, 3], index=p_mrg - 1, format_func=lambda x: {1: "Married", 2: "Single", 3: "Others"}[x])

    with col2:
        st.markdown("**Past 6 Months Repayment Status (PAY_1 to PAY_6)**")
        st.caption("-1=Paid full, 0=Revolving credit, 1-8=Months delayed")
        pay_1 = st.slider("Sept 2005 (PAY_1 - Latest)", min_value=-2, max_value=8, value=int(p_p1))
        pay_2 = st.slider("Aug 2005 (PAY_2)", min_value=-2, max_value=8, value=int(p_p2))
        pay_3 = st.slider("July 2005 (PAY_3)", min_value=-2, max_value=8, value=int(p_p3))
        pay_4 = st.slider("June 2005 (PAY_4)", min_value=-2, max_value=8, value=int(p_p4))
        pay_5 = st.slider("May 2005 (PAY_5)", min_value=-2, max_value=8, value=int(p_p5))
        pay_6 = st.slider("April 2005 (PAY_6)", min_value=-2, max_value=8, value=int(p_p6))

    with col3:
        st.markdown("**Recent Bill & Payment History (NT$)**")
        b1 = st.number_input("Latest Bill (Sept)", min_value=0.0, max_value=1000000.0, value=float(p_b1), step=5000.0)
        p1 = st.number_input("Latest Payment (Sept)", min_value=0.0, max_value=500000.0, value=float(p_a1), step=2000.0)
        b2 = st.number_input("Bill (Aug)", min_value=0.0, max_value=1000000.0, value=float(p_b2), step=5000.0)
        p2 = st.number_input("Payment (Aug)", min_value=0.0, max_value=500000.0, value=float(p_a2), step=2000.0)
        threshold_override = st.slider("Decision Threshold Applied", min_value=0.10, max_value=0.90, value=default_threshold, step=0.01)

    if st.button("Evaluate Applicant Credit Risk", type="primary", use_container_width=True):
        input_data = {
            "LIMIT_BAL": limit_bal,
            "SEX": sex,
            "EDUCATION": education,
            "MARRIAGE": marriage,
            "AGE": age,
            "PAY_1": pay_1, "PAY_2": pay_2, "PAY_3": pay_3, "PAY_4": pay_4, "PAY_5": pay_5, "PAY_6": pay_6,
            "BILL_AMT1": b1, "BILL_AMT2": b2, "BILL_AMT3": p_b3, "BILL_AMT4": p_b4, "BILL_AMT5": p_b5, "BILL_AMT6": p_b6,
            "PAY_AMT1": p1, "PAY_AMT2": p2, "PAY_AMT3": p_a3, "PAY_AMT4": p_a4, "PAY_AMT5": p_a5, "PAY_AMT6": p_a6,
        }
        df_single = pd.DataFrame([input_data])
        proba = float(pipeline.predict_proba(df_single)[0, 1])
        is_default = int(proba >= threshold_override)

        st.markdown("---")
        st.subheader("Risk Assessment Results")

        res_col1, res_col2, res_col3, res_col4 = st.columns(4)
        with res_col1:
            st.metric("Predicted Default Probability", f"{proba:.1%}")
        with res_col2:
            st.metric("Operating Threshold", f"{threshold_override:.1%}")
        with res_col3:
            if proba < 0.20:
                badge = '<span class="risk-badge-low">LOW RISK</span>'
            elif proba < threshold_override:
                badge = '<span class="risk-badge-medium">MODERATE RISK</span>'
            else:
                badge = '<span class="risk-badge-high">HIGH RISK</span>'
            st.markdown(f"**Risk Category:**<br>{badge}", unsafe_allow_html=True)
        with res_col4:
            if is_default:
                rec_html = '<span style="color:#DC2626; font-weight:bold;">⚠️ REFER TO SENIOR UNDERWRITER</span>'
            else:
                rec_html = '<span style="color:#16A34A; font-weight:bold;">✅ STANDARD APPROVAL</span>'
            st.markdown(f"**System Recommendation:**<br>{rec_html}", unsafe_allow_html=True)

        # Quick feature breakdown
        st.markdown("#### Key Applicant Financial Signals")
        utilization = b1 / max(limit_bal, 1.0)
        pay_ratio = p1 / (b1 + 1.0)
        sig_col1, sig_col2, sig_col3 = st.columns(3)
        sig_col1.metric("Credit Utilization Ratio", f"{utilization:.1%}", help="Latest bill divided by credit limit")
        sig_col2.metric("Payment-to-Bill Ratio", f"{pay_ratio:.1%}", help="Latest payment divided by bill")
        sig_col3.metric("Latest Repayment Status", f"PAY_1 = {pay_1}", help=">0 indicates delinquent months")


# -------------------------------------------------------------
# TAB 2: PORTFOLIO & BATCH SCREENING
# -------------------------------------------------------------
with tab2:
    st.subheader("Batch Applicant Cohort Screening")
    st.markdown("Upload a CSV cohort of applicants or screen a sample partition from the validation dataset.")

    sample_csv_path = project_root / "data" / "credit_card_clients_cleaned.csv"
    if sample_csv_path.exists():
        if st.button("Load 50 Sample Applicants from Validation Partition", use_container_width=False):
            df_sample = pd.read_csv(sample_csv_path).head(50)
            st.session_state["batch_df"] = df_sample

    uploaded_file = st.file_uploader("Upload Applicant CSV:", type=["csv"])
    if uploaded_file is not None:
        st.session_state["batch_df"] = pd.read_csv(uploaded_file)

    if "batch_df" in st.session_state:
        df_batch = st.session_state["batch_df"].copy()
        target_col = "default_payment_next_month"
        has_ground_truth = target_col in df_batch.columns

        X_eval = df_batch.drop(columns=[target_col]) if has_ground_truth else df_batch

        batch_probas = pipeline.predict_proba(X_eval)[:, 1]
        batch_preds = (batch_probas >= default_threshold).astype(int)

        df_batch["default_probability"] = np.round(batch_probas, 4)
        df_batch["risk_decision"] = np.where(batch_preds == 1, "High Risk (Review)", "Approved")

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Applicants Screened", len(df_batch))
        kpi2.metric("Approved (Low/Medium Risk)", int(np.sum(batch_preds == 0)))
        kpi3.metric("Flagged for Review (High Risk)", int(np.sum(batch_preds == 1)))
        kpi4.metric("Flagged Default Rate", f"{np.mean(batch_preds):.1%}")

        st.markdown("#### Portfolio Risk Distribution")
        chart_data = pd.DataFrame({"Default Probability": batch_probas})
        st.bar_chart(chart_data["Default Probability"].value_counts(bins=10).sort_index())

        st.markdown("#### Screened Applicants Table")
        st.dataframe(df_batch[["LIMIT_BAL", "AGE", "PAY_1", "BILL_AMT1", "PAY_AMT1", "default_probability", "risk_decision"]].head(25), use_container_width=True)

        csv_download = df_batch.to_csv(index=False).encode("utf-8")
        st.download_button("Download Full Predictions CSV", csv_download, "ignosis_credit_predictions.csv", "text/csv")


# -------------------------------------------------------------
# TAB 3: MODEL BENCHMARK & EVALUATION
# -------------------------------------------------------------
with tab3:
    st.subheader("Model Benchmark & Evaluation Suite")
    st.markdown("Comparative performance of 5 evaluated model architectures on the stratified validation dataset.")

    benchmarks = metadata.get("validation_benchmark_summary", {})
    if benchmarks:
        summary_rows = []
        for name, m in benchmarks.items():
            summary_rows.append({
                "Model Architecture": name,
                "PR-AUC (Target)": m.get("pr_auc", 0.0),
                "ROC-AUC": m.get("roc_auc", 0.0),
                "Recall (Catching Defaults)": m.get("recall", 0.0),
                "Precision": m.get("precision", 0.0),
                "F1 Score": m.get("f1_score", 0.0),
                "Brier Score (Calibration)": m.get("brier_score", 0.0),
                "Training Time (s)": m.get("training_time_seconds", 0.0),
            })
        bench_df = pd.DataFrame(summary_rows).sort_values("PR-AUC (Target)", ascending=False)
        st.dataframe(bench_df, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Evaluation Curves (Publication Quality Artifacts)")
    fig_col1, fig_col2 = st.columns(2)

    roc_path = project_root / "artifacts" / "figures" / "roc_curve_comparison.png"
    pr_path = project_root / "artifacts" / "figures" / "pr_curve_comparison.png"
    calib_path = project_root / "artifacts" / "figures" / "calibration_curve_comparison.png"
    cm_path = project_root / "artifacts" / "figures" / "confusion_matrix_champion.png"

    if roc_path.exists():
        fig_col1.image(str(roc_path), caption="ROC Curves: Champion achieves 0.795 AUC", use_container_width=True)
    if pr_path.exists():
        fig_col2.image(str(pr_path), caption="Precision-Recall Curves: Champion achieves 0.579 PR-AUC", use_container_width=True)

    fig_col3, fig_col4 = st.columns(2)
    if calib_path.exists():
        fig_col3.image(str(calib_path), caption="Probability Calibration Curves (Reliability Diagram)", use_container_width=True)
    if cm_path.exists():
        fig_col4.image(str(cm_path), caption="Champion Confusion Matrix at Optimal Threshold (0.38)", use_container_width=True)


# -------------------------------------------------------------
# TAB 4: THRESHOLD & BUSINESS ROI
# -------------------------------------------------------------
with tab4:
    st.subheader("Decision Threshold Optimization & Business Financial Impact")
    st.markdown(
        """
        In consumer lending, a standard 0.50 threshold treats the cost of approving a defaulter (False Negative)
        identically to rejecting a good borrower (False Positive). In reality:
        - **False Negative (Default Loss):** Loss of principal and collection overhead (~NT$ 50,000).
        - **False Positive (Lost Interest):** Lost net interest margin & customer acquisition cost (~NT$ 8,000).
        """
    )

    cost_opt = metadata.get("business_cost_optimization", {})
    if cost_opt:
        c1, c2, c3 = st.columns(3)
        c1.metric("Loss at Default Threshold (0.50)", f"NT$ {cost_opt.get('loss_at_default_0_5', 0):,.0f}")
        c2.metric("Loss at Optimal Threshold (0.38)", f"NT$ {cost_opt.get('loss_at_optimal_threshold', 0):,.0f}")
        c3.metric("Net Financial Savings", f"NT$ {cost_opt.get('cost_savings', 0):,.0f}", delta=f"{cost_opt.get('percent_cost_savings', 0):.1f}% Cost Reduction")

    tradeoff_path = project_root / "artifacts" / "figures" / "threshold_cost_tradeoff.png"
    if tradeoff_path.exists():
        st.image(str(tradeoff_path), caption="Total Portfolio Expected Loss vs. Decision Threshold", use_container_width=True)


# -------------------------------------------------------------
# TAB 5: EXPLAINABILITY & SHAP
# -------------------------------------------------------------
with tab5:
    st.subheader("Model Interpretability & SHAP Feature Attributions")
    st.markdown(
        """
        To maintain transparency and auditability, we employ **SHAP (SHapley Additive exPlanations)**
        rooted in cooperative game theory to measure feature contributions to default risk.
        """
    )

    shap_path = project_root / "artifacts" / "figures" / "shap_summary_plot.png"
    if shap_path.exists():
        st.image(str(shap_path), caption="SHAP Summary Beeswarm Plot: Impact on Credit Default Risk", use_container_width=True)

    st.markdown("#### Top Predictive Financial Drivers (Global SHAP)")
    top_shap = metadata.get("top_shap_features", [])
    if top_shap:
        st.dataframe(pd.DataFrame(top_shap), use_container_width=True)

    st.markdown(
        """
        > **Key Engineering Insight:** Notice how domain-engineered features (`months_delayed_count`, `max_delay_months`,
        `avg_pay_amt`, and `credit_utilization_ratio`) rank among the most influential predictors, outperforming raw
        demographic variables. Delinquency history is the primary behavioral driver of credit default.
        """
    )
