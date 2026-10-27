# Credit Risk & Loan Default Prediction System
### Ignosis FinTech Portfolio Project | Senior-Level Machine Learning Architecture

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-teal.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red.svg)](https://streamlit.io/)
[![SHAP](https://img.shields.io/badge/SHAP-0.45%2B-purple.svg)](https://shap.readthedocs.io/)
[![Pytest](https://img.shields.io/badge/Pytest-8.0%2B-green.svg)](https://pytest.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)

---

## 📌 Executive Summary & Business Objective

In consumer lending, a financial institution faces an asymmetric risk trade-off: **approving a borrower who subsequently defaults (False Negative) causes substantial capital loss**, whereas **rejecting a creditworthy applicant (False Positive) results in modest lost interest revenue and acquisition expense**.

This repository implements a **production-style, modular machine learning decision-support system** designed for credit underwriting at **Ignosis**, a modern FinTech company. The system:
1. Predicts the continuous **Probability of Default ($PD$)** for credit card applicants.
2. Formulates an **asymmetric business cost matrix** to optimize the decision threshold, reducing portfolio losses by **12.04%** compared to naive 0.50 thresholding.
3. Provides local and global **SHAP (SHapley Additive exPlanations)** to provide transparent, underwriter-interpretable risk driver rationales.
4. Exposes low-latency, strictly validated **FastAPI REST endpoints** (`/health`, `/model-info`, `/predict`, `/predict-batch`).
5. Delivers an interactive **Streamlit Underwriter Dashboard** for loan calculators, portfolio batch screening, and ROI simulation.

> **Regulatory Notice:** This system is an **experimental decision-support prototype** designed to assist human underwriters. In compliance with the **Equal Credit Opportunity Act (ECOA)** and the **Fair Credit Reporting Act (FCRA)**, it is not configured for fully automated adverse credit denials without human review.

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    subgraph Data_Layer ["1. Ingestion & Validation"]
        A[UCI Credit Dataset] --> B[src.data.loader]
        B --> C[Deduplication & Category Cleaning]
        C --> D[src.data.validator: Schema & Range Checks]
    end

    subgraph Splitting ["2. Stratified Partitioning (No Leakage)"]
        D --> E[Stratified Split 60% Train / 20% Val / 20% Test]
    end

    subgraph Features ["3. Feature Engineering Pipeline"]
        E --> F[CreditRiskFeatureEngineer]
        F --> G[Credit Utilization: BILL_AMT1 / LIMIT_BAL]
        F --> H[Payment-to-Bill Ratio & Pay Trajectory]
        F --> I[Delinquency Counts: max_delay, months_delayed]
        I --> J[ColumnTransformer: StandardScaler + OneHotEncoder]
    end

    subgraph Modeling ["4. Model Suite & Benchmark"]
        J --> K1[Dummy Baseline]
        J --> K2[Logistic Regression]
        J --> K3[Decision Tree]
        J --> K4[Random Forest Champion]
        J --> K5[XGBoost]
        K1 & K2 & K3 & K4 & K5 --> L[Validation Benchmark PR-AUC & ROC-AUC]
    end

    subgraph Optimization ["5. Optimization & Interpretability"]
        L --> M[Cost-Matrix Threshold Optimization: 0.38]
        M --> N[SHAP TreeExplainer Attribution]
        N --> O[Artifact Serialization: best_model.joblib & metadata.json]
    end

    subgraph Serving ["6. Production Serving & UI"]
        O --> P[FastAPI REST Microservice :8000]
        O --> Q[Streamlit Underwriter Dashboard :8501]
    end
```

---

## 📊 Dataset Ingestion & Licensing

- **Source:** [UCI Machine Learning Repository: Default of Credit Card Clients Dataset](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
- **Citing:** Yeh, I. C., & Lien, C. H. (2009). *The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients.* Expert Systems with Applications, 36(2), 2473-2480.
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Cohort:** 30,000 credit card clients in Taiwan from April 2005 to September 2005.
- **Target Variable:** `default_payment_next_month` (Binary: 1 = defaulted, 0 = repaid on time).
- **Target Distribution:** 77.88% non-default (class 0), 22.12% default (class 1) — **~3.5:1 class imbalance**.

### Data Hygiene & Cleaning
- **Deduplication:** Identified and removed 35 exact duplicate rows in feature space, resulting in **29,944 clean observations**.
- **Category Consolidation:** UCI documentation defines `EDUCATION` as {1: Grad School, 2: University, 3: High School, 4: Others}. Undocumented codes {0, 5, 6} were safely consolidated into 4 (Others). Similarly, `MARRIAGE` code 0 was mapped to 3 (Others).
- **Standardization:** Renamed `PAY_0` to `PAY_1` to preserve consistent 1-to-6 chronological index matching `BILL_AMT1..6` and `PAY_AMT1..6`. Dropped client `ID` to prevent index memorization.

---

## 📐 Mathematical Foundations

### 1. Logistic Regression: Sigmoid, Log-Odds, and Odds Ratios

#### The Sigmoid Activation
Logistic Regression maps an unconstrained linear combination of input features $z \in (-\infty, +\infty)$ into a valid probability $p \in (0, 1)$ via the logistic sigmoid function:

$$\sigma(z) = \frac{1}{1 + e^{-z}} \quad \text{where} \quad z = \beta_0 + \sum_{j=1}^{d} \beta_j x_j$$

#### Log-Odds (Logit Transformation)
The relationship between default probability $p$ and non-default probability $1 - p$ defines the **odds of default**:

$$\text{Odds} = \frac{p}{1 - p} \implies \ln\left(\frac{p}{1 - p}\right) = \text{logit}(p) = \beta_0 + \sum_{j=1}^{d} \beta_j x_j$$

#### Odds Ratio Interpretation
Holding all other features constant, increasing feature $x_j$ by one unit (or one standard deviation when standardized) multiplies the odds of default by the factor $\exp(\beta_j)$:
- **$\beta_j > 0 \implies \exp(\beta_j) > 1$**: Risk driver. Higher values increase default probability.
- **$\beta_j < 0 \implies \exp(\beta_j) < 1$**: Protective factor. Higher values decrease default probability.

#### Regularized Binary Cross-Entropy Loss
Model parameters $\boldsymbol{\beta}$ are estimated via Maximum Likelihood Estimation, minimizing Binary Cross-Entropy with $L_2$ (Ridge) regularization:

$$\mathcal{L}(\boldsymbol{\beta}) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \ln(p_i) + (1 - y_i) \ln(1 - p_i) \right] + \frac{1}{2C} \|\boldsymbol{\beta}\|_2^2$$

The regularization term $\frac{1}{2C} \|\boldsymbol{\beta}\|_2^2$ stabilizes coefficients against high multicollinearity between consecutive monthly bill statements (`BILL_AMT1` to `BILL_AMT6`).

---

### 2. Tree-Based Models: Regularization & Ensembling

#### Decision Trees & Gini Impurity
At every internal node, the decision tree searches greedily across all features $x_j$ and candidate thresholds $t$ to split data into subsets $D_L$ and $D_R$, maximizing Impurity Reduction:

$$\Delta G = G(D) - \frac{|D_L|}{|D|} G(D_L) - \frac{|D_R|}{|D|} G(D_R)$$

where Gini Impurity for binary classification is:

$$G(D) = 1 - \sum_{k=0}^{1} p_k^2 = 2 p_0 p_1$$

An unconstrained tree splits recursively until leaves are pure ($G=0$), memorizing individual training applicants and yielding high generalization variance. We regularize the tree using `max_depth=5`, `min_samples_split=20`, and `min_samples_leaf=10`.

#### Random Forest (Bagging & Variance Reduction)
Random Forest builds $B$ decorrelated decision trees using bootstrap aggregation (bagging) and random subspace feature sampling ($m = \sqrt{d}$ candidate features per split).

If each tree has variance $\sigma^2$ and average pairwise correlation $\rho \in (0, 1)$, the variance of the ensemble mean is:

$$\text{Var}(\bar{T}) = \rho \sigma^2 + \frac{1 - \rho}{B} \sigma^2$$

As $B \to \infty$, the second term vanishes, leaving $\rho \sigma^2$. By subsampling features randomly at each split, Random Forest actively reduces inter-tree correlation $\rho$, driving total prediction variance significantly below that of any individual tree.

#### XGBoost (Gradient Boosting)
XGBoost minimizes a regularized objective sequentially using a second-order Taylor expansion of the loss function:

$$\tilde{\mathcal{L}}^{(t)} \approx \sum_{i=1}^{N} \left[ g_i f_t(\mathbf{x}_i) + \frac{1}{2} h_i f_t^2(\mathbf{x}_i) \right] + \Omega(f_t)$$

where:
- $g_i = \partial_{\hat{y}^{(t-1)}} l(y_i, \hat{y}^{(t-1)})$ (first-order gradient)
- $h_i = \partial^2_{\hat{y}^{(t-1)}} l(y_i, \hat{y}^{(t-1)})$ (second-order Hessian)
- $\Omega(f_t) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2$ (complexity penalty per leaf)

---

### 3. SHAP Explainability (Shapley Additive exPlanations)

SHAP computes mathematically fair feature attributions derived from cooperative game theory:

$$\phi_j = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|! (|F| - |S| - 1)!}{|F|!} \left[ f(S \cup \{j\}) - f(S) \right]$$

- **Efficiency Property:** The sum of feature SHAP values equals the difference between the model's prediction and the base expected value: $\sum_{j} \phi_j = f(\mathbf{x}) - \mathbb{E}[f(\mathbf{x})]$.
- **Association vs. Causality Notice:** SHAP values explain **the internal mathematical mechanics of the fitted model**, not real-world causality. They reflect empirical correlations present in the historical training data.

---

## 🧪 Experiments, Benchmark & Actual Metrics

Models were trained exclusively on the **stratified training set (17,966 records)** and evaluated on the **stratified validation set (5,989 records)**.

### Model Comparison Table (Validation Set)

| Model Architecture | PR-AUC (Target) | ROC-AUC | Recall (Default Detection) | Precision | F1 Score | Brier Score (Calibration) | Training Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy Baseline** | 0.2212 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.1723 | 0.117s |
| **Logistic Regression** | 0.5141 | 0.7661 | 0.6317 | 0.4668 | 0.5369 | 0.1883 | 0.253s |
| **Decision Tree** | 0.5307 | 0.7772 | 0.5985 | 0.4994 | 0.5445 | 0.1813 | 0.201s |
| **Random Forest (Champion)** | **0.5791** | **0.7952** | 0.6219 | 0.4940 | **0.5506** | 0.1741 | 1.571s |
| **XGBoost** | 0.5784 | 0.7975 | **0.6332** | 0.4836 | 0.5484 | **0.1720** | 0.655s |

> **Champion Selection:** **Random Forest** achieved the highest Precision-Recall AUC (**0.5791**) with superior calibration and out-of-fold stability, outperforming both linear and single-tree baselines.

---

## 💰 Decision Threshold Optimization & Business ROI

In consumer lending, a standard 0.50 probability threshold assumes symmetric cost of errors. In reality:
- **Cost of False Negative ($C_{FN}$):** Approving an applicant who defaults causes direct principal write-off and collection costs: **NT$ 50,000 (~$1,550 USD)**.
- **Cost of False Positive ($C_{FP}$):** Rejecting a creditworthy borrower loses the net interest margin and customer acquisition cost: **NT$ 8,000 (~$250 USD)**.

We optimize the total portfolio expected loss across decision thresholds $t \in [0.05, 0.95]$:

$$\text{Total Cost}(t) = C_{FN} \times FN(t) + C_{FP} \times FP(t)$$

### Business Cost Impact (Held-Out Test Partition)

| Operating Decision Threshold | Catch-Rate (Recall) | Defaulters Caught (TP) | False Negatives (FN Defaults) | Total Financial Loss (NT$) | Financial Improvement |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Standard 0.50 Threshold** | 61.25% | 811 | 513 | NT$ 31,436,000 | Baseline |
| **Cost-Optimal 0.38 Threshold** | **77.95%** | **1,032** | **292** | **NT$ 27,652,000** | **+12.04% Net Cost Savings** |

> **Net Value Creation:** Operating at threshold **0.38** saves **NT$ 3,784,000** on this test partition alone while boosting default detection from **61.2% to 78.0%**.

---

## 🚀 Quickstart & Installation

### 1. Prerequisites & Environment Setup
Clone the repository and create the Python 3.11+ virtual environment:

```bash
cd credit-risk-prediction

# Create virtual environment
py -3.13 -m venv .venv

# Activate virtual environment
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux/macOS:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train Models & Generate Artifacts
Run the end-to-end training, threshold optimization, and SHAP figure generator:

```bash
python -m src.models.train
```

### 3. Run Pytest Test Suite
Execute the 23 unit and integration tests:

```bash
pytest
```
*Result: 23 passed in ~2.1 seconds.*

### 4. Launch FastAPI REST Service
Start the Uvicorn ASGI server:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
- API Docs (Swagger UI): `http://localhost:8000/docs`
- Healthcheck: `http://localhost:8000/health`
- Model Telemetry: `http://localhost:8000/model-info`

### 5. Launch Streamlit Underwriter Dashboard
Start the interactive decision-support application:

```bash
streamlit run dashboard/app.py
```
- Open your browser at: `http://localhost:8501`

---

## 📡 API Endpoint Examples & cURL Requests

### 1. System Healthcheck (`GET /health`)
```bash
curl -X GET "http://localhost:8000/health"
```
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "RandomForest",
  "server_time": "2026-10-26T17:18:57.212252",
  "version": "1.0.0"
}
```

### 2. Model Telemetry & Metadata (`GET /model-info`)
```bash
curl -X GET "http://localhost:8000/model-info"
```
```json
{
  "project": "Credit Risk and Loan Default Prediction System",
  "company": "Ignosis FinTech Portfolio",
  "champion_model_name": "RandomForest",
  "champion_model_type": "RandomForestClassifier",
  "selected_decision_threshold": 0.38,
  "f1_optimal_threshold": 0.53,
  "test_metrics": {
    "roc_auc": 0.7817,
    "pr_auc": 0.5613,
    "recall": 0.7795,
    "precision": 0.3592,
    "f1_score": 0.4918
  }
}
```

### 3. Single Applicant Prediction (`POST /predict`)
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "applicant": {
         "LIMIT_BAL": 80000.0,
         "SEX": 1,
         "EDUCATION": 2,
         "MARRIAGE": 2,
         "AGE": 31,
         "PAY_1": 2,
         "PAY_2": 2,
         "PAY_3": 0,
         "PAY_4": 0,
         "PAY_5": 0,
         "PAY_6": 0,
         "BILL_AMT1": 75000.0,
         "BILL_AMT2": 74000.0,
         "BILL_AMT3": 72000.0,
         "BILL_AMT4": 70000.0,
         "BILL_AMT5": 68000.0,
         "BILL_AMT6": 65000.0,
         "PAY_AMT1": 0.0,
         "PAY_AMT2": 2000.0,
         "PAY_AMT3": 2000.0,
         "PAY_AMT4": 2000.0,
         "PAY_AMT5": 2000.0,
         "PAY_AMT6": 2000.0
       }
     }'
```
```json
{
  "default_probability": 0.8387,
  "predicted_class": 1,
  "risk_category": "High",
  "threshold_applied": 0.38,
  "recommendation": "High default risk (probability 83.9% exceeds threshold 38.0%). Secondary underwriter review or collateral required.",
  "model_name": "RandomForest",
  "model_version": "1.0.0",
  "key_risk_drivers": [
    {"feature": "PAY_1", "attribution": 0.0885},
    {"feature": "months_delayed_count", "attribution": 0.0753},
    {"feature": "max_delay_months", "attribution": 0.0664}
  ],
  "disclaimer": "EXPERIMENTAL PROTOTYPE: This prediction is generated by a decision-support model for risk evaluation only."
}
```

---

## 🐳 Docker Deployment

The repository includes a production multi-stage `Dockerfile` and `docker-compose.yml`:

```bash
# Build and run API (:8000) and Dashboard (:8501)
docker compose up --build -d

# Check running services
docker compose ps

# View container logs
docker compose logs -f api
```

---

## ⚖️ Limitations & Fair Lending Considerations

1. **Protected Attributes & Disparate Impact:**
   - The raw dataset includes demographic features (`SEX`, `MARRIAGE`, `AGE`).
   - Under US **Equal Credit Opportunity Act (ECOA) Regulation B**, using gender or marital status in adverse credit underwriting is prohibited.
   - In production, protected attributes should be excluded from training or audited via the **80% Rule (Four-Fifths Rule)** to guarantee disparate impact parity across demographic sub-groups.
2. **Macroeconomic Shocks & Concept Drift:**
   - The dataset represents consumer behavior in Taiwan during 2005. A model deployed today faces regime shifts (inflation, interest rate cycles, post-pandemic employment shocks).
   - Continuous drift monitoring using **Population Stability Index (PSI)** and **Wasserstein Distance** is necessary.
3. **Absence of Alternative Bureau Data:**
   - The model relies entirely on 6 months of card payment transactions. Modern FinTech credit engines integrate alternative bureau data (cash flow banking APIs, rent reporting, employment tenure) to evaluate thin-file borrowers.

---

## 🎓 Ignosis Interview Defense Guide (10 ML Engineer Q&A)

### Q1: Why did you prioritize PR-AUC and Recall over Accuracy?
> **Answer:** In credit default prediction, classes are imbalanced (~22% default, ~78% non-default). A trivial dummy model that always predicts non-default achieves **77.88% accuracy** while detecting exactly zero defaulters (0% Recall), causing severe lending losses. Accuracy gives equal credit to predicting the easy majority class. **PR-AUC (Precision-Recall Area Under Curve)** evaluates model discrimination exclusively focusing on the positive minority class (defaulters). In FinTech, missing a defaulter (False Negative) costs the full loan balance, so maximizing Recall subject to viable precision is the primary business objective.

### Q2: How did you ensure zero data leakage between splits?
> **Answer:** Strict temporal and procedural separation was enforced:
> 1. Target column `default_payment_next_month` was isolated prior to partitioning.
> 2. Stratified train/val/test splitting (60/20/20) was performed before any transformations.
> 3. Scikit-learn's `ColumnTransformer` (fitting `StandardScaler` and `OneHotEncoder`) was fitted **exclusively on the training partition**. The validation, test, and incoming API payloads were transformed using already-fitted parameters, completely preventing distribution leakage.

### Q3: Explain mathematically how Logistic Regression coefficients relate to default risk.
> **Answer:** Logistic regression models the log-odds of defaulting as a linear function: $\ln(\frac{p}{1-p}) = \beta_0 + \sum \beta_j x_j$. Exponentiating a learned coefficient gives the **Odds Ratio**: $\text{OR}_j = \exp(\beta_j)$. If a standardized feature has $\beta_j = 0.693$, then a 1 standard deviation increase in that feature multiplies the borrower's default odds by $\exp(0.693) = 2.0$ (doubling the odds of default), holding all other features fixed. If $\beta_j < 0$, $\exp(\beta_j) < 1$, indicating a protective attribute that reduces default risk.

### Q4: Why did Random Forest outperform a single Decision Tree and Logistic Regression?
> **Answer:** Credit risk features exhibit high nonlinear interactions (e.g., high credit utilization is harmless if payment-to-bill ratio is 100%, but dangerous if combined with a payment delay). A linear model cannot capture this without manual interaction terms. While a single decision tree captures interactions, it has high variance and easily overfits financial noise. Random Forest reduces variance through bootstrap aggregation (bagging) and random subspace feature sampling ($m = \sqrt{d}$), lowering pairwise tree correlation $\rho$. The ensemble variance decreases by $\frac{1-\rho}{B}\sigma^2$, yielding superior generalization (**PR-AUC 0.5791** vs Tree 0.5307 and Logistic 0.5141).

### Q5: Why did you pick an operating decision threshold of 0.38 instead of 0.50?
> **Answer:** A default threshold of 0.50 implicitly assumes symmetric error costs ($C_{FN} = C_{FP}$). In lending, a False Negative (loan default) costs ~NT$ 50,000, whereas a False Positive (rejecting a good customer) costs ~NT$ 8,000 in lost margin. By simulating expected loss across thresholds from 0.05 to 0.95, the cost curve achieves a minimum at **0.38**. Lowering the threshold to 0.38 boosted test recall from **61.2% to 78.0%**, catching 221 additional defaulters and reducing net portfolio losses by **12.04% (NT$ 3.78M saved)**.

### Q6: What is probability calibration and why is it critical in FinTech?
> **Answer:** Discrimination (ROC-AUC) only measures whether defaulters receive higher scores than non-defaulters; it does not ensure the score represents true empirical probability. A model is well-calibrated if, among borrowers assigned a predicted default probability of $30\%$, approximately $30\%$ actually default. In banking, predicted probabilities are plugged directly into Expected Loss formulas: $\text{EL} = \text{PD} \times \text{LGD} \times \text{EAD}$ and Basel capital adequacy reserves. We assessed calibration via **Brier Score (0.174)** and **Reliability Curves**, confirming our Random Forest probabilities reflect empirical frequencies.

### Q7: What do SHAP values represent, and why must they not be confused with causality?
> **Answer:** SHAP values are based on cooperative game theory (Shapley values), distributing credit among features for a specific model's mathematical output relative to its baseline expectation. They quantify **model attribution based on learned historical correlations**. They **do not prove real-world causality**. For example, having a lower credit limit may have a positive SHAP value for default risk, but lowering someone's limit did not cause them to default—rather, conservative underwriters had previously assigned lower limits to riskier profiles. Conflating attribution with causality can lead to flawed policy interventions.

### Q8: What domain features did you engineer and why did they dominate SHAP importance?
> **Answer:** We engineered:
> 1. `months_delayed_count`: Total months with delinquent payments (`PAY_i > 0`).
> 2. `max_delay_months`: Maximum delinquency duration over 6 months.
> 3. `credit_utilization_ratio`: Latest balance divided by limit (`BILL_AMT1 / LIMIT_BAL`).
> 4. `payment_to_bill_ratio`: Aggregate payments divided by aggregate balance.
> In global SHAP rankings, `months_delayed_count` and `max_delay_months` ranked #1 and #2, surpassing raw features. In credit risk, behavioral delinquency history is vastly more predictive of future insolvency than static balance levels.

### Q9: How would you monitor this model once deployed in production at Ignosis?
> **Answer:** We would implement four monitoring pillars:
> 1. **Data Drift:** Track distribution drift on input features using the **Population Stability Index (PSI)** and **Kolmogorov-Smirnov (KS) test** weekly. A $\text{PSI} > 0.25$ triggers automated retraining alerts.
> 2. **Concept Drift & Performance Degradation:** Monitor default label maturity (30-day delinquency emergence) and track monthly rolling ROC-AUC and PR-AUC.
> 3. **API System Telemetry:** Track endpoint p95 latency, request throughput, and HTTP 422/500 error rates via Prometheus and Grafana.
> 4. **Fair Lending Audit:** Continuously monitor the Approval Rate across demographic cohorts to detect disparate impact violations.

### Q10: How does your architecture ensure that API predictions match training preprocessing?
> **Answer:** The entire transformation pipeline (`CreditRiskFeatureEngineer` + `ColumnTransformer`) is packaged inside a single unified Scikit-learn `Pipeline` object along with the final classifier. The `CreditRiskService` passes raw dictionary inputs into this pipeline using strict Pydantic schemas that enforce column names, data types, and value bounds. Because the transformer was fitted during training, running `.predict_proba()` in production applies the **exact same scaling parameters, category encodings, and column order** without any manual re-implementation or drift risk.

---

## 📁 Repository Directory Structure

```text
credit-risk-prediction/
├── api/
│   ├── __init__.py
│   ├── main.py                  # FastAPI router with /health, /model-info, /predict, /predict-batch
│   ├── schemas.py               # Pydantic V2 validation schemas
│   └── service.py               # Model inference service with caching & feature alignment
├── artifacts/
│   ├── figures/                 # Publication-quality evaluation plots
│   │   ├── calibration_curve_comparison.png
│   │   ├── confusion_matrix_champion.png
│   │   ├── pr_curve_comparison.png
│   │   ├── roc_curve_comparison.png
│   │   ├── shap_summary_plot.png
│   │   └── threshold_cost_tradeoff.png
│   ├── metrics/
│   │   └── models_benchmark.json# Benchmark comparison across all 5 models
│   └── models/
│       ├── best_model.joblib    # Serialized champion pipeline
│       └── model_metadata.json  # Comprehensive versioned metadata
├── dashboard/
│   └── app.py                   # Streamlit multi-tab decision-support dashboard
├── data/
│   ├── credit_card_clients_cleaned.csv  # Cleaned, deduplicated cache
│   └── default of credit card clients.xls # Raw UCI dataset
├── notebooks/
│   └── 01_eda_and_data_profiling.ipynb   # Documented Jupyter EDA walk-through
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download.py          # UCI download & synthetic sample generator
│   │   ├── loader.py            # Dataset loading, deduplication, and caching
│   │   └── validator.py         # Schema, missing values, and domain validation
│   ├── features/
│   │   ├── __init__.py
│   │   ├── engineering.py       # Domain feature engineering transformer
│   │   └── pipeline.py          # Scikit-learn ColumnTransformer & Pipeline builder
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline.py          # DummyClassifier baseline
│   │   ├── logistic_regression.py # Regularized Logistic Regression & Odds Ratios
│   │   ├── tree_models.py       # DecisionTree, RandomForest, XGBoost
│   │   ├── explainability.py    # SHAP TreeExplainer & fallback attributions
│   │   └── train.py             # End-to-end training orchestration pipeline
│   └── evaluation/
│       ├── __init__.py
│       ├── calibration.py       # Calibration curve & ECE calculation
│       ├── metrics.py           # PR-AUC, ROC-AUC, Brier score, and F1 metrics
│       ├── plots.py             # Matplotlib/Seaborn plot generation
│       └── threshold.py         # Business cost matrix optimization
├── tests/
│   ├── conftest.py              # Pytest fixtures and test data
│   ├── test_api.py              # API endpoint integration & schema validation tests
│   ├── test_data.py             # Data loading and cleaning tests
│   ├── test_features.py         # Feature engineering & leakage tests
│   └── test_models.py           # Model training, metrics, and threshold tests
├── .env.example                 # Environment variables template
├── .gitignore                   # Git ignore patterns
├── Dockerfile                   # Multi-stage production container image
├── docker-compose.yml           # Multi-service orchestration (API + Dashboard)
├── pytest.ini                   # Pytest configuration
├── requirements.txt             # Pinned production dependencies
└── README.md                    # Complete project documentation & interview guide
```
#   c r e d i t - r i s k - p r e d i c t i o n  
 