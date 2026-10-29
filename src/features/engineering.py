"""Domain-specific feature engineering for Credit Risk Assessment.

Implements Scikit-learn BaseEstimator and TransformerMixin for seamless
pipeline integration and strict prevention of data leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CreditRiskFeatureEngineer(BaseEstimator, TransformerMixin):
    """Generates financial domain features from raw credit card client data.

    Features created:
    - credit_utilization_ratio: BILL_AMT1 / LIMIT_BAL
    - avg_bill_amt: Mean of BILL_AMT1 to BILL_AMT6
    - avg_pay_amt: Mean of PAY_AMT1 to PAY_AMT6
    - bill_to_limit_avg: avg_bill_amt / LIMIT_BAL
    - payment_to_bill_ratio: sum(PAY_AMT) / (sum(BILL_AMT) + 1.0)
    - max_delay_months: Maximum delinquency status across PAY_1 to PAY_6
    - months_delayed_count: Count of months where payment was delayed (PAY_i > 0)
    - pay_trend: Difference between latest delay and 6-month-ago delay (PAY_1 - PAY_6)
    """

    def __init__(self, clip_utilization: bool = True):
        self.clip_utilization = clip_utilization
        self.feature_names_: list[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        """Fit transformer and record output column names."""
        # Feature engineering is deterministic row-wise, but we inspect column names
        dummy_out = self._transform_df(X.head(2))
        self.feature_names_ = list(dummy_out.columns)
        return self

    def _transform_df(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # Bill and Pay column lists
        bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
        pay_cols = [f"PAY_AMT{i}" for i in range(1, 7)]
        status_cols = [f"PAY_{i}" for i in range(1, 7)]

        # 1. Credit utilization ratio (Latest bill / Limit)
        # Safe division handling limit <= 0
        limit_safe = np.maximum(df["LIMIT_BAL"].values, 1.0)
        utilization = df["BILL_AMT1"].values / limit_safe
        if self.clip_utilization:
            utilization = np.clip(utilization, -2.0, 5.0)
        df["credit_utilization_ratio"] = utilization

        # 2. Average bill amount & Average payment amount
        df["avg_bill_amt"] = df[bill_cols].mean(axis=1)
        df["avg_pay_amt"] = df[pay_cols].mean(axis=1)

        # 3. Average bill to limit ratio
        avg_bill_util = df["avg_bill_amt"].values / limit_safe
        if self.clip_utilization:
            avg_bill_util = np.clip(avg_bill_util, -2.0, 5.0)
        df["bill_to_limit_avg"] = avg_bill_util

        # 4. Total payment to total bill ratio (Liquidity / Payoff capacity)
        sum_bills = np.maximum(df[bill_cols].sum(axis=1).values, 0.0) + 1.0
        sum_pays = np.maximum(df[pay_cols].sum(axis=1).values, 0.0)
        payment_ratio = np.clip(sum_pays / sum_bills, 0.0, 10.0)
        df["payment_to_bill_ratio"] = payment_ratio

        # 5. Maximum delay months across past 6 months
        df["max_delay_months"] = df[status_cols].max(axis=1)

        # 6. Delinquency frequency (number of months with delay > 0)
        df["months_delayed_count"] = (df[status_cols] > 0).sum(axis=1)

        # 7. Payment status trajectory/trend (PAY_1 - PAY_6)
        # Positive value indicates deteriorating delinquency
        df["pay_trend"] = df["PAY_1"] - df["PAY_6"]

        return df

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply domain feature engineering to dataset."""
        return self._transform_df(X)

    def get_feature_names_out(self, input_features=None) -> list[str]:
        """Return generated output feature names."""
        return list(self.feature_names_)
