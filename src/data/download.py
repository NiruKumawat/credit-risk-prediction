"""Data download and acquisition utilities for the Credit Risk Prediction System.

Dataset: Default of Credit Card Clients (UCI Machine Learning Repository)
URL: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
License: Creative Commons Attribution 4.0 International (CC BY 4.0)
Original Citation:
    Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques
    for the predictive accuracy of probability of default of credit card clients.
    Expert Systems with Applications, 36(2), 2473-2480.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

UCI_DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
)
RAW_DATA_FILENAME = "default of credit card clients.xls"


def get_data_dir() -> Path:
    """Return the absolute path to the data directory."""
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def check_existing_data(data_dir: Optional[Path] = None) -> Optional[Path]:
    """Check if the raw or processed dataset already exists locally."""
    data_dir = data_dir or get_data_dir()
    xls_path = data_dir / RAW_DATA_FILENAME
    csv_path = data_dir / "credit_card_clients.csv"

    if xls_path.exists():
        logger.info(f"Found existing raw dataset: {xls_path}")
        return xls_path
    if csv_path.exists():
        logger.info(f"Found existing processed CSV dataset: {csv_path}")
        return csv_path
    return None


def download_uci_dataset(data_dir: Optional[Path] = None) -> Path:
    """Download the official dataset from the UCI Machine Learning Repository."""
    data_dir = data_dir or get_data_dir()
    existing_file = check_existing_data(data_dir)
    if existing_file:
        logger.info(f"Dataset already exists at {existing_file}. Skipping download.")
        return existing_file

    zip_path = data_dir / "default_of_credit_card_clients.zip"
    logger.info(f"Downloading dataset from UCI repository: {UCI_DATASET_URL}")

    try:
        response = requests.get(UCI_DATASET_URL, stream=True, timeout=60)
        response.raise_for_status()

        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info("Extracting downloaded zip archive...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)

        if zip_path.exists():
            zip_path.unlink()

        target_file = data_dir / RAW_DATA_FILENAME
        if target_file.exists():
            logger.info(f"Successfully downloaded and extracted to {target_file}")
            return target_file
        else:
            raise FileNotFoundError(
                f"Expected file '{RAW_DATA_FILENAME}' was not found after unzipping."
            )
    except Exception as exc:
        logger.error(
            "Automatic download failed. You can manually download the file:\n"
            "1. Visit: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients\n"
            f"2. Place '{RAW_DATA_FILENAME}' inside the 'data/' folder."
        )
        raise exc


def generate_synthetic_test_sample(
    output_path: Optional[Path] = None, n_samples: int = 200, random_seed: int = 42
) -> Path:
    """Generate a clearly labeled synthetic sample dataset for CI/CD and unit testing.

    This synthetic sample is strictly used for unit tests and smoke tests when the full
    raw dataset is not available. It is NEVER substituted silently for model training.
    """
    rng = np.random.RandomState(random_seed)
    output_path = output_path or (get_data_dir() / "synthetic_sample_for_testing.csv")

    ids = np.arange(1, n_samples + 1)
    limit_bal = rng.choice([20000, 50000, 100000, 200000, 500000], size=n_samples)
    sex = rng.choice([1, 2], size=n_samples)
    education = rng.choice([1, 2, 3, 4], size=n_samples, p=[0.35, 0.45, 0.15, 0.05])
    marriage = rng.choice([1, 2, 3], size=n_samples, p=[0.45, 0.50, 0.05])
    age = rng.randint(21, 65, size=n_samples)

    data_dict = {
        "ID": ids,
        "LIMIT_BAL": limit_bal,
        "SEX": sex,
        "EDUCATION": education,
        "MARRIAGE": marriage,
        "AGE": age,
    }

    # Generate 6-month repayment status (-2 to 8)
    for i in [1, 2, 3, 4, 5, 6]:
        col_name = "PAY_0" if i == 1 else f"PAY_{i}"
        data_dict[col_name] = rng.choice([-1, 0, 1, 2], size=n_samples, p=[0.2, 0.6, 0.1, 0.1])

    # Generate 6-month bill statements and payments
    for i in range(1, 7):
        bills = np.maximum(0, rng.normal(loc=30000, scale=20000, size=n_samples).astype(int))
        payments = np.maximum(0, (bills * rng.uniform(0.1, 1.0, size=n_samples)).astype(int))
        data_dict[f"BILL_AMT{i}"] = bills
        data_dict[f"PAY_AMT{i}"] = payments

    # Target variable (default payment next month) ~22% default rate
    target_prob = 0.15 + 0.3 * (data_dict["PAY_0"] > 0)
    defaults = (rng.uniform(0, 1, size=n_samples) < target_prob).astype(int)
    data_dict["default payment next month"] = defaults

    df = pd.DataFrame(data_dict)
    df.to_csv(output_path, index=False)
    logger.info(
        f"Generated {n_samples} synthetic test records at: {output_path} "
        "(Clearly labeled for testing only)"
    )
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Acquire UCI Credit Card Dataset")
    parser.add_argument(
        "--download", action="store_true", help="Download official dataset from UCI"
    )
    parser.add_argument(
        "--synthetic", action="store_true", help="Generate synthetic test sample"
    )
    args = parser.parse_args()

    data_dir = get_data_dir()
    if args.synthetic:
        generate_synthetic_test_sample()
    elif args.download or not check_existing_data(data_dir):
        download_uci_dataset(data_dir)
    else:
        logger.info(f"Dataset already available at {check_existing_data(data_dir)}")
