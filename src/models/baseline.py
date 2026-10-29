"""Baseline model implementations for credit risk default prediction."""

from __future__ import annotations

from sklearn.dummy import DummyClassifier


def build_dummy_baseline(strategy: str = "prior", random_state: int = 42) -> DummyClassifier:
    """Create a DummyClassifier baseline.

    In imbalanced classification (~78% non-default, ~22% default), a naive model
    predicting non-default for every applicant achieves 78% accuracy while detecting
    0% of defaulters (0 Recall).

    A Dummy baseline proves whether trained models provide genuine discriminative lift.

    Args:
        strategy: Baseline strategy:
            - 'prior': Always predicts class probabilities equal to the training class distribution.
            - 'most_frequent': Always predicts the majority class (0).
            - 'stratified': Generates predictions by respecting the training class distribution.
        random_state: Fixed random seed for reproducibility.

    Returns:
        Configured DummyClassifier estimator.
    """
    return DummyClassifier(strategy=strategy, random_state=random_state)
