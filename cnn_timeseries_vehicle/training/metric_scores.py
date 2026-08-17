"""Regression metrics, computed in the original unit of the target.

Names match scikit-learn so they can be looked up in its docs.
"""

from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Reported for every split, in this order.
REGRESSION_METRIC_NAMES = (
    "r2",
    "mean_absolute_error",
    "root_mean_squared_error",
    "mean_absolute_percentage_error",
)

# Relative error is meaningless where the true value is ~0.
MAPE_MIN_DENOMINATOR = 1e-6


def flatten_pairs(y_true: np.ndarray, y_pred: np.ndarray):
    """Flatten multi-step predictions so every (step, sample) counts once."""
    true_flat = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred_flat = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if true_flat.shape != pred_flat.shape:
        raise ValueError(f"shape mismatch: {true_flat.shape} vs {pred_flat.shape}")
    return true_flat, pred_flat


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error, skipping near-zero true values."""
    usable = np.abs(y_true) > MAPE_MIN_DENOMINATOR
    if not usable.any():
        return float("nan")
    errors = np.abs((y_true[usable] - y_pred[usable]) / y_true[usable])
    return float(np.mean(errors) * 100.0)


def calculate_regression_scores(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """All reported metrics for one split."""
    true_flat, pred_flat = flatten_pairs(y_true, y_pred)
    return {
        "r2": float(r2_score(true_flat, pred_flat)),
        "mean_absolute_error": float(mean_absolute_error(true_flat, pred_flat)),
        "root_mean_squared_error": float(
            np.sqrt(mean_squared_error(true_flat, pred_flat))
        ),
        "mean_absolute_percentage_error": calculate_mape(true_flat, pred_flat),
    }


def calculate_scores_per_step(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Scores for each forecast step; accuracy usually decays with distance."""
    true_2d = np.atleast_2d(np.asarray(y_true, dtype=np.float64))
    pred_2d = np.atleast_2d(np.asarray(y_pred, dtype=np.float64))
    return {
        f"step_{step + 1}": calculate_regression_scores(
            true_2d[:, step], pred_2d[:, step]
        )
        for step in range(true_2d.shape[1])
    }


def format_scores(scores: Dict[str, float]) -> str:
    """One-line rendering, e.g. `r2=0.981 mean_absolute_error=0.412 ...`."""
    return " ".join(
        f"{name}={scores[name]:.4f}"
        for name in REGRESSION_METRIC_NAMES
        if name in scores
    )
