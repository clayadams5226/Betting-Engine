"""Backtesting and calibration evaluation for the HR prediction model."""

from datetime import date, timedelta

import numpy as np

from ..models.calibrator import HRCalibrator
from ..models.combiner import HRCombiner
from ..models.pca import HRPcaModel


def evaluate_calibration(
    predicted: list[float],
    actual: list[int],
    n_bins: int = 10,
) -> dict:
    """Evaluate model calibration with reliability diagram data.

    Args:
        predicted: Predicted HR probabilities.
        actual: Binary outcomes.
        n_bins: Number of bins for calibration curve.

    Returns calibration metrics and bin data.
    """
    pred_arr = np.array(predicted)
    actual_arr = np.array(actual)

    # Overall metrics
    brier_score = float(np.mean((pred_arr - actual_arr) ** 2))
    log_loss = _log_loss(pred_arr, actual_arr)

    # Calibration bins
    bins = []
    edges = np.linspace(0, max(pred_arr.max(), 0.001) + 0.001, n_bins + 1)

    for i in range(n_bins):
        mask = (pred_arr >= edges[i]) & (pred_arr < edges[i + 1])
        count = int(mask.sum())
        if count == 0:
            continue
        bins.append({
            "bin_start": round(float(edges[i]), 4),
            "bin_end": round(float(edges[i + 1]), 4),
            "mean_predicted": round(float(pred_arr[mask].mean()), 4),
            "mean_actual": round(float(actual_arr[mask].mean()), 4),
            "count": count,
        })

    # Expected Calibration Error
    ece = 0.0
    for b in bins:
        ece += b["count"] / len(pred_arr) * abs(b["mean_predicted"] - b["mean_actual"])

    return {
        "brier_score": round(brier_score, 6),
        "log_loss": round(log_loss, 6),
        "ece": round(ece, 6),
        "base_rate": round(float(actual_arr.mean()), 4),
        "mean_predicted": round(float(pred_arr.mean()), 4),
        "n_samples": len(predicted),
        "n_positive": int(actual_arr.sum()),
        "bins": bins,
    }


def evaluate_discrimination(
    predicted: list[float],
    actual: list[int],
) -> dict:
    """Evaluate model's ability to rank batters (discrimination, not calibration).

    Returns AUC-ROC approximation and top-N precision.
    """
    pred_arr = np.array(predicted)
    actual_arr = np.array(actual)

    # Sort by predicted probability descending
    sorted_idx = np.argsort(-pred_arr)
    sorted_actual = actual_arr[sorted_idx]

    # Top-N precision (how many of our top picks actually hit HR?)
    top_n_results = {}
    for n in [10, 25, 50, 100]:
        if n <= len(sorted_actual):
            precision = float(sorted_actual[:n].mean())
            top_n_results[f"top_{n}_precision"] = round(precision, 4)

    # Simple AUC approximation using Mann-Whitney U statistic
    positive_preds = pred_arr[actual_arr == 1]
    negative_preds = pred_arr[actual_arr == 0]

    if len(positive_preds) > 0 and len(negative_preds) > 0:
        # Sample for efficiency if large
        if len(negative_preds) > 10000:
            neg_sample = np.random.choice(negative_preds, 10000, replace=False)
        else:
            neg_sample = negative_preds

        u_stat = 0
        for p in positive_preds:
            u_stat += (p > neg_sample).sum() + 0.5 * (p == neg_sample).sum()
        auc = u_stat / (len(positive_preds) * len(neg_sample))
    else:
        auc = 0.5

    return {
        "auc_roc": round(float(auc), 4),
        **top_n_results,
    }


def _log_loss(predicted: np.ndarray, actual: np.ndarray, eps: float = 1e-7) -> float:
    """Compute binary log loss."""
    p = np.clip(predicted, eps, 1 - eps)
    return float(-np.mean(actual * np.log(p) + (1 - actual) * np.log(1 - p)))
