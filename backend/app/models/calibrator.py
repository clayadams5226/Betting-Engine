"""Isotonic regression calibration for post-processing predicted probabilities.

Corrects systematic bias in probability tails — ensures that when we predict
8% HR probability, it actually happens ~8% of the time.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression

MODEL_DIR = Path(__file__).parent / "saved"


class HRCalibrator:
    """Isotonic regression calibrator for HR probabilities."""

    def __init__(self):
        self._model: Optional[IsotonicRegression] = None
        self._is_fitted = False

    def fit(self, predicted: list[float], actual: list[int]) -> dict:
        """Fit calibration model on held-out data.

        Args:
            predicted: Raw model probabilities.
            actual: Binary outcomes (1 = HR, 0 = no HR).

        Returns calibration metrics.
        """
        pred_arr = np.array(predicted)
        actual_arr = np.array(actual)

        self._model = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            out_of_bounds="clip",
        )
        self._model.fit(pred_arr, actual_arr)
        self._is_fitted = True

        calibrated = self._model.predict(pred_arr)

        return {
            "mean_predicted_raw": float(pred_arr.mean()),
            "mean_predicted_calibrated": float(calibrated.mean()),
            "actual_rate": float(actual_arr.mean()),
            "n_samples": len(predicted),
            "calibration_bins": _compute_calibration_bins(pred_arr, actual_arr, n_bins=10),
        }

    def calibrate(self, probability: float) -> float:
        """Calibrate a single predicted probability.

        If not fitted, returns the raw probability unchanged.
        """
        if not self._is_fitted or self._model is None:
            return probability

        result = self._model.predict(np.array([probability]))[0]
        return float(np.clip(result, 0.001, 0.999))

    def calibrate_batch(self, probabilities: list[float]) -> list[float]:
        """Calibrate a batch of probabilities."""
        if not self._is_fitted or self._model is None:
            return probabilities

        arr = np.array(probabilities)
        calibrated = self._model.predict(arr)
        return np.clip(calibrated, 0.001, 0.999).tolist()

    def save(self, path: Optional[Path] = None) -> None:
        if path is None:
            path = MODEL_DIR
        path.mkdir(parents=True, exist_ok=True)

        if not self._is_fitted or self._model is None:
            return

        data = {
            "X_thresholds": self._model.X_thresholds_.tolist(),
            "y_thresholds": self._model.y_thresholds_.tolist(),
            "is_fitted": True,
        }
        with open(path / "calibrator_model.json", "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: Optional[Path] = None) -> None:
        if path is None:
            path = MODEL_DIR

        filepath = path / "calibrator_model.json"
        if not filepath.exists():
            return

        with open(filepath) as f:
            data = json.load(f)

        if data.get("is_fitted"):
            # Rebuild by fitting on the saved thresholds directly
            X_thresh = np.array(data["X_thresholds"])
            y_thresh = np.array(data["y_thresholds"])
            self._model = IsotonicRegression(
                y_min=0.0, y_max=1.0, out_of_bounds="clip",
            )
            self._model.fit(X_thresh, y_thresh)
            self._is_fitted = True


def _compute_calibration_bins(
    predicted: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
) -> list[dict]:
    """Compute calibration curve bins for reliability diagrams."""
    bins = []
    edges = np.linspace(0, predicted.max() + 0.001, n_bins + 1)

    for i in range(n_bins):
        mask = (predicted >= edges[i]) & (predicted < edges[i + 1])
        if mask.sum() == 0:
            continue
        bins.append({
            "bin_start": float(edges[i]),
            "bin_end": float(edges[i + 1]),
            "mean_predicted": float(predicted[mask].mean()),
            "mean_actual": float(actual[mask].mean()),
            "count": int(mask.sum()),
        })

    return bins
