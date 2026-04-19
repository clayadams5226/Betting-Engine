"""Logistic regression combining batter, pitcher, and environmental scores.

logit(P_HR) = beta_0 + beta_1 * Batter_Score + beta_2 * Pitcher_Score
            + beta_3 * ln(Env_Score) + beta_4 * (Batter_Score * Pitcher_Score)

P_HR = 1 / (1 + exp(-logit))
"""

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression

MODEL_DIR = Path(__file__).parent / "saved"

# Default coefficients (reasonable starting point before training)
# These produce sensible probabilities centered around the ~4% base rate
DEFAULT_COEFFICIENTS = {
    "beta_0": -3.2,    # intercept (log-odds at average matchup)
    "beta_1": 0.55,    # batter power score
    "beta_2": 0.40,    # pitcher vulnerability score
    "beta_3": 2.0,     # ln(env_score)
    "beta_4": 0.10,    # interaction term
}


class HRCombiner:
    """Combines batter, pitcher, and environmental scores into HR probability."""

    def __init__(self, coefficients: Optional[dict] = None):
        self.coefficients = coefficients or DEFAULT_COEFFICIENTS.copy()
        self._sklearn_model: Optional[LogisticRegression] = None

    def predict_proba(
        self,
        batter_score: float,
        pitcher_score: float,
        env_score: float,
    ) -> float:
        """Compute P(HR >= 1) for a batter in a game.

        Returns probability between 0 and 1.
        """
        b = self.coefficients
        ln_env = math.log(max(env_score, 0.01))
        interaction = batter_score * pitcher_score

        logit = (
            b["beta_0"]
            + b["beta_1"] * batter_score
            + b["beta_2"] * pitcher_score
            + b["beta_3"] * ln_env
            + b["beta_4"] * interaction
        )

        return _sigmoid(logit)

    def fit(
        self,
        batter_scores: list[float],
        pitcher_scores: list[float],
        env_scores: list[float],
        hr_outcomes: list[int],
    ) -> dict:
        """Train the logistic model on historical data.

        Args:
            batter_scores: PCA-derived batter scores.
            pitcher_scores: PCA-derived pitcher scores.
            env_scores: Environmental multipliers.
            hr_outcomes: Binary (1 = hit HR, 0 = no HR).

        Returns training metrics.
        """
        X = self._build_features(batter_scores, pitcher_scores, env_scores)
        y = np.array(hr_outcomes)

        self._sklearn_model = LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
        )
        self._sklearn_model.fit(X, y)

        # Extract coefficients
        coefs = self._sklearn_model.coef_[0]
        self.coefficients = {
            "beta_0": float(self._sklearn_model.intercept_[0]),
            "beta_1": float(coefs[0]),
            "beta_2": float(coefs[1]),
            "beta_3": float(coefs[2]),
            "beta_4": float(coefs[3]),
        }

        # Training metrics
        preds = self._sklearn_model.predict_proba(X)[:, 1]
        return {
            "coefficients": self.coefficients,
            "mean_predicted": float(preds.mean()),
            "actual_hr_rate": float(y.mean()),
            "n_samples": len(y),
            "n_positive": int(y.sum()),
        }

    def save(self, path: Optional[Path] = None) -> None:
        if path is None:
            path = MODEL_DIR
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "combiner_model.json", "w") as f:
            json.dump(self.coefficients, f, indent=2)

    def load(self, path: Optional[Path] = None) -> None:
        if path is None:
            path = MODEL_DIR
        with open(path / "combiner_model.json") as f:
            self.coefficients = json.load(f)

    @staticmethod
    def _build_features(
        batter_scores: list[float],
        pitcher_scores: list[float],
        env_scores: list[float],
    ) -> np.ndarray:
        """Build feature matrix with interaction term."""
        n = len(batter_scores)
        X = np.zeros((n, 4))
        for i in range(n):
            X[i, 0] = batter_scores[i]
            X[i, 1] = pitcher_scores[i]
            X[i, 2] = math.log(max(env_scores[i], 0.01))
            X[i, 3] = batter_scores[i] * pitcher_scores[i]
        return X


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        ex = math.exp(x)
        return ex / (1.0 + ex)
