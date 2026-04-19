"""PCA decorrelation for batter and pitcher stat groups.

Separate PCA per pillar:
- Batter: 6 stats -> 2 principal components (power + launch approach)
- Pitcher: 6 stats -> 2 principal components (HR vulnerability + expected contact quality)
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA

BATTER_STAT_KEYS = [
    "barrel_pct", "hard_hit_pct", "avg_ev",
    "max_ev", "avg_launch_angle", "sweet_spot_pct",
]

PITCHER_STAT_KEYS = [
    "hr_per_9", "hr_fb_pct", "barrel_pct_allowed",
    "hard_hit_pct_allowed", "xslg_against", "xwoba_against",
]

MODEL_DIR = Path(__file__).parent / "saved"


class HRPcaModel:
    """Separate PCA models for batter and pitcher stat decorrelation."""

    def __init__(self, n_batter_components: int = 2, n_pitcher_components: int = 2):
        self.n_batter_components = n_batter_components
        self.n_pitcher_components = n_pitcher_components
        self.batter_pca: Optional[PCA] = None
        self.pitcher_pca: Optional[PCA] = None

    def fit_batter(self, stats_list: list[dict]) -> dict:
        """Fit batter PCA on a population of normalized batter stats.

        Args:
            stats_list: List of z-score normalized batter stat dicts.

        Returns dict with explained_variance_ratio and component loadings.
        """
        X = self._stats_to_matrix(stats_list, BATTER_STAT_KEYS)
        if X.shape[0] < self.n_batter_components:
            self.batter_pca = PCA(n_components=min(X.shape[0], 1))
        else:
            self.batter_pca = PCA(n_components=self.n_batter_components)

        self.batter_pca.fit(X)

        return {
            "explained_variance_ratio": self.batter_pca.explained_variance_ratio_.tolist(),
            "cumulative_variance": np.cumsum(self.batter_pca.explained_variance_ratio_).tolist(),
            "components": self.batter_pca.components_.tolist(),
            "feature_names": BATTER_STAT_KEYS,
        }

    def fit_pitcher(self, stats_list: list[dict]) -> dict:
        """Fit pitcher PCA on a population of normalized pitcher stats."""
        X = self._stats_to_matrix(stats_list, PITCHER_STAT_KEYS)
        if X.shape[0] < self.n_pitcher_components:
            self.pitcher_pca = PCA(n_components=min(X.shape[0], 1))
        else:
            self.pitcher_pca = PCA(n_components=self.n_pitcher_components)

        self.pitcher_pca.fit(X)

        return {
            "explained_variance_ratio": self.pitcher_pca.explained_variance_ratio_.tolist(),
            "cumulative_variance": np.cumsum(self.pitcher_pca.explained_variance_ratio_).tolist(),
            "components": self.pitcher_pca.components_.tolist(),
            "feature_names": PITCHER_STAT_KEYS,
        }

    def transform_batter(self, stats: dict) -> dict:
        """Transform a single batter's normalized stats into PCA space.

        Returns variance-weighted composite score and individual PC values.
        """
        if self.batter_pca is None:
            return {"batter_score": 0.0, "pc_values": []}

        x = np.array([[stats.get(k, 0.0) for k in BATTER_STAT_KEYS]])
        pcs = self.batter_pca.transform(x)[0]

        # Variance-weighted composite
        ratios = self.batter_pca.explained_variance_ratio_
        score = float(np.dot(pcs, ratios) / ratios.sum())

        return {
            "batter_score": score,
            "pc_values": pcs.tolist(),
            "variance_ratios": ratios.tolist(),
        }

    def transform_pitcher(self, stats: dict) -> dict:
        """Transform a single pitcher's normalized stats into PCA space."""
        if self.pitcher_pca is None:
            return {"pitcher_score": 0.0, "pc_values": []}

        x = np.array([[stats.get(k, 0.0) for k in PITCHER_STAT_KEYS]])
        pcs = self.pitcher_pca.transform(x)[0]

        ratios = self.pitcher_pca.explained_variance_ratio_
        score = float(np.dot(pcs, ratios) / ratios.sum())

        return {
            "pitcher_score": score,
            "pc_values": pcs.tolist(),
            "variance_ratios": ratios.tolist(),
        }

    def save(self, path: Optional[Path] = None) -> None:
        """Save PCA model parameters to JSON."""
        if path is None:
            path = MODEL_DIR
        path.mkdir(parents=True, exist_ok=True)

        data = {}
        if self.batter_pca is not None:
            data["batter"] = {
                "n_components": self.n_batter_components,
                "components": self.batter_pca.components_.tolist(),
                "mean": self.batter_pca.mean_.tolist(),
                "explained_variance": self.batter_pca.explained_variance_.tolist(),
                "explained_variance_ratio": self.batter_pca.explained_variance_ratio_.tolist(),
            }
        if self.pitcher_pca is not None:
            data["pitcher"] = {
                "n_components": self.n_pitcher_components,
                "components": self.pitcher_pca.components_.tolist(),
                "mean": self.pitcher_pca.mean_.tolist(),
                "explained_variance": self.pitcher_pca.explained_variance_.tolist(),
                "explained_variance_ratio": self.pitcher_pca.explained_variance_ratio_.tolist(),
            }

        with open(path / "pca_model.json", "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: Optional[Path] = None) -> None:
        """Load PCA model parameters from JSON."""
        if path is None:
            path = MODEL_DIR

        with open(path / "pca_model.json") as f:
            data = json.load(f)

        if "batter" in data:
            b = data["batter"]
            self.n_batter_components = b["n_components"]
            self.batter_pca = PCA(n_components=self.n_batter_components)
            self.batter_pca.components_ = np.array(b["components"])
            self.batter_pca.mean_ = np.array(b["mean"])
            self.batter_pca.explained_variance_ = np.array(b["explained_variance"])
            self.batter_pca.explained_variance_ratio_ = np.array(b["explained_variance_ratio"])
            self.batter_pca.n_features_in_ = len(BATTER_STAT_KEYS)

        if "pitcher" in data:
            p = data["pitcher"]
            self.n_pitcher_components = p["n_components"]
            self.pitcher_pca = PCA(n_components=self.n_pitcher_components)
            self.pitcher_pca.components_ = np.array(p["components"])
            self.pitcher_pca.mean_ = np.array(p["mean"])
            self.pitcher_pca.explained_variance_ = np.array(p["explained_variance"])
            self.pitcher_pca.explained_variance_ratio_ = np.array(p["explained_variance_ratio"])
            self.pitcher_pca.n_features_in_ = len(PITCHER_STAT_KEYS)

    @staticmethod
    def _stats_to_matrix(stats_list: list[dict], keys: list[str]) -> np.ndarray:
        """Convert list of stat dicts to numpy matrix."""
        rows = []
        for s in stats_list:
            rows.append([s.get(k, 0.0) for k in keys])
        return np.array(rows, dtype=np.float64)
