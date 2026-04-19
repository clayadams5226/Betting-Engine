"""Z-score normalization against league population per handedness split."""

import numpy as np
import pandas as pd

from .loader import compute_batter_stats, compute_pitcher_stats

# Stat keys for each type
BATTER_STAT_KEYS = [
    "barrel_pct", "hard_hit_pct", "avg_ev",
    "max_ev", "avg_launch_angle", "sweet_spot_pct",
]

PITCHER_STAT_KEYS = [
    "hr_per_9", "hr_fb_pct", "barrel_pct_allowed",
    "hard_hit_pct_allowed", "xslg_against", "xwoba_against",
]


class LeagueNormalizer:
    """Computes and stores league-wide z-score parameters."""

    def __init__(self):
        self.batter_params: dict[str, dict] = {}  # {hand: {stat: {mean, std}}}
        self.pitcher_params: dict[str, dict] = {}

    def fit_batters(
        self,
        season_data: pd.DataFrame,
        min_pa: int = 50,
    ) -> None:
        """Compute league mean/std for batter stats by pitcher hand faced.

        Args:
            season_data: Full season batted ball data.
            min_pa: Minimum plate appearances to be included in population.
        """
        for hand in ["L", "R"]:
            # Get all unique batters who face this pitcher hand
            hand_data = season_data[season_data["p_throws"] == hand] if "p_throws" in season_data.columns else season_data
            batter_ids = hand_data["batter"].unique() if "batter" in hand_data.columns else []

            all_stats = []
            for bid in batter_ids:
                stats = compute_batter_stats(season_data, bid, vs_hand=hand)
                if stats["pa_count"] >= min_pa:
                    all_stats.append(stats)

            if not all_stats:
                self.batter_params[hand] = {k: {"mean": 0.0, "std": 1.0} for k in BATTER_STAT_KEYS}
                continue

            df = pd.DataFrame(all_stats)
            self.batter_params[hand] = {}
            for key in BATTER_STAT_KEYS:
                if key in df.columns:
                    mean = df[key].mean()
                    std = df[key].std()
                    self.batter_params[hand][key] = {
                        "mean": float(mean) if not np.isnan(mean) else 0.0,
                        "std": float(std) if not np.isnan(std) and std > 0 else 1.0,
                    }
                else:
                    self.batter_params[hand][key] = {"mean": 0.0, "std": 1.0}

    def fit_pitchers(
        self,
        season_data: pd.DataFrame,
        min_pa: int = 50,
    ) -> None:
        """Compute league mean/std for pitcher stats by batter hand faced."""
        for hand in ["L", "R"]:
            hand_data = season_data[season_data["stand"] == hand] if "stand" in season_data.columns else season_data
            pitcher_ids = hand_data["pitcher"].unique() if "pitcher" in hand_data.columns else []

            all_stats = []
            for pid in pitcher_ids:
                stats = compute_pitcher_stats(season_data, pid, vs_hand=hand)
                if stats["pa_count"] >= min_pa:
                    all_stats.append(stats)

            if not all_stats:
                self.pitcher_params[hand] = {k: {"mean": 0.0, "std": 1.0} for k in PITCHER_STAT_KEYS}
                continue

            df = pd.DataFrame(all_stats)
            self.pitcher_params[hand] = {}
            for key in PITCHER_STAT_KEYS:
                if key in df.columns:
                    mean = df[key].mean()
                    std = df[key].std()
                    self.pitcher_params[hand][key] = {
                        "mean": float(mean) if not np.isnan(mean) else 0.0,
                        "std": float(std) if not np.isnan(std) and std > 0 else 1.0,
                    }
                else:
                    self.pitcher_params[hand][key] = {"mean": 0.0, "std": 1.0}

    def normalize_batter(self, stats: dict, vs_hand: str) -> dict:
        """Z-score normalize a batter's stats against league population.

        Z_i = (X_i - mu) / sigma
        """
        params = self.batter_params.get(vs_hand, self.batter_params.get("R", {}))
        normalized = {}
        for key in BATTER_STAT_KEYS:
            val = stats.get(key, 0.0)
            p = params.get(key, {"mean": 0.0, "std": 1.0})
            normalized[key] = (val - p["mean"]) / p["std"]
        normalized["pa_count"] = stats.get("pa_count", 0)
        return normalized

    def normalize_pitcher(self, stats: dict, vs_hand: str) -> dict:
        """Z-score normalize a pitcher's stats against league population."""
        params = self.pitcher_params.get(vs_hand, self.pitcher_params.get("R", {}))
        normalized = {}
        for key in PITCHER_STAT_KEYS:
            val = stats.get(key, 0.0)
            p = params.get(key, {"mean": 0.0, "std": 1.0})
            normalized[key] = (val - p["mean"]) / p["std"]
        normalized["pa_count"] = stats.get("pa_count", 0)
        return normalized

    def to_dict(self) -> dict:
        """Serialize for caching."""
        return {
            "batter_params": self.batter_params,
            "pitcher_params": self.pitcher_params,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LeagueNormalizer":
        """Deserialize from cache."""
        n = cls()
        n.batter_params = data.get("batter_params", {})
        n.pitcher_params = data.get("pitcher_params", {})
        return n
