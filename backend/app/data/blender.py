"""Time-window blending: 30-day rolling + full season with stat-specific weights."""

from datetime import date, timedelta
from typing import Optional

import pandas as pd

from .loader import compute_batter_stats, compute_pitcher_stats

# Stat-specific weights for rolling 30-day window
# Higher = more weight on recent performance
BATTER_RECENT_WEIGHTS = {
    "barrel_pct": 0.35,
    "hard_hit_pct": 0.45,
    "avg_ev": 0.50,
    "max_ev": 0.55,
    "avg_launch_angle": 0.40,
    "sweet_spot_pct": 0.38,
}

PITCHER_RECENT_WEIGHTS = {
    "hr_per_9": 0.25,
    "hr_fb_pct": 0.20,
    "barrel_pct_allowed": 0.35,
    "hard_hit_pct_allowed": 0.45,
    "xslg_against": 0.40,
    "xwoba_against": 0.40,
}

# Stat keys (excluding pa_count) for each type
BATTER_STAT_KEYS = list(BATTER_RECENT_WEIGHTS.keys())
PITCHER_STAT_KEYS = list(PITCHER_RECENT_WEIGHTS.keys())


def blend_batter_stats(
    season_data: pd.DataFrame,
    rolling_data: pd.DataFrame,
    batter_id: int,
    vs_hand: Optional[str] = None,
    prior_season_data: Optional[pd.DataFrame] = None,
    current_date: Optional[date] = None,
) -> dict:
    """Blend 30-day rolling and full season batter stats.

    X_blended = w_recent * X_30day + (1 - w_recent) * X_season

    Early-season fallback (pre-May): 30% current + 70% prior regressed.
    Small rolling sample: scale w_recent proportionally to PA/120.
    """
    season_stats = compute_batter_stats(season_data, batter_id, vs_hand)
    rolling_stats = compute_batter_stats(rolling_data, batter_id, vs_hand)

    # Early-season fallback
    if _is_early_season(current_date) and prior_season_data is not None:
        prior_stats = compute_batter_stats(prior_season_data, batter_id, vs_hand)
        season_stats = _regress_to_prior(season_stats, prior_stats, BATTER_STAT_KEYS)

    return _blend(
        season_stats=season_stats,
        rolling_stats=rolling_stats,
        recent_weights=BATTER_RECENT_WEIGHTS,
        stat_keys=BATTER_STAT_KEYS,
    )


def blend_pitcher_stats(
    season_data: pd.DataFrame,
    rolling_data: pd.DataFrame,
    pitcher_id: int,
    vs_hand: Optional[str] = None,
    prior_season_data: Optional[pd.DataFrame] = None,
    current_date: Optional[date] = None,
) -> dict:
    """Blend 30-day rolling and full season pitcher stats."""
    season_stats = compute_pitcher_stats(season_data, pitcher_id, vs_hand)
    rolling_stats = compute_pitcher_stats(rolling_data, pitcher_id, vs_hand)

    if _is_early_season(current_date) and prior_season_data is not None:
        prior_stats = compute_pitcher_stats(prior_season_data, pitcher_id, vs_hand)
        season_stats = _regress_to_prior(season_stats, prior_stats, PITCHER_STAT_KEYS)

    return _blend(
        season_stats=season_stats,
        rolling_stats=rolling_stats,
        recent_weights=PITCHER_RECENT_WEIGHTS,
        stat_keys=PITCHER_STAT_KEYS,
    )


def _blend(
    season_stats: dict,
    rolling_stats: dict,
    recent_weights: dict,
    stat_keys: list[str],
) -> dict:
    """Core blending logic."""
    rolling_pa = rolling_stats.get("pa_count", 0)

    blended = {"pa_count": season_stats.get("pa_count", 0)}

    for key in stat_keys:
        w_recent = recent_weights.get(key, 0.4)

        # Scale down rolling weight if small sample
        if rolling_pa < 120:
            w_recent = max(0.10, (rolling_pa / 120) * w_recent)

        season_val = season_stats.get(key, 0.0)
        rolling_val = rolling_stats.get(key, 0.0)

        # If no rolling data, use season only
        if rolling_pa == 0:
            blended[key] = season_val
        else:
            blended[key] = w_recent * rolling_val + (1 - w_recent) * season_val

    return blended


def _is_early_season(current_date: Optional[date]) -> bool:
    """Check if we're in the early season (before May 1)."""
    if current_date is None:
        current_date = date.today()
    return current_date.month <= 4


def _regress_to_prior(
    current_stats: dict,
    prior_stats: dict,
    stat_keys: list[str],
    current_weight: float = 0.30,
    regression_to_mean: float = 0.25,
) -> dict:
    """Early-season: blend current with prior season, regressed toward league avg.

    30% current season + 70% prior (regressed 25% toward league average).
    """
    # League average approximations for regression
    league_avg = {
        # Batter stats
        "barrel_pct": 0.065,
        "hard_hit_pct": 0.35,
        "avg_ev": 88.5,
        "max_ev": 108.0,
        "avg_launch_angle": 12.0,
        "sweet_spot_pct": 0.33,
        # Pitcher stats
        "hr_per_9": 1.25,
        "hr_fb_pct": 0.12,
        "barrel_pct_allowed": 0.065,
        "hard_hit_pct_allowed": 0.35,
        "xslg_against": 0.400,
        "xwoba_against": 0.315,
    }

    prior_weight = 1.0 - current_weight
    result = {"pa_count": current_stats.get("pa_count", 0)}

    for key in stat_keys:
        current_val = current_stats.get(key, 0.0)
        prior_val = prior_stats.get(key, 0.0)
        avg_val = league_avg.get(key, 0.0)

        # Regress prior toward league average
        regressed_prior = (1 - regression_to_mean) * prior_val + regression_to_mean * avg_val

        # If no current season data, rely entirely on regressed prior
        if current_stats.get("pa_count", 0) == 0:
            result[key] = regressed_prior
        else:
            result[key] = current_weight * current_val + prior_weight * regressed_prior

    return result
