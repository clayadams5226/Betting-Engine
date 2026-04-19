"""End-to-end daily prediction pipeline.

Date in -> sorted HR predictions out, orchestrating all stages:
1. Fetch schedule + lineups
2. Pull Statcast data (season + 30-day rolling)
3. Compute batter/pitcher stats with platoon splits + Bayesian shrinkage
4. Blend time windows
5. Z-score normalize
6. PCA transform
7. Compute environmental score
8. Combine via logistic model
9. Calibrate
10. Sort and return
"""

from datetime import date, timedelta
from typing import Optional

import pandas as pd

from ..cache import db as cache
from ..data.blender import blend_batter_stats, blend_pitcher_stats
from ..data.env_score import compute_env_score
from ..data.loader import (
    apply_bayesian_shrinkage,
    compute_batter_stats,
    compute_pitcher_stats,
    fetch_statcast_range,
    get_lineup_with_roster_fallback,
    get_schedule,
    load_stadiums,
)
from ..data.normalizer import LeagueNormalizer
from ..data.weather import get_weather
from .calibrator import HRCalibrator
from .combiner import HRCombiner
from .pca import HRPcaModel


class HRPredictionPipeline:
    """Orchestrates the full HR prediction pipeline for a given game day."""

    def __init__(self):
        self.pca_model = HRPcaModel()
        self.combiner = HRCombiner()
        self.calibrator = HRCalibrator()
        self.normalizer = LeagueNormalizer()
        self.stadiums = load_stadiums()
        self._models_loaded = False

    def load_models(self) -> None:
        """Load pre-trained models from disk."""
        import json
        from pathlib import Path

        model_dir = Path(__file__).parent / "saved"

        try:
            self.pca_model.load(model_dir)
            self.combiner.load(model_dir)
            self._models_loaded = True
        except FileNotFoundError:
            self._models_loaded = False

        try:
            self.calibrator.load(model_dir)
        except FileNotFoundError:
            pass  # Calibrator is optional

        # Load pre-baked normalizer params (stable league averages)
        normalizer_path = model_dir / "normalizer.json"
        if normalizer_path.exists():
            with open(normalizer_path) as f:
                self.normalizer = LeagueNormalizer.from_dict(json.load(f))

    def predict_day(self, game_date: date) -> list[dict]:
        """Run full prediction pipeline for all games on a date.

        Returns list of game dicts, each containing batters sorted by HR probability.
        """
        # Check cache first
        cached = cache.get_daily("predictions", game_date)
        if cached:
            return cached["games"]

        # Step 1: Get schedule
        schedule = get_schedule(game_date)
        if not schedule:
            return []

        # Step 2: Fetch Statcast data
        season_start = date(game_date.year, 3, 20)  # approx Opening Day
        rolling_start = game_date - timedelta(days=30)

        season_data = self._fetch_cached_statcast(
            "season", season_start, game_date - timedelta(days=1)
        )
        rolling_data = self._fetch_cached_statcast(
            "rolling", rolling_start, game_date - timedelta(days=1)
        )

        # Step 3: If no pre-baked normalizer, fit from season data
        if not self.normalizer.batter_params and not season_data.empty:
            self.normalizer.fit_batters(season_data)
            self.normalizer.fit_pitchers(season_data)

        # Step 4: If no pre-trained PCA, fit from season data
        if not self._models_loaded and not season_data.empty:
            self._fit_pca_from_season(season_data)

        # Step 5: Process each game
        games = []
        for game in schedule:
            game_result = self._process_game(
                game, season_data, rolling_data, game_date
            )
            if game_result:
                games.append(game_result)

        # Cache results
        cache.put_daily("predictions", game_date, {"games": games})

        return games

    def _process_game(
        self,
        game: dict,
        season_data: pd.DataFrame,
        rolling_data: pd.DataFrame,
        game_date: date,
    ) -> Optional[dict]:
        """Process a single game: compute predictions for all batters."""
        # Get lineups (falls back to active roster if lineups not posted)
        lineups = get_lineup_with_roster_fallback(
            game["game_id"],
            home_team_id=game.get("home_id", 0),
            away_team_id=game.get("away_id", 0),
        )

        # Get stadium info
        home_abbr = game.get("home_abbr", "")
        stadium = self.stadiums.get(home_abbr, {})

        # Get weather
        weather = get_weather(
            lat=stadium.get("lat", 40.0),
            lon=stadium.get("lon", -74.0),
        )

        # Compute environmental score
        env_result = compute_env_score(
            park_hr_factor=stadium.get("hr_park_factor", 100),
            weather=weather,
            cf_bearing_deg=stadium.get("cf_bearing_deg", 0),
            roof=stadium.get("roof", "open"),
        )

        all_batters = []

        # Process home batters vs away pitcher
        away_pitcher_id = game.get("away_pitcher_id")
        away_pitcher_hand = game.get("away_pitcher_hand", "R")
        for batter in lineups.get("home", []):
            pred = self._predict_batter(
                batter=batter,
                pitcher_id=away_pitcher_id,
                pitcher_hand=away_pitcher_hand,
                pitcher_name=game.get("away_pitcher_name", "TBD"),
                team=game.get("home_abbr", ""),
                season_data=season_data,
                rolling_data=rolling_data,
                env_score=env_result["env_score"],
                game_date=game_date,
            )
            if pred:
                all_batters.append(pred)

        # Process away batters vs home pitcher
        home_pitcher_id = game.get("home_pitcher_id")
        home_pitcher_hand = game.get("home_pitcher_hand", "R")
        for batter in lineups.get("away", []):
            pred = self._predict_batter(
                batter=batter,
                pitcher_id=home_pitcher_id,
                pitcher_hand=home_pitcher_hand,
                pitcher_name=game.get("home_pitcher_name", "TBD"),
                team=game.get("away_abbr", ""),
                season_data=season_data,
                rolling_data=rolling_data,
                env_score=env_result["env_score"],
                game_date=game_date,
            )
            if pred:
                all_batters.append(pred)

        # Sort by HR probability descending
        all_batters.sort(key=lambda x: x["hr_probability"], reverse=True)

        return {
            "game_id": game["game_id"],
            "home_team": game.get("home", ""),
            "home_abbr": game.get("home_abbr", ""),
            "away_team": game.get("away", ""),
            "away_abbr": game.get("away_abbr", ""),
            "venue": game.get("venue", ""),
            "home_pitcher": game.get("home_pitcher_name", "TBD"),
            "away_pitcher": game.get("away_pitcher_name", "TBD"),
            "weather": {
                "temp_f": weather.get("temp_f"),
                "wind_speed_mph": weather.get("wind_speed_mph"),
                "wind_deg": weather.get("wind_deg"),
                "humidity_pct": weather.get("humidity_pct"),
                "description": weather.get("description"),
            },
            "env_score": env_result["env_score"],
            "env_details": env_result,
            "batters": all_batters,
        }

    def _predict_batter(
        self,
        batter: dict,
        pitcher_id: Optional[int],
        pitcher_hand: str,
        pitcher_name: str,
        team: str,
        season_data: pd.DataFrame,
        rolling_data: pd.DataFrame,
        env_score: float,
        game_date: date,
    ) -> Optional[dict]:
        """Compute HR probability for a single batter matchup."""
        batter_id = batter.get("id")
        batter_hand = batter.get("batting_hand", "R")
        if not batter_id:
            return None

        # Batter stats: platoon split + Bayesian shrinkage
        batter_split = blend_batter_stats(
            season_data, rolling_data, batter_id,
            vs_hand=pitcher_hand, current_date=game_date,
        )
        batter_overall = blend_batter_stats(
            season_data, rolling_data, batter_id,
            vs_hand=None, current_date=game_date,
        )
        batter_blended = apply_bayesian_shrinkage(batter_split, batter_overall)

        # Pitcher stats: platoon split + shrinkage
        # Default to league-average pitcher stats when pitcher_id is missing
        from ..data.loader import _league_avg_pitcher_stats
        pitcher_blended = _league_avg_pitcher_stats()
        if pitcher_id:
            pitcher_split = blend_pitcher_stats(
                season_data, rolling_data, pitcher_id,
                vs_hand=batter_hand, current_date=game_date,
            )
            pitcher_overall = blend_pitcher_stats(
                season_data, rolling_data, pitcher_id,
                vs_hand=None, current_date=game_date,
            )
            pitcher_blended = apply_bayesian_shrinkage(pitcher_split, pitcher_overall)

        # Normalize
        batter_norm = self.normalizer.normalize_batter(batter_blended, vs_hand=pitcher_hand)
        pitcher_norm = self.normalizer.normalize_pitcher(pitcher_blended, vs_hand=batter_hand)

        # PCA transform
        batter_pca = self.pca_model.transform_batter(batter_norm)
        pitcher_pca = self.pca_model.transform_pitcher(pitcher_norm)

        batter_score = batter_pca["batter_score"]
        pitcher_score = pitcher_pca["pitcher_score"]

        # Combine
        raw_prob = self.combiner.predict_proba(batter_score, pitcher_score, env_score)

        # Calibrate
        calibrated_prob = self.calibrator.calibrate(raw_prob)

        # Confidence tier
        batter_pa = batter_blended.get("pa_count", 0)
        pitcher_pa = pitcher_blended.get("pa_count", 0)
        confidence = _confidence_tier(batter_pa, pitcher_pa)

        return {
            "batter_name": batter.get("name", "Unknown"),
            "batter_id": batter_id,
            "team": team,
            "position": batter.get("position", ""),
            "batting_order": batter.get("batting_order", ""),
            "opposing_pitcher": pitcher_name,
            "hr_probability": round(calibrated_prob, 4),
            "batter_score": round(batter_score, 3),
            "pitcher_score": round(pitcher_score, 3),
            "env_score": round(env_score, 3),
            "confidence": confidence,
        }

    def _fetch_cached_statcast(
        self, namespace: str, start: date, end: date
    ) -> pd.DataFrame:
        """Fetch Statcast data with caching."""
        cache_key = f"statcast:{namespace}:{start}:{end}"
        cached = cache.get(cache_key)
        if cached is not None:
            return pd.DataFrame(cached)

        df = fetch_statcast_range(start, end)

        if not df.empty:
            # Only cache if data is reasonable size
            records = df.head(50000).to_dict("records")
            cache.put(cache_key, records)

        return df

    def _fit_pca_from_season(self, season_data: pd.DataFrame) -> None:
        """Fit PCA models from current season data when no pre-trained model exists."""
        if "batter" not in season_data.columns:
            return

        # Collect normalized stats for all qualifying batters
        batter_stats = []
        for hand in ["L", "R"]:
            hand_data = season_data[season_data["p_throws"] == hand] if "p_throws" in season_data.columns else season_data
            for bid in hand_data["batter"].unique():
                stats = compute_batter_stats(season_data, bid, vs_hand=hand)
                if stats["pa_count"] >= 50:
                    norm = self.normalizer.normalize_batter(stats, vs_hand=hand)
                    batter_stats.append(norm)

        if len(batter_stats) >= 10:
            self.pca_model.fit_batter(batter_stats)

        # Same for pitchers
        pitcher_stats = []
        for hand in ["L", "R"]:
            hand_data = season_data[season_data["stand"] == hand] if "stand" in season_data.columns else season_data
            for pid in hand_data["pitcher"].unique():
                stats = compute_pitcher_stats(season_data, pid, vs_hand=hand)
                if stats["pa_count"] >= 50:
                    norm = self.normalizer.normalize_pitcher(stats, vs_hand=hand)
                    pitcher_stats.append(norm)

        if len(pitcher_stats) >= 10:
            self.pca_model.fit_pitcher(pitcher_stats)


def _confidence_tier(batter_pa: int, pitcher_pa: int) -> str:
    if batter_pa >= 150 and pitcher_pa >= 80:
        return "high"
    elif batter_pa >= 60 or pitcher_pa >= 30:
        return "medium"
    else:
        return "low"
