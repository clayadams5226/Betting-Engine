"""Historical data training pipeline using pre-aggregated Statcast leaderboards.

Uses pybaseball's leaderboard endpoints (much faster than raw pitch data):
- statcast_batter_exitvelo_barrels: Barrel%, Hard-Hit%, EV stats
- statcast_batter_expected_stats: xSLG, xwOBA
- statcast_pitcher_exitvelo_barrels: pitcher barrel/EV allowed
- statcast_pitcher_expected_stats: pitcher xSLG/xwOBA allowed

Usage:
  cd backend
  venv\\Scripts\\activate
  python -m app.training.train
"""

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from pybaseball import (
    cache as pb_cache,
    statcast,
    statcast_batter_exitvelo_barrels,
    statcast_batter_expected_stats,
    statcast_pitcher_exitvelo_barrels,
    statcast_pitcher_expected_stats,
)

from ..data.normalizer import BATTER_STAT_KEYS, PITCHER_STAT_KEYS, LeagueNormalizer
from ..models.calibrator import HRCalibrator
from ..models.combiner import HRCombiner
from ..models.pca import HRPcaModel

MODEL_DIR = Path(__file__).parent.parent / "models" / "saved"

# Enable pybaseball caching so re-runs are instant
pb_cache.enable()


def train_full_pipeline(
    seasons: list[int] | None = None,
    calibration_split: float = 0.2,
    min_pa_batter: int = 100,
    min_bip_pitcher: int = 100,
) -> dict:
    """Train all models on historical leaderboard data.

    Args:
        seasons: Years to train on (default: [2023, 2024]).
        calibration_split: Fraction held out for calibration.
        min_pa_batter: Minimum plate appearances for batter inclusion.
        min_bip_pitcher: Minimum balls in play for pitcher inclusion.
    """
    if seasons is None:
        seasons = [2023, 2024]

    print(f"=== Training HR Prediction Model ===")
    print(f"Seasons: {seasons}")
    print()

    # -------------------------------------------------------
    # Step 1: Fetch pre-aggregated leaderboard stats
    # -------------------------------------------------------
    print("Step 1: Fetching leaderboard data from Baseball Savant...")
    batter_dfs = []
    pitcher_dfs = []

    for year in seasons:
        print(f"  {year} batters (exit velo + barrels)...")
        b_ev = statcast_batter_exitvelo_barrels(year, min_pa_batter)
        print(f"    -> {len(b_ev)} batters")

        print(f"  {year} batters (expected stats)...")
        b_exp = statcast_batter_expected_stats(year, min_pa_batter)
        print(f"    -> {len(b_exp)} batters")

        print(f"  {year} pitchers (exit velo + barrels)...")
        p_ev = statcast_pitcher_exitvelo_barrels(year, min_bip_pitcher)
        print(f"    -> {len(p_ev)} pitchers")

        print(f"  {year} pitchers (expected stats)...")
        p_exp = statcast_pitcher_expected_stats(year, min_bip_pitcher)
        print(f"    -> {len(p_exp)} pitchers")

        # Merge batter EV + expected stats
        b_merged = _merge_batter_leaderboards(b_ev, b_exp, year)
        if b_merged is not None:
            batter_dfs.append(b_merged)

        # Merge pitcher EV + expected stats
        p_merged = _merge_pitcher_leaderboards(p_ev, p_exp, year)
        if p_merged is not None:
            pitcher_dfs.append(p_merged)

    if not batter_dfs or not pitcher_dfs:
        print("ERROR: No data fetched.")
        return {"error": "no_data"}

    all_batters = pd.concat(batter_dfs, ignore_index=True)
    all_pitchers = pd.concat(pitcher_dfs, ignore_index=True)
    print(f"\n  Total batter-seasons: {len(all_batters)}")
    print(f"  Total pitcher-seasons: {len(all_pitchers)}")

    # -------------------------------------------------------
    # Step 2: Fit normalizer from population stats
    # -------------------------------------------------------
    print("\nStep 2: Fitting normalizer...")
    normalizer = _fit_normalizer(all_batters, all_pitchers)
    print("  Done.")

    # -------------------------------------------------------
    # Step 3: Normalize all stats and fit PCA
    # -------------------------------------------------------
    print("\nStep 3: Normalizing stats and fitting PCA...")
    batter_norm = _normalize_batters(all_batters, normalizer)
    pitcher_norm = _normalize_pitchers(all_pitchers, normalizer)

    pca_model = HRPcaModel()
    b_pca_info = pca_model.fit_batter(batter_norm)
    p_pca_info = pca_model.fit_pitcher(pitcher_norm)

    print(f"  Batter PCA variance: {[round(v, 3) for v in b_pca_info['cumulative_variance']]}")
    print(f"  Pitcher PCA variance: {[round(v, 3) for v in p_pca_info['cumulative_variance']]}")

    # -------------------------------------------------------
    # Step 4: Build per-game training data for combiner
    # -------------------------------------------------------
    print("\nStep 4: Building per-game training data...")
    print("  Fetching game-level Statcast data for HR outcomes...")

    training = _build_combiner_training_data(
        seasons, all_batters, all_pitchers, normalizer, pca_model
    )

    n_samples = len(training["hr_outcomes"])
    hr_rate = np.mean(training["hr_outcomes"]) if n_samples > 0 else 0
    print(f"  Training samples: {n_samples}")
    print(f"  HR rate: {hr_rate:.4f} ({hr_rate*100:.1f}%)")

    if n_samples < 100:
        print("WARNING: Too few training samples. Results may be unreliable.")

    # -------------------------------------------------------
    # Step 5: Train combiner (logistic regression)
    # -------------------------------------------------------
    print("\nStep 5: Training logistic combiner...")
    n = n_samples
    indices = np.random.permutation(n)
    split = int(n * (1 - calibration_split))
    train_idx = indices[:split]
    cal_idx = indices[split:]

    combiner = HRCombiner()
    train_result = combiner.fit(
        batter_scores=[training["batter_scores"][i] for i in train_idx],
        pitcher_scores=[training["pitcher_scores"][i] for i in train_idx],
        env_scores=[training["env_scores"][i] for i in train_idx],
        hr_outcomes=[training["hr_outcomes"][i] for i in train_idx],
    )
    print(f"  Coefficients: {train_result['coefficients']}")
    print(f"  Mean predicted: {train_result['mean_predicted']:.4f}")
    print(f"  Actual HR rate: {train_result['actual_hr_rate']:.4f}")

    # -------------------------------------------------------
    # Step 6: Train calibrator on held-out data
    # -------------------------------------------------------
    print("\nStep 6: Training calibrator...")
    calibrator = HRCalibrator()
    cal_preds = [
        combiner.predict_proba(
            training["batter_scores"][i],
            training["pitcher_scores"][i],
            training["env_scores"][i],
        )
        for i in cal_idx
    ]
    cal_actual = [training["hr_outcomes"][i] for i in cal_idx]
    cal_result = calibrator.fit(cal_preds, cal_actual)
    print(f"  Calibrated mean: {cal_result['mean_predicted_calibrated']:.4f}")
    print(f"  Actual rate: {cal_result['actual_rate']:.4f}")

    # -------------------------------------------------------
    # Step 7: Save everything
    # -------------------------------------------------------
    print(f"\nStep 7: Saving models to {MODEL_DIR}...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    pca_model.save(MODEL_DIR)
    combiner.save(MODEL_DIR)
    calibrator.save(MODEL_DIR)

    with open(MODEL_DIR / "normalizer.json", "w") as f:
        json.dump(normalizer.to_dict(), f, indent=2)

    # Clear prediction cache so new model is used
    cache_path = Path(__file__).parent.parent / "cache" / "cache.db"
    if cache_path.exists():
        cache_path.unlink()
        print("  Cleared prediction cache.")

    print("\n=== Training Complete! ===")
    print(f"  Restart your backend to use the new model.")

    return {
        "seasons": seasons,
        "batter_seasons": len(all_batters),
        "pitcher_seasons": len(all_pitchers),
        "training_samples": n_samples,
        "hr_rate": hr_rate,
        "batter_pca": b_pca_info,
        "pitcher_pca": p_pca_info,
        "combiner": train_result,
        "calibration": cal_result,
    }


# -------------------------------------------------------
# Leaderboard merging helpers
# -------------------------------------------------------

def _merge_batter_leaderboards(ev_df: pd.DataFrame, exp_df: pd.DataFrame, year: int) -> pd.DataFrame | None:
    """Merge batter exit-velo/barrel stats with expected stats."""
    if ev_df is None or ev_df.empty or exp_df is None or exp_df.empty:
        return None

    # Standardize column names from EV leaderboard
    ev = ev_df.copy()
    ev_col_map = {
        "player_id": "player_id",
        "avg_hit_speed": "avg_ev",
        "max_hit_speed": "max_ev",
        "brl_percent": "barrel_pct",
        "ev50": "avg_ev",  # alternate name
    }
    # Find the right columns
    ev_result = pd.DataFrame()
    ev_result["player_id"] = ev["player_id"] if "player_id" in ev.columns else ev.index

    for target, source in [
        ("avg_ev", ["avg_hit_speed", "ev50"]),
        ("max_ev", ["max_hit_speed", "ev95percent"]),
        ("barrel_pct", ["brl_percent"]),
        ("hard_hit_pct", ["hard_hit_percent", "ev95percent"]),
    ]:
        for col in source:
            if col in ev.columns:
                ev_result[target] = pd.to_numeric(ev[col], errors="coerce")
                break

    # Hard-hit% might need conversion (some formats are 0-100, some 0-1)
    if "hard_hit_pct" in ev_result.columns:
        if ev_result["hard_hit_pct"].mean() > 1:
            ev_result["hard_hit_pct"] = ev_result["hard_hit_pct"] / 100.0
    if "barrel_pct" in ev_result.columns:
        if ev_result["barrel_pct"].mean() > 1:
            ev_result["barrel_pct"] = ev_result["barrel_pct"] / 100.0

    # Expected stats
    exp = exp_df.copy()
    exp_result = pd.DataFrame()
    exp_result["player_id"] = exp["player_id"] if "player_id" in exp.columns else exp.index

    for target, sources in [
        ("xslg", ["est_slg", "xslg"]),
        ("xwoba", ["est_woba", "xwoba"]),
        ("pa", ["pa", "bip"]),
    ]:
        for col in sources:
            if col in exp.columns:
                exp_result[target] = pd.to_numeric(exp[col], errors="coerce")
                break

    # Merge on player_id
    merged = ev_result.merge(exp_result, on="player_id", how="inner")
    merged["year"] = year

    # Compute missing fields with defaults
    if "avg_launch_angle" not in merged.columns:
        merged["avg_launch_angle"] = 12.0  # Will be updated if available
    if "sweet_spot_pct" not in merged.columns:
        merged["sweet_spot_pct"] = 0.33

    # Try to get launch angle from EV dataframe
    for col in ["launch_angle_avg", "avg_launch_angle", "anglesweetspotpercent"]:
        if col in ev.columns:
            temp = pd.to_numeric(ev[col], errors="coerce")
            if col == "anglesweetspotpercent":
                merged["sweet_spot_pct"] = temp.values[:len(merged)] / 100.0 if temp.mean() > 1 else temp.values[:len(merged)]
            else:
                merged["avg_launch_angle"] = temp.values[:len(merged)]
            break

    # Sweet spot from EV dataframe
    for col in ["anglesweetspotpercent", "sweet_spot_percent"]:
        if col in ev.columns:
            temp = pd.to_numeric(ev[col], errors="coerce")
            vals = temp.reindex(merged.index).values
            if len(vals) == len(merged):
                merged["sweet_spot_pct"] = vals / 100.0 if temp.mean() > 1 else vals
            break

    return merged


def _merge_pitcher_leaderboards(ev_df: pd.DataFrame, exp_df: pd.DataFrame, year: int) -> pd.DataFrame | None:
    """Merge pitcher exit-velo/barrel stats with expected stats."""
    if ev_df is None or ev_df.empty or exp_df is None or exp_df.empty:
        return None

    ev = ev_df.copy()
    ev_result = pd.DataFrame()
    ev_result["player_id"] = ev["player_id"] if "player_id" in ev.columns else ev.index

    for target, sources in [
        ("barrel_pct_allowed", ["brl_percent"]),
        ("hard_hit_pct_allowed", ["hard_hit_percent"]),
    ]:
        for col in sources:
            if col in ev.columns:
                ev_result[target] = pd.to_numeric(ev[col], errors="coerce")
                break

    if "barrel_pct_allowed" in ev_result.columns and ev_result["barrel_pct_allowed"].mean() > 1:
        ev_result["barrel_pct_allowed"] = ev_result["barrel_pct_allowed"] / 100.0
    if "hard_hit_pct_allowed" in ev_result.columns and ev_result["hard_hit_pct_allowed"].mean() > 1:
        ev_result["hard_hit_pct_allowed"] = ev_result["hard_hit_pct_allowed"] / 100.0

    exp = exp_df.copy()
    exp_result = pd.DataFrame()
    exp_result["player_id"] = exp["player_id"] if "player_id" in exp.columns else exp.index

    for target, sources in [
        ("xslg_against", ["est_slg", "xslg"]),
        ("xwoba_against", ["est_woba", "xwoba"]),
        ("pa", ["pa", "bip"]),
    ]:
        for col in sources:
            if col in exp.columns:
                exp_result[target] = pd.to_numeric(exp[col], errors="coerce")
                break

    merged = ev_result.merge(exp_result, on="player_id", how="inner")
    merged["year"] = year

    # HR/9 and HR/FB% defaults (will be refined by combiner training)
    if "hr_per_9" not in merged.columns:
        merged["hr_per_9"] = 1.25
    if "hr_fb_pct" not in merged.columns:
        merged["hr_fb_pct"] = 0.115

    return merged


# -------------------------------------------------------
# Normalizer fitting
# -------------------------------------------------------

def _fit_normalizer(all_batters: pd.DataFrame, all_pitchers: pd.DataFrame) -> LeagueNormalizer:
    """Fit normalizer from leaderboard population stats."""
    normalizer = LeagueNormalizer()

    # For leaderboard data we don't have platoon splits, so use same params for L and R
    batter_params = {}
    for key in BATTER_STAT_KEYS:
        if key in all_batters.columns:
            vals = all_batters[key].dropna()
            batter_params[key] = {
                "mean": float(vals.mean()),
                "std": float(vals.std()) if vals.std() > 0 else 1.0,
            }
        else:
            batter_params[key] = {"mean": 0.0, "std": 1.0}

    normalizer.batter_params = {"L": batter_params, "R": batter_params.copy()}

    pitcher_params = {}
    for key in PITCHER_STAT_KEYS:
        if key in all_pitchers.columns:
            vals = all_pitchers[key].dropna()
            pitcher_params[key] = {
                "mean": float(vals.mean()),
                "std": float(vals.std()) if vals.std() > 0 else 1.0,
            }
        else:
            pitcher_params[key] = {"mean": 0.0, "std": 1.0}

    normalizer.pitcher_params = {"L": pitcher_params, "R": pitcher_params.copy()}

    return normalizer


def _normalize_batters(df: pd.DataFrame, normalizer: LeagueNormalizer) -> list[dict]:
    """Normalize batter stats for PCA training."""
    stats = []
    for _, row in df.iterrows():
        raw = {k: float(row.get(k, 0.0) or 0.0) for k in BATTER_STAT_KEYS}
        raw["pa_count"] = int(row.get("pa", 100) or 100)
        norm = normalizer.normalize_batter(raw, vs_hand="R")
        stats.append(norm)
    return stats


def _normalize_pitchers(df: pd.DataFrame, normalizer: LeagueNormalizer) -> list[dict]:
    """Normalize pitcher stats for PCA training."""
    stats = []
    for _, row in df.iterrows():
        raw = {k: float(row.get(k, 0.0) or 0.0) for k in PITCHER_STAT_KEYS}
        raw["pa_count"] = int(row.get("pa", 100) or 100)
        norm = normalizer.normalize_pitcher(raw, vs_hand="R")
        stats.append(norm)
    return stats


# -------------------------------------------------------
# Combiner training data (needs per-game HR outcomes)
# -------------------------------------------------------

def _build_combiner_training_data(
    seasons: list[int],
    all_batters: pd.DataFrame,
    all_pitchers: pd.DataFrame,
    normalizer: LeagueNormalizer,
    pca_model: HRPcaModel,
) -> dict:
    """Build training data for the logistic combiner.

    Fetches a sample of Statcast game data to get per-batter HR outcomes,
    then matches with leaderboard stats + PCA scores.
    """
    batter_scores = []
    pitcher_scores = []
    env_scores = []
    hr_outcomes = []

    # Pre-compute PCA scores for all batters and pitchers
    batter_score_map = {}
    for _, row in all_batters.iterrows():
        pid = row.get("player_id")
        if pd.isna(pid):
            continue
        pid = int(pid)
        raw = {k: float(row.get(k, 0.0) or 0.0) for k in BATTER_STAT_KEYS}
        raw["pa_count"] = int(row.get("pa", 100) or 100)
        norm = normalizer.normalize_batter(raw, vs_hand="R")
        pca_result = pca_model.transform_batter(norm)
        year = row.get("year", 2024)
        batter_score_map[(pid, year)] = pca_result["batter_score"]

    pitcher_score_map = {}
    for _, row in all_pitchers.iterrows():
        pid = row.get("player_id")
        if pd.isna(pid):
            continue
        pid = int(pid)
        raw = {k: float(row.get(k, 0.0) or 0.0) for k in PITCHER_STAT_KEYS}
        raw["pa_count"] = int(row.get("pa", 100) or 100)
        norm = normalizer.normalize_pitcher(raw, vs_hand="R")
        pca_result = pca_model.transform_pitcher(norm)
        year = row.get("year", 2024)
        pitcher_score_map[(pid, year)] = pca_result["pitcher_score"]

    # Fetch game-level data to get HR outcomes
    # Sample 2 weeks from each season (enough for training, fast to download)
    for year in seasons:
        for month, day_start, day_end in [(6, 1, 14), (8, 1, 14)]:
            start = date(year, month, day_start)
            end = date(year, month, day_end)
            print(f"    Fetching games {start} to {end}...")

            try:
                df = statcast(
                    start_dt=start.strftime("%Y-%m-%d"),
                    end_dt=end.strftime("%Y-%m-%d"),
                )
            except Exception as e:
                print(f"    Error: {e}")
                continue

            if df is None or df.empty:
                continue

            # Group by game + batter to find HR outcomes
            if "game_pk" not in df.columns or "batter" not in df.columns:
                continue

            game_batters = df.groupby(["game_pk", "batter"]).agg({
                "events": lambda x: "home_run" in x.values,
                "pitcher": "first",
            }).reset_index()

            for _, row in game_batters.iterrows():
                bid = int(row["batter"])
                pid = int(row["pitcher"])
                hit_hr = int(row["events"])

                b_score = batter_score_map.get((bid, year))
                p_score = pitcher_score_map.get((pid, year))

                if b_score is None or p_score is None:
                    continue

                batter_scores.append(b_score)
                pitcher_scores.append(p_score)
                env_scores.append(1.0)  # Neutral env for training
                hr_outcomes.append(hit_hr)

    return {
        "batter_scores": batter_scores,
        "pitcher_scores": pitcher_scores,
        "env_scores": env_scores,
        "hr_outcomes": hr_outcomes,
    }


if __name__ == "__main__":
    result = train_full_pipeline()
    if "error" not in result:
        print("\n=== Summary ===")
        print(f"Seasons: {result['seasons']}")
        print(f"Batter-seasons: {result['batter_seasons']}")
        print(f"Pitcher-seasons: {result['pitcher_seasons']}")
        print(f"Training samples: {result['training_samples']}")
        print(f"HR rate: {result['hr_rate']:.4f}")
        print(f"Coefficients: {result['combiner']['coefficients']}")
