"""Data loader for Statcast (pybaseball) and MLB Stats API."""

import json
import math
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
import statsapi
from pybaseball import statcast

STADIUMS_PATH = Path(__file__).parent / "stadiums.json"
PRIOR_YEAR_BATTERS_PATH = Path(__file__).parent / "prior_year_batters.json"

# Cache for prior-year batter stats (loaded once)
_prior_year_batters: dict | None = None


def _get_prior_year_batter(player_id: int) -> dict | None:
    """Get a batter's prior-year stats as regression target.

    Caps values at realistic maximums to prevent garbage small-sample data
    from producing absurd predictions (e.g., 100% barrel rate from 1 BIP).
    """
    global _prior_year_batters
    if _prior_year_batters is None:
        if PRIOR_YEAR_BATTERS_PATH.exists():
            with open(PRIOR_YEAR_BATTERS_PATH) as f:
                _prior_year_batters = json.load(f)
        else:
            _prior_year_batters = {}

    prior = _prior_year_batters.get(str(player_id))
    if prior is None:
        return None

    # Cap at realistic maximums (no batter sustains above these)
    caps = {
        "barrel_pct": 0.30,       # Judge-level max ~27%
        "hard_hit_pct": 0.65,     # elite max ~62%
        "avg_ev": 97.0,           # elite max ~96
        "max_ev": 122.0,          # Stanton-level
        "sweet_spot_pct": 0.50,   # realistic ceiling
    }
    capped = dict(prior)
    for key, max_val in caps.items():
        if key in capped and capped[key] > max_val:
            capped[key] = max_val

    return capped

# Statcast columns we need from pitch-level data
BATTED_BALL_COLS = [
    "game_date", "batter", "pitcher", "stand", "p_throws",
    "events", "launch_speed", "launch_angle", "barrel",
    "estimated_slg_using_speedangle", "estimated_woba_using_speedangle",
    "bb_type", "game_pk",
]


def load_stadiums() -> dict:
    """Load stadium data keyed by team abbreviation."""
    with open(STADIUMS_PATH) as f:
        data = json.load(f)
    return {s["team"]: s for s in data["stadiums"]}


def get_schedule(game_date: date) -> list[dict]:
    """Get MLB schedule with probable pitchers for a date.

    Uses raw MLB Stats API with hydration to get pitcher IDs (the Python
    statsapi wrapper doesn't return them).
    """
    date_str = game_date.strftime("%Y-%m-%d")
    resp = requests.get(
        "https://statsapi.mlb.com/api/v1/schedule",
        params={"sportId": 1, "date": date_str, "hydrate": "probablePitcher,team"},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()

    game_list = data.get("dates", [{}])[0].get("games", [])
    games = []

    for g in game_list:
        home = g.get("teams", {}).get("home", {})
        away = g.get("teams", {}).get("away", {})
        hp = home.get("probablePitcher", {})
        ap = away.get("probablePitcher", {})

        home_team = home.get("team", {})
        away_team = away.get("team", {})

        game = {
            "game_id": g.get("gamePk"),
            "home": home_team.get("name", ""),
            "home_abbr": home_team.get("abbreviation", _team_abbr(home_team.get("name", ""))),
            "home_id": home_team.get("id"),
            "away": away_team.get("name", ""),
            "away_abbr": away_team.get("abbreviation", _team_abbr(away_team.get("name", ""))),
            "away_id": away_team.get("id"),
            "time": g.get("gameDate", ""),
            "venue": g.get("venue", {}).get("name", ""),
            "home_pitcher_id": hp.get("id"),
            "home_pitcher_name": hp.get("fullName", "TBD"),
            "away_pitcher_id": ap.get("id"),
            "away_pitcher_name": ap.get("fullName", "TBD"),
            "home_pitcher_hand": hp.get("pitchHand", {}).get("code", "R"),
            "away_pitcher_hand": ap.get("pitchHand", {}).get("code", "R"),
        }
        games.append(game)

    return games


def get_lineup(game_id: int) -> dict:
    """Get lineups for a game. Returns {home: [player_dicts], away: [player_dicts]}.

    Tries multiple sources in order:
    1. Live game feed (works for in-progress/completed games and posted lineups)
    2. Boxscore data (completed games)
    """
    # Try live feed first (works for posted lineups + in-progress + completed)
    result = _get_lineup_from_live_feed(game_id)
    if result["home"] or result["away"]:
        return result

    # Try boxscore (completed games)
    try:
        boxscore = statsapi.boxscore_data(game_id)
        home_batters = []
        away_batters = []

        for pid_str, pdata in boxscore.get("homeBatters", {}).items():
            if pid_str != "0" and isinstance(pdata, dict):
                home_batters.append({
                    "id": int(pdata.get("personId", 0)),
                    "name": pdata.get("name", ""),
                    "batting_hand": pdata.get("batSide", {}).get("code", "R"),
                    "position": pdata.get("position", {}).get("abbreviation", ""),
                    "batting_order": pdata.get("battingOrder", ""),
                })

        for pid_str, pdata in boxscore.get("awayBatters", {}).items():
            if pid_str != "0" and isinstance(pdata, dict):
                away_batters.append({
                    "id": int(pdata.get("personId", 0)),
                    "name": pdata.get("name", ""),
                    "batting_hand": pdata.get("batSide", {}).get("code", "R"),
                    "position": pdata.get("position", {}).get("abbreviation", ""),
                    "batting_order": pdata.get("battingOrder", ""),
                })

        if home_batters or away_batters:
            return {"home": home_batters, "away": away_batters}
    except Exception:
        pass

    return {"home": [], "away": []}


def _get_lineup_from_live_feed(game_id: int) -> dict:
    """Get lineup from the live game feed API."""
    try:
        url = f"https://statsapi.mlb.com/api/v1.1/game/{game_id}/feed/live"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        home_batters = []
        away_batters = []

        game_data = data.get("gameData", {})
        players = game_data.get("players", {})

        live_data = data.get("liveData", {})
        boxscore = live_data.get("boxscore", {})

        for side, batter_list in [("home", home_batters), ("away", away_batters)]:
            team_info = boxscore.get("teams", {}).get(side, {})
            batting_order = team_info.get("battingOrder", [])

            for pid in batting_order:
                player_key = f"ID{pid}"
                player = players.get(player_key, {})
                batter_list.append({
                    "id": pid,
                    "name": player.get("fullName", ""),
                    "batting_hand": player.get("batSide", {}).get("code", "R"),
                    "position": player.get("primaryPosition", {}).get("abbreviation", ""),
                })

        return {"home": home_batters, "away": away_batters}
    except Exception:
        return {"home": [], "away": []}


def get_team_roster(team_id: int) -> list[dict]:
    """Get a team's active roster with position players only.

    Used as fallback when lineups aren't posted yet.
    Uses hydrate=person to get batting hand in one request.
    """
    try:
        url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster"
        resp = requests.get(url, params={
            "rosterType": "active",
            "hydrate": "person",
        }, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        batters = []
        pitcher_positions = {"P", "TWP"}

        for entry in data.get("roster", []):
            person = entry.get("person", {})
            position = entry.get("position", {})
            pos_abbr = position.get("abbreviation", "")

            # Skip pitchers
            if pos_abbr in pitcher_positions:
                continue

            # Get batting hand from hydrated person data
            bat_side = person.get("batSide", {}).get("code", "R")

            batters.append({
                "id": person.get("id"),
                "name": person.get("fullName", ""),
                "batting_hand": bat_side,
                "position": pos_abbr,
                "batting_order": "",
            })

        return batters
    except Exception:
        return []


def get_lineup_with_roster_fallback(game_id: int, home_team_id: int, away_team_id: int) -> dict:
    """Get lineups, falling back to active roster if lineups aren't posted.

    This is the primary function to call for upcoming games.
    """
    result = get_lineup(game_id)

    if not result["home"]:
        result["home"] = get_team_roster(home_team_id)

    if not result["away"]:
        result["away"] = get_team_roster(away_team_id)

    return result


def _get_batting_hand(player_id: Optional[int]) -> str:
    """Look up a player's batting hand."""
    if not player_id:
        return "R"
    try:
        url = f"https://statsapi.mlb.com/api/v1/people/{player_id}"
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        people = data.get("people", [])
        if people:
            return people[0].get("batSide", {}).get("code", "R")
    except Exception:
        pass
    return "R"


def fetch_statcast_range(start: date, end: date) -> pd.DataFrame:
    """Fetch Statcast pitch-level data for a date range.

    Returns only batted ball events with relevant columns.
    """
    df = statcast(
        start_dt=start.strftime("%Y-%m-%d"),
        end_dt=end.strftime("%Y-%m-%d"),
    )
    if df is None or df.empty:
        return pd.DataFrame(columns=BATTED_BALL_COLS)

    # Filter to balls in play only (exclude fouls which have launch_speed but no bb_type)
    batted = df[df["launch_speed"].notna() & df["bb_type"].notna()].copy()
    # Keep only columns we need
    available = [c for c in BATTED_BALL_COLS if c in batted.columns]
    batted = batted[available].copy()

    # Compute derived fields
    if "launch_speed" in batted.columns:
        batted["hard_hit"] = batted["launch_speed"] >= 95.0
        batted["sweet_spot"] = batted["launch_angle"].between(8, 32)
    if "barrel" not in batted.columns:
        batted["barrel"] = _compute_barrel(batted)

    return batted


def compute_batter_stats(
    batted_balls: pd.DataFrame,
    batter_id: int,
    vs_hand: Optional[str] = None,
) -> dict:
    """Compute aggregate batter stats from batted ball data.

    Applies regression toward league average based on sample size.
    With 0 BIP returns league average. With small samples, blends
    observed stats with league average using: weight = BIP / (BIP + REGRESS_BIP).

    Args:
        batted_balls: Pitch-level Statcast data with batted balls.
        batter_id: MLB player ID.
        vs_hand: Filter to only ABs vs this pitcher hand ('L' or 'R'). None = all.

    Returns dict with: barrel_pct, hard_hit_pct, avg_ev, max_ev,
        avg_launch_angle, sweet_spot_pct, pa_count.
    """
    df = batted_balls[batted_balls["batter"] == batter_id].copy()
    if vs_hand and "p_throws" in df.columns:
        df = df[df["p_throws"] == vs_hand]

    n = len(df)
    if n == 0:
        return _league_avg_batter_stats()

    raw = {
        "barrel_pct": df["barrel"].sum() / n if "barrel" in df.columns else 0.0,
        "hard_hit_pct": df["hard_hit"].sum() / n if "hard_hit" in df.columns else 0.0,
        "avg_ev": df["launch_speed"].mean() if "launch_speed" in df.columns else 0.0,
        "max_ev": df["launch_speed"].max() if "launch_speed" in df.columns else 0.0,
        "avg_launch_angle": df["launch_angle"].mean() if "launch_angle" in df.columns else 0.0,
        "sweet_spot_pct": df["sweet_spot"].sum() / n if "sweet_spot" in df.columns else 0.0,
        "pa_count": n,
    }

    # Regress toward prior-year stats (or league avg if no prior data)
    return _regress_batter_stats(raw, n, batter_id)


def compute_pitcher_stats(
    batted_balls: pd.DataFrame,
    pitcher_id: int,
    vs_hand: Optional[str] = None,
) -> dict:
    """Compute aggregate pitcher stats from batted ball data.

    Applies regression toward league average for small samples.

    Args:
        batted_balls: Pitch-level Statcast data with batted balls.
        pitcher_id: MLB player ID.
        vs_hand: Filter to only ABs vs this batter hand ('L' or 'R'). None = all.

    Returns dict with: hr_per_9, hr_fb_pct, barrel_pct_allowed,
        hard_hit_pct_allowed, xslg_against, xwoba_against, pa_count.
    """
    df = batted_balls[batted_balls["pitcher"] == pitcher_id].copy()
    if vs_hand and "stand" in df.columns:
        df = df[df["stand"] == vs_hand]

    n = len(df)
    if n == 0:
        return _league_avg_pitcher_stats()

    hr_count = df["events"].eq("home_run").sum() if "events" in df.columns else 0
    fb_count = df["bb_type"].isin(["fly_ball", "popup"]).sum() if "bb_type" in df.columns else max(n, 1)

    # Estimate innings from batted balls (rough: ~3.1 batted balls per inning)
    est_innings = n / 3.1

    raw = {
        "hr_per_9": (hr_count / est_innings * 9) if est_innings > 0 else 0.0,
        "hr_fb_pct": (hr_count / fb_count) if fb_count > 0 else 0.0,
        "barrel_pct_allowed": df["barrel"].sum() / n if "barrel" in df.columns else 0.0,
        "hard_hit_pct_allowed": df["hard_hit"].sum() / n if "hard_hit" in df.columns else 0.0,
        "xslg_against": df["estimated_slg_using_speedangle"].mean() if "estimated_slg_using_speedangle" in df.columns else 0.0,
        "xwoba_against": df["estimated_woba_using_speedangle"].mean() if "estimated_woba_using_speedangle" in df.columns else 0.0,
        "pa_count": n,
    }

    return _regress_pitcher_stats(raw, n)


def apply_bayesian_shrinkage(
    split_stats: dict,
    overall_stats: dict,
    shrinkage_pa: int = 50,
) -> dict:
    """Apply Bayesian shrinkage when platoon split sample is small.

    X_blend = (PA / (PA + shrinkage_pa)) * X_split + (shrinkage_pa / (PA + shrinkage_pa)) * X_overall
    """
    pa = split_stats.get("pa_count", 0)
    if pa == 0:
        return overall_stats.copy()

    weight_split = pa / (pa + shrinkage_pa)
    weight_overall = shrinkage_pa / (pa + shrinkage_pa)

    blended = {}
    for key in split_stats:
        if key == "pa_count":
            blended[key] = pa
            continue
        split_val = split_stats.get(key, 0.0)
        overall_val = overall_stats.get(key, 0.0)
        if isinstance(split_val, (int, float)) and isinstance(overall_val, (int, float)):
            blended[key] = weight_split * split_val + weight_overall * overall_val
        else:
            blended[key] = split_val

    return blended


# --- Private helpers ---

def _compute_barrel(df: pd.DataFrame) -> pd.Series:
    """Compute barrel classification: 98+ mph EV with optimal launch angle."""
    if "launch_speed" not in df.columns or "launch_angle" not in df.columns:
        return pd.Series(0, index=df.index)

    ev = df["launch_speed"]
    la = df["launch_angle"]

    # Barrel zone: EV >= 98 mph, LA in sweet zone that expands with EV
    min_la = 26.0 - (ev - 98.0) * 0.5
    max_la = 30.0 + (ev - 98.0) * 1.0
    min_la = min_la.clip(lower=8.0)
    max_la = max_la.clip(upper=50.0)

    return ((ev >= 98.0) & (la >= min_la) & (la <= max_la)).astype(int)


# Stat-specific regression BIP thresholds.
# Rare events (barrels) need heavy regression; continuous stats (EV) stabilize fast.
_BATTER_REGRESS_BIP = {
    "barrel_pct": 80,        # rare event, very slow to stabilize
    "hard_hit_pct": 40,      # moderate
    "avg_ev": 25,            # continuous, stabilizes fast
    "max_ev": 12,            # just need a handful of hard hits
    "avg_launch_angle": 50,  # approach-driven, moderate
    "sweet_spot_pct": 55,    # moderate
}
_PITCHER_REGRESS_BIP = {
    "hr_per_9": 100,
    "hr_fb_pct": 100,
    "barrel_pct_allowed": 80,
    "hard_hit_pct_allowed": 40,
    "xslg_against": 50,
    "xwoba_against": 50,
}


def _league_avg_batter_stats() -> dict:
    """League average batter stats (used for 0 PA or as regression target)."""
    return {
        "barrel_pct": 0.080,
        "hard_hit_pct": 0.390,
        "avg_ev": 88.8,
        "max_ev": 110.5,
        "avg_launch_angle": 12.0,
        "sweet_spot_pct": 0.337,
        "pa_count": 0,
    }


def _league_avg_pitcher_stats() -> dict:
    """League average pitcher stats (used for 0 PA or as regression target)."""
    return {
        "hr_per_9": 1.25,
        "hr_fb_pct": 0.115,
        "barrel_pct_allowed": 0.078,
        "hard_hit_pct_allowed": 0.390,
        "xslg_against": 0.405,
        "xwoba_against": 0.316,
        "pa_count": 0,
    }


def _regress_batter_stats(raw: dict, n: int, batter_id: int = 0) -> dict:
    """Regress batter stats toward prior-year stats (or league avg) based on sample size.

    If we have the player's 2024 stats, use those as the regression target
    (regressed 15% toward league avg to account for year-to-year change).
    Otherwise fall back to league average.
    """
    league = _league_avg_batter_stats()

    # Use prior-year stats as regression target if available
    prior = _get_prior_year_batter(batter_id) if batter_id else None
    if prior:
        # Regress prior-year 15% toward league avg (aging/variance adjustment)
        target = {}
        for key in league:
            if key == "pa_count":
                continue
            prior_val = prior.get(key, league[key])
            target[key] = 0.85 * prior_val + 0.15 * league[key]
    else:
        target = league

    result = {"pa_count": n}
    for key in league:
        if key == "pa_count":
            continue
        observed = raw.get(key, 0.0)
        regress_val = target.get(key, league[key])
        regress_bip = _BATTER_REGRESS_BIP.get(key, 50)
        weight = n / (n + regress_bip)
        result[key] = weight * observed + (1 - weight) * regress_val

    return result


def _regress_pitcher_stats(raw: dict, n: int) -> dict:
    """Regress pitcher stats toward league average based on sample size."""
    league = _league_avg_pitcher_stats()

    result = {"pa_count": n}
    for key in league:
        if key == "pa_count":
            continue
        observed = raw.get(key, 0.0)
        avg = league[key]
        regress_bip = _PITCHER_REGRESS_BIP.get(key, 60)
        weight = n / (n + regress_bip)
        result[key] = weight * observed + (1 - weight) * avg

    return result


def _empty_batter_stats() -> dict:
    return _league_avg_batter_stats()


def _empty_pitcher_stats() -> dict:
    return _league_avg_pitcher_stats()


def _get_pitcher_hand(pitcher_id: Optional[int]) -> str:
    """Look up pitcher throwing hand from MLB Stats API."""
    if not pitcher_id:
        return "R"
    try:
        url = f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}"
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        people = data.get("people", [])
        if people:
            return people[0].get("pitchHand", {}).get("code", "R")
    except Exception:
        pass
    return "R"


# Team name -> abbreviation mapping
_TEAM_ABBR_MAP = {
    "Arizona Diamondbacks": "ARI",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",
    "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Mets": "NYM",
    "New York Yankees": "NYY",
    "Oakland Athletics": "OAK",
    "Athletics": "OAK",
    "Sacramento Athletics": "OAK",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SD",
    "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSH",
}


def _team_abbr(team_name: str) -> str:
    return _TEAM_ABBR_MAP.get(team_name, team_name[:3].upper())
