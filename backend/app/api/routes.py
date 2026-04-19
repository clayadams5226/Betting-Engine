"""API routes for HR predictions."""

from datetime import date, datetime

from fastapi import APIRouter, HTTPException, Query

from ..models.pipeline import HRPredictionPipeline

router = APIRouter()

# Initialize pipeline (singleton)
_pipeline = None


def _get_pipeline() -> HRPredictionPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = HRPredictionPipeline()
        _pipeline.load_models()
    return _pipeline


@router.get("/predictions/{game_date}")
def get_predictions(game_date: str):
    """Get HR predictions for all games on a date.

    Date format: YYYY-MM-DD

    Returns games with batters sorted by HR probability, grouped by team.
    """
    try:
        parsed_date = date.fromisoformat(game_date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    pipeline = _get_pipeline()

    try:
        games = pipeline.predict_day(parsed_date)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    # Group batters by team within each game
    for game in games:
        teams = {}
        for batter in game.get("batters", []):
            team = batter.get("team", "UNK")
            if team not in teams:
                teams[team] = []
            teams[team].append(batter)

        # Sort each team's batters by probability
        for team in teams:
            teams[team].sort(key=lambda x: x["hr_probability"], reverse=True)

        game["batters_by_team"] = teams

    return {
        "date": game_date,
        "game_count": len(games),
        "games": games,
    }


@router.get("/games/{game_date}")
def get_games(game_date: str):
    """Get MLB schedule with probable pitchers for a date."""
    try:
        parsed_date = date.fromisoformat(game_date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    from ..data.loader import get_schedule

    try:
        schedule = get_schedule(parsed_date)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Schedule fetch failed: {str(e)}")

    return {
        "date": game_date,
        "game_count": len(schedule),
        "games": schedule,
    }


@router.get("/top-picks/{game_date}")
def get_top_picks(
    game_date: str,
    limit: int = Query(default=20, ge=1, le=100),
):
    """Get top N batters most likely to hit a HR across all games on a date."""
    try:
        parsed_date = date.fromisoformat(game_date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    pipeline = _get_pipeline()

    try:
        games = pipeline.predict_day(parsed_date)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    # Flatten all batters across all games
    all_batters = []
    for game in games:
        for batter in game.get("batters", []):
            batter_with_game = {
                **batter,
                "venue": game.get("venue", ""),
                "game_id": game.get("game_id"),
                "home_team": game.get("home_abbr", ""),
                "away_team": game.get("away_abbr", ""),
            }
            all_batters.append(batter_with_game)

    # Sort by probability and take top N
    all_batters.sort(key=lambda x: x["hr_probability"], reverse=True)

    return {
        "date": game_date,
        "total_batters": len(all_batters),
        "top_picks": all_batters[:limit],
    }
