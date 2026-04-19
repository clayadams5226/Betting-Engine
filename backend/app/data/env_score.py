"""Physics-informed environmental score from park factor, weather, and roof status."""

import math

from .weather import compute_wind_to_cf


def compute_env_score(
    park_hr_factor: int,
    weather: dict,
    cf_bearing_deg: float,
    roof: str,
    roof_closed: bool = False,
) -> dict:
    """Compute environmental modifier for HR probability.

    Env_Score = Park_adj * (1 + Temp_adj + Wind_adj + Humidity_adj + Roof_adj)

    Args:
        park_hr_factor: Park HR factor (100 = league average).
        weather: Weather dict from weather.py.
        cf_bearing_deg: Compass bearing from home plate to CF.
        roof: Roof type: "open", "retractable", or "dome".
        roof_closed: Whether retractable roof is closed.

    Returns dict with env_score and component breakdowns.
    """
    park_adj = park_hr_factor / 100.0

    temp_f = weather.get("temp_f", 72.0)
    humidity_pct = weather.get("humidity_pct", 50.0)
    wind_speed = weather.get("wind_speed_mph", 0.0)
    wind_deg = weather.get("wind_deg", 0)

    # Temperature adjustment: ~2% per 10 degrees from neutral (72F)
    temp_adj = 0.002 * (temp_f - 72.0)

    # Wind adjustment: component toward CF * 0.003
    if roof == "dome" or (roof == "retractable" and roof_closed):
        wind_to_cf = 0.0
        wind_adj = 0.0
    else:
        wind_to_cf = compute_wind_to_cf(wind_speed, wind_deg, cf_bearing_deg)
        wind_adj = wind_to_cf * 0.003

    # Humidity adjustment: humid air is less dense
    humidity_adj = 0.001 * (humidity_pct - 50.0)

    # Roof adjustment
    if roof == "dome":
        roof_adj = -0.03
    elif roof == "retractable" and roof_closed:
        roof_adj = -0.02
    else:
        roof_adj = 0.0

    env_score = park_adj * (1.0 + temp_adj + wind_adj + humidity_adj + roof_adj)

    return {
        "env_score": env_score,
        "park_adj": park_adj,
        "temp_adj": temp_adj,
        "wind_adj": wind_adj,
        "wind_to_cf_mph": wind_to_cf if roof != "dome" else 0.0,
        "humidity_adj": humidity_adj,
        "roof_adj": roof_adj,
        "temp_f": temp_f,
        "humidity_pct": humidity_pct,
        "wind_speed_mph": wind_speed,
        "wind_deg": wind_deg,
    }
