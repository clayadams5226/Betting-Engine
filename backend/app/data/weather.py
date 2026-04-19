"""OpenWeatherMap integration for game-time weather at stadium coordinates."""

import math
import os
from datetime import datetime
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

OWM_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY", "")
OWM_BASE_URL = "https://api.openweathermap.org/data/2.5"


def get_weather(
    lat: float,
    lon: float,
    game_time: Optional[datetime] = None,
) -> dict:
    """Fetch weather for a stadium location.

    Uses current weather if game_time is within 2 hours, otherwise forecast.

    Returns: {
        temp_f, humidity_pct, wind_speed_mph, wind_deg,
        description, is_forecast
    }
    """
    if not OWM_API_KEY:
        return _default_weather()

    try:
        if game_time and _hours_until(game_time) > 2:
            return _get_forecast(lat, lon, game_time)
        return _get_current(lat, lon)
    except Exception:
        return _default_weather()


def compute_wind_to_cf(
    wind_speed_mph: float,
    wind_deg: float,
    cf_bearing_deg: float,
) -> float:
    """Compute wind component blowing toward center field.

    Positive = blowing out (favorable for HR).
    Negative = blowing in (suppresses HR).

    Args:
        wind_speed_mph: Wind speed in mph.
        wind_deg: Meteorological wind direction (degrees, where wind is FROM).
        cf_bearing_deg: Compass bearing from home plate to center field.
    """
    # Meteorological convention: wind_deg is where wind comes FROM
    # Wind blows TOWARD: (wind_deg + 180) % 360
    wind_toward_deg = (wind_deg + 180) % 360

    # Angle between wind direction (toward) and CF bearing
    angle_diff = math.radians(wind_toward_deg - cf_bearing_deg)

    # Component of wind in the CF direction
    return wind_speed_mph * math.cos(angle_diff)


def _get_current(lat: float, lon: float) -> dict:
    url = f"{OWM_BASE_URL}/weather"
    resp = requests.get(url, params={
        "lat": lat,
        "lon": lon,
        "appid": OWM_API_KEY,
        "units": "imperial",
    }, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    return {
        "temp_f": data["main"]["temp"],
        "humidity_pct": data["main"]["humidity"],
        "wind_speed_mph": data["wind"]["speed"],
        "wind_deg": data["wind"].get("deg", 0),
        "description": data["weather"][0]["description"] if data.get("weather") else "",
        "is_forecast": False,
    }


def _get_forecast(lat: float, lon: float, game_time: datetime) -> dict:
    """Get 5-day/3-hour forecast and find closest time slot to game_time."""
    url = f"{OWM_BASE_URL}/forecast"
    resp = requests.get(url, params={
        "lat": lat,
        "lon": lon,
        "appid": OWM_API_KEY,
        "units": "imperial",
    }, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    target_ts = game_time.timestamp()
    best = None
    best_diff = float("inf")

    for entry in data.get("list", []):
        dt = entry["dt"]
        diff = abs(dt - target_ts)
        if diff < best_diff:
            best_diff = diff
            best = entry

    if not best:
        return _default_weather()

    return {
        "temp_f": best["main"]["temp"],
        "humidity_pct": best["main"]["humidity"],
        "wind_speed_mph": best["wind"]["speed"],
        "wind_deg": best["wind"].get("deg", 0),
        "description": best["weather"][0]["description"] if best.get("weather") else "",
        "is_forecast": True,
    }


def _default_weather() -> dict:
    """Neutral weather when API is unavailable."""
    return {
        "temp_f": 72.0,
        "humidity_pct": 50.0,
        "wind_speed_mph": 0.0,
        "wind_deg": 0,
        "description": "unknown",
        "is_forecast": False,
    }


def _hours_until(dt: datetime) -> float:
    now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
    delta = dt - now
    return delta.total_seconds() / 3600
