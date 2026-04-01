"""Rule-based forecast engine for LeafLens Insight.

This module is intentionally IoT-only and does not depend on external weather APIs.
"""

from __future__ import annotations

from typing import Any, Dict, List


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _to_float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def normalize_sensor_data(sensor_data: Dict[str, Any] | None) -> Dict[str, float]:
    """Normalize sensor payload into safe float values.

    Expected keys: temperature, humidity, soil_moisture, stress_index.
    Missing or invalid values safely fall back to 0.0.
    """
    sensor_data = sensor_data or {}
    return {
        "temperature": _to_float(sensor_data.get("temperature"), 0.0),
        "humidity": _clamp(_to_float(sensor_data.get("humidity"), 0.0), 0.0, 100.0),
        "soil_moisture": _clamp(_to_float(sensor_data.get("soil_moisture"), 0.0), 0.0, 100.0),
        "stress_index": _clamp(_to_float(sensor_data.get("stress_index"), 0.0), 0.0, 100.0),
    }


def _condition_from_humidity(humidity: float) -> str:
    if humidity > 80.0:
        return "Rain Likely"
    if humidity >= 60.0:
        return "Moderate"
    return "Dry"


def build_3_day_forecast(sensor_data: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    """Build deterministic 3-day forecast from IoT values.

    Rules:
    - Temperature increases slightly each day (+0.5 to +1.5)
    - Humidity decreases gradually
    - Condition from humidity buckets
    """
    sensor = normalize_sensor_data(sensor_data)
    base_temp = sensor["temperature"]
    base_humidity = sensor["humidity"]
    stress = sensor["stress_index"]

    forecast: List[Dict[str, Any]] = []
    next_temp = base_temp
    next_humidity = base_humidity

    for day_index in range(1, 4):
        temp_increase = _clamp(0.5 + (stress / 100.0) + ((day_index - 1) * 0.1), 0.5, 1.5)
        humidity_drop = _clamp(1.5 + ((day_index - 1) * 1.0), 1.5, 6.0)

        next_temp = next_temp + temp_increase
        next_humidity = _clamp(next_humidity - humidity_drop, 0.0, 100.0)

        forecast.append(
            {
                "day": f"Day {day_index}",
                "temperature": round(next_temp, 1),
                "humidity": round(next_humidity, 1),
                "condition": _condition_from_humidity(next_humidity),
            }
        )

    return forecast


def build_rain_prediction(sensor_data: Dict[str, Any] | None) -> Dict[str, Any]:
    """Compute rain chance/probability from weighted IoT score.

    Weighted score combines:
    - humidity (50%)
    - normalized temperature (20%)
    - soil moisture (30%)
    """
    sensor = normalize_sensor_data(sensor_data)
    humidity = sensor["humidity"]
    soil_moisture = sensor["soil_moisture"]
    temperature = sensor["temperature"]

    normalized_temp = _clamp((temperature / 50.0) * 100.0, 0.0, 100.0)

    weighted_probability = (
        (humidity * 0.5)
        + (normalized_temp * 0.2)
        + (soil_moisture * 0.3)
    )

    probability = round(_clamp(weighted_probability, 0.0, 100.0), 1)

    if probability >= 70.0:
        chance = "High"
    elif probability >= 40.0:
        chance = "Moderate"
    else:
        chance = "Low"

    return {
        "chance": chance,
        "probability": probability,
    }
