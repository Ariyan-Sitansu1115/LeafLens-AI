"""Carbon footprint utilities for irrigation motor usage."""

from __future__ import annotations

from datetime import datetime
from threading import Lock
from typing import Any, Dict

MOTOR_POWER_KW = 0.75
EMISSION_FACTOR_KG_PER_KWH = 0.82
HIGH_SOIL_MOISTURE_THRESHOLD = 70.0
HIGH_MOTOR_USAGE_HOURS_THRESHOLD = 2.0

# In-memory totals for lightweight analytics (process-local, no DB).
_tracking_lock = Lock()
_carbon_tracking = {
    "daily_date": None,
    "daily_total_co2": 0.0,
    "monthly_key": None,
    "monthly_total_co2": 0.0,
}


def _to_non_negative_float(value: Any, fallback: float = 0.0) -> float:
    """Safely coerce input to non-negative float."""
    try:
        parsed = float(value)
        if parsed < 0:
            return fallback
        return parsed
    except (TypeError, ValueError):
        return fallback


def _classify_co2_status(co2_kg: float) -> str:
    """Classify carbon emission status for dashboard display."""
    if co2_kg < 1.0:
        return "Low"
    if co2_kg <= 2.0:
        return "Moderate"
    return "High"


def _compute_eco_score(co2_kg: float) -> float:
    """Eco score in [0, 100] where lower emissions produce higher score."""
    score = 100.0 - (co2_kg * 10.0)
    return max(0.0, min(100.0, score))


def _build_recommendation(status: str, soil_moisture: float, usage_hours: float) -> str:
    """Return intelligent recommendation using moisture, runtime, and CO2 status."""
    if (
        soil_moisture >= HIGH_SOIL_MOISTURE_THRESHOLD
        and usage_hours >= HIGH_MOTOR_USAGE_HOURS_THRESHOLD
    ):
        return "Reduce irrigation, soil already has sufficient moisture"

    if status == "High":
        return "High carbon emission detected, optimize irrigation schedule"
    if status == "Low":
        return "Good sustainable farming practice"
    return "Optimize water usage"


def _update_running_totals(co2_kg: float) -> Dict[str, float]:
    """Update and return date-aware daily/monthly CO2 totals safely."""
    now = datetime.utcnow()
    day_key = now.date().isoformat()
    month_key = f"{now.year:04d}-{now.month:02d}"

    with _tracking_lock:
        if _carbon_tracking["daily_date"] != day_key:
            _carbon_tracking["daily_date"] = day_key
            _carbon_tracking["daily_total_co2"] = 0.0

        if _carbon_tracking["monthly_key"] != month_key:
            _carbon_tracking["monthly_key"] = month_key
            _carbon_tracking["monthly_total_co2"] = 0.0

        _carbon_tracking["daily_total_co2"] += co2_kg
        _carbon_tracking["monthly_total_co2"] += co2_kg

        return {
            "daily_total_co2": round(_carbon_tracking["daily_total_co2"], 3),
            "monthly_total_co2": round(_carbon_tracking["monthly_total_co2"], 3),
        }


def calculate_carbon_footprint(
    motor_usage_hours: Any,
    soil_moisture: Any | None = None,
) -> Dict[str, Any]:
    """Calculate electricity usage and CO2 emissions from motor runtime.

    Formula:
    - electricity_kwh = power_kw * time_hours
    - co2_kg = electricity_kwh * emission_factor
    """
    usage_hours = _to_non_negative_float(motor_usage_hours, fallback=0.0)
    electricity_kwh = MOTOR_POWER_KW * usage_hours
    co2_kg = electricity_kwh * EMISSION_FACTOR_KG_PER_KWH
    moisture_value = _to_non_negative_float(soil_moisture, fallback=0.0)
    status = _classify_co2_status(co2_kg)
    eco_score = _compute_eco_score(co2_kg)
    totals = _update_running_totals(co2_kg)

    return {
        "electricity_kwh": round(electricity_kwh, 3),
        "co2_kg": round(co2_kg, 3),
        "status": status,
        "recommendation": _build_recommendation(status, moisture_value, usage_hours),
        "daily_total_co2": totals["daily_total_co2"],
        "monthly_total_co2": totals["monthly_total_co2"],
        "eco_score": round(eco_score, 2),
    }
