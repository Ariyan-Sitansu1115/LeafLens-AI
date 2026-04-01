#!/usr/bin/env python3
"""Basic tests for carbon footprint analyzer integration."""

from app.services.carbon_footprint_service import calculate_carbon_footprint
from app.routers.iot import _extract_motor_usage_hours


def test_low_status_threshold() -> None:
    result = calculate_carbon_footprint(1.0)
    assert result["electricity_kwh"] == 0.75
    assert result["co2_kg"] == 0.615
    assert result["status"] == "Low"


def test_moderate_status_threshold() -> None:
    result = calculate_carbon_footprint(2.0)
    assert result["co2_kg"] == 1.23
    assert result["status"] == "Moderate"


def test_high_status_threshold() -> None:
    result = calculate_carbon_footprint(4.0)
    assert result["co2_kg"] == 2.46
    assert result["status"] == "High"


def test_eco_score_and_totals_fields() -> None:
    result = calculate_carbon_footprint(2.0)
    assert "daily_total_co2" in result
    assert "monthly_total_co2" in result
    assert "eco_score" in result
    assert 0 <= result["eco_score"] <= 100


def test_eco_score_lower_bound_clamp() -> None:
    # Large runtime should clamp eco score to 0.
    result = calculate_carbon_footprint(20.0)
    assert result["eco_score"] == 0


def test_seconds_to_hours_extraction() -> None:
    payload = {"motor_usage_seconds": 5400}
    hours = _extract_motor_usage_hours(payload)
    assert hours == 1.5


def test_missing_usage_extraction() -> None:
    payload = {"temperature": 30}
    hours = _extract_motor_usage_hours(payload)
    assert hours is None


if __name__ == "__main__":
    test_low_status_threshold()
    test_moderate_status_threshold()
    test_high_status_threshold()
    test_eco_score_and_totals_fields()
    test_eco_score_lower_bound_clamp()
    test_seconds_to_hours_extraction()
    test_missing_usage_extraction()
    print("All carbon footprint tests passed.")
