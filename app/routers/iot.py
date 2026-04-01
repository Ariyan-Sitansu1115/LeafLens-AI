"""
IoT Device Integration Router – ESP32 Sensor Data Collection.

Handles real-time sensor data from ESP32 IoT devices including:
- Temperature readings
- Humidity levels
- Soil moisture content
- Plant stress indices
- Integration with IoT-driven, rule-based 3-day forecasting

Data validation, rule-based forecasting, and logging for dashboard display.
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse

from app.services.irrigation_advice_service import (
    get_irrigation_advice,
    GeminiAPIError,
    IrrigationAdviceError
)
from app.services.insight_forecast_service import (
    normalize_sensor_data,
    build_3_day_forecast,
    build_rain_prediction,
)
from app.services.carbon_footprint_service import calculate_carbon_footprint

logger = logging.getLogger("leaflens")


class SensorDataPayload(BaseModel):
    """Pydantic model for ESP32 sensor data validation."""
    
    temperature: float = Field(..., description="Temperature in Celsius")
    humidity: float = Field(..., description="Humidity percentage (0-100)")
    soil_moisture: float = Field(..., description="Soil moisture level (0-100)")
    stress_index: float = Field(..., description="Plant stress index (0-100). Calculated as: (temperature / 40.0) * (100 - soil_moisture)")
    motor_usage_hours: Optional[float] = Field(
        default=None,
        description="Optional irrigation motor ON time in hours",
    )
    motor_usage_seconds: Optional[float] = Field(
        default=None,
        description="Optional irrigation motor ON time in seconds",
    )
    pump_status: Optional[str] = Field(
        default="OFF",
        description='Optional pump status string: "ON" or "OFF"',
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "temperature": 28.5,
                "humidity": 65.3,
                "soil_moisture": 58.2,
                "stress_index": 72.5,
                "motor_usage_hours": 1.5,
                "pump_status": "ON",
            }
        }


router = APIRouter(prefix="/api", tags=["iot", "sensor"])

# In-memory latest sensor snapshot for dashboard consumption.
# Structure: {"temperature": float, "humidity": float, "soil_moisture": float, "stress_index": float}
latest_sensor_data: Dict[str, Any] = {}


def _normalize_pump_status(raw_status: Any) -> str:
    """Normalize pump status to ON/OFF with safe OFF fallback."""
    if isinstance(raw_status, str) and raw_status.strip().upper() == "ON":
        return "ON"
    return "OFF"


def _extract_motor_usage_hours(sensor_payload: Dict[str, Any]) -> Optional[float]:
    """Extract motor runtime in hours from available payload keys.

    Supports either `motor_usage_hours` or `motor_usage_seconds`.
    Returns None when no valid usage value is available.
    """
    if not sensor_payload:
        return None

    raw_hours = sensor_payload.get("motor_usage_hours")
    if raw_hours is not None:
        try:
            hours = float(raw_hours)
            if hours >= 0:
                return hours
        except (TypeError, ValueError):
            return None

    raw_seconds = sensor_payload.get("motor_usage_seconds")
    if raw_seconds is not None:
        try:
            seconds = float(raw_seconds)
            if seconds >= 0:
                return seconds / 3600.0
        except (TypeError, ValueError):
            return None

    return None


@router.post("/sensor-data")
async def receive_sensor_data(payload: SensorDataPayload) -> JSONResponse:
    """
    POST /api/sensor-data
    
    Receive and validate real-time sensor data from ESP32 IoT device.
    Stores sensor snapshot in-memory for dashboard consumption.
    Rule-based forecast is generated separately via GET /api/insight-data.
    
    Request body (JSON):
    ```json
    {
        "temperature": <float>,
        "humidity": <float>,
        "soil_moisture": <float>,
        "stress_index": <float>
    }
    ```
    
    Returns (200):
    ```json
    {
        "status": "success",
        "message": "Sensor data received successfully",
        "timestamp": "2026-02-23T14:30:45.123456"
    }
    ```
    
    Args:
        payload: SensorDataPayload object with validated sensor readings
    
    Returns:
        JSONResponse with status, message, and timestamp
    
    Raises:
        HTTPException: If validation fails (422) or unexpected error occurs (500)
    """
    try:
        timestamp = datetime.utcnow().isoformat()
        
        # Log incoming sensor data
        logger.info(
            f"IoT Sensor Data Received - "
            f"Temperature: {payload.temperature}°C, "
            f"Humidity: {payload.humidity}%, "
            f"Soil Moisture: {payload.soil_moisture}%, "
            f"Stress Index: {payload.stress_index}, "
            f"Motor Usage (h): {payload.motor_usage_hours}, "
            f"Motor Usage (s): {payload.motor_usage_seconds} | "
            f"Pump Status: {_normalize_pump_status(payload.pump_status)} | "
            f"Timestamp: {timestamp}"
        )
        
        # Store sensor snapshot for dashboard consumption
        try:
            latest_sensor_data.clear()
            latest_sensor_data.update(
                {
                    "temperature": float(payload.temperature),
                    "humidity": float(payload.humidity),
                    "soil_moisture": float(payload.soil_moisture),
                    "stress_index": float(payload.stress_index),
                    "motor_usage_hours": payload.motor_usage_hours,
                    "motor_usage_seconds": payload.motor_usage_seconds,
                    "pump_status": _normalize_pump_status(payload.pump_status),
                }
            )
            logger.debug("Updated latest_sensor_data in-memory snapshot")

        except Exception:
            # Do not interrupt device ingestion on storage error; log and continue
            logger.exception("Failed updating latest_sensor_data snapshot")

        response_data = {
            "status": "success",
            "message": "Sensor data received successfully",
            "timestamp": timestamp,
        }

        return JSONResponse(status_code=200, content=response_data)
        
    except ValueError as e:
        logger.error(f"Validation error in sensor data: {e}")
        raise HTTPException(
            status_code=422,
            detail=f"Invalid sensor data: {str(e)}"
        )
    
    except Exception as e:
        logger.exception(f"Unexpected error processing sensor data: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error processing sensor data"
        )


@router.get("/insight-data")
async def get_insight_data() -> JSONResponse:
    """
    GET /api/insight-data

    Returns the latest sensor snapshot combined with a 3-day IoT rule-based forecast.

    Response (200):
    {
      "sensor": { ... },
      "forecast": [ ... ],
      "rain_prediction": { "chance": "Low|Moderate|High", "probability": 0-100 }
    }
    """
    try:
        sensor_data = normalize_sensor_data(latest_sensor_data)
        forecast = build_3_day_forecast(sensor_data)
        rain_prediction = build_rain_prediction(sensor_data)
        motor_usage_hours = _extract_motor_usage_hours(latest_sensor_data)

        response = {
            "sensor": sensor_data,
            "forecast": forecast,
            "rain_prediction": rain_prediction,
            "pump_status": latest_sensor_data.get("pump_status", "OFF"),
            "timestamp": datetime.utcnow().isoformat(),
        }

        if motor_usage_hours is not None:
            carbon_footprint = calculate_carbon_footprint(
                motor_usage_hours,
                soil_moisture=sensor_data.get("soil_moisture"),
            )
            response["carbon_footprint"] = carbon_footprint
            logger.info(
                "Carbon footprint computed | motor_hours=%.3f electricity_kwh=%.3f co2_kg=%.3f status=%s eco_score=%.2f daily_total=%.3f monthly_total=%.3f",
                motor_usage_hours,
                carbon_footprint["electricity_kwh"],
                carbon_footprint["co2_kg"],
                carbon_footprint["status"],
                carbon_footprint.get("eco_score", 0.0),
                carbon_footprint.get("daily_total_co2", 0.0),
                carbon_footprint.get("monthly_total_co2", 0.0),
            )

        logger.info(
            "Providing insight data | temp=%.1f humidity=%.1f soil=%.1f stress=%.1f",
            sensor_data["temperature"],
            sensor_data["humidity"],
            sensor_data["soil_moisture"],
            sensor_data["stress_index"],
        )

        return JSONResponse(status_code=200, content=response)

    except Exception as e:
        logger.exception(f"Unexpected error in get_insight_data: {e}")
        return JSONResponse(
            status_code=200,
            content={
                "sensor": {
                    "temperature": 0.0,
                    "humidity": 0.0,
                    "soil_moisture": 0.0,
                    "stress_index": 0.0,
                },
                "forecast": build_3_day_forecast({}),
                "rain_prediction": build_rain_prediction({}),
                "pump_status": "OFF",
                "timestamp": datetime.utcnow().isoformat(),
            },
        )


@router.get("/irrigation-advice")
async def irrigation_advice(location: str = Query(..., description="City or location name")):
    """
    GET /api/irrigation-advice?location=CityName

    Generates smart irrigation advice using Gemini LLM
    based on latest sensor snapshot and 3-day IoT rule-based forecast.
    """

    try:
        logger.info(f"Irrigation advice requested for location: {location}")

        # 1️⃣ Ensure sensor snapshot exists
        if not latest_sensor_data:
            msg = "No sensor data available. Ingest data via POST /api/sensor-data first."
            logger.warning(msg)
            raise HTTPException(status_code=404, detail=msg)

        # 2️⃣ Build IoT forecast locally (no external weather dependency)
        forecast = build_3_day_forecast(latest_sensor_data)

        # 3️⃣ Call Gemini Irrigation Service
        try:
            advice = await get_irrigation_advice(
                sensor_data=latest_sensor_data.copy(),
                forecast_3_days=forecast,
                location=location
            )
        except GeminiAPIError as e:
            logger.error(f"Gemini API error: {e}")
            raise HTTPException(status_code=503, detail=str(e))
        except IrrigationAdviceError as e:
            logger.error(f"Irrigation advice error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

        # 4️⃣ Return structured response
        response = {
            "status": "success",
            "location": location,
            "irrigation_advice": advice,
            "timestamp": datetime.utcnow().isoformat()
        }

        logger.info(
            f"Irrigation advice generated for {location} | "
            f"Required: {advice['irrigation_required']} | "
            f"Urgency: {advice['urgency']}"
        )

        return JSONResponse(status_code=200, content=response)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error in irrigation_advice route: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
