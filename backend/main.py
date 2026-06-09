from __future__ import annotations

import csv
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import joblib
    import numpy as np
    import pandas as pd
except Exception:  # Dependencies may not be installed yet.
    joblib = None
    np = None
    pd = None

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
MODEL_PATH = BASE_DIR / "aerosat_model.pkl"
DATA_PATH = PROJECT_DIR / "data" / "final_master_training_data.csv"

LOCATION_MAP = {
    "rohini_pitampura_area": {"site_id": 1, "name": "Rohini/Pitampura Area"},
    "rohinipitampura_area": {"site_id": 1, "name": "Rohini/Pitampura Area"},
    "dwarka_sector_area": {"site_id": 2, "name": "Dwarka Sector Area"},
    "dwarka_sec_10": {"site_id": 2, "name": "Dwarka Sector Area"},
    "dwarka_sec_21": {"site_id": 2, "name": "Dwarka Sector Area"},
    "lodhi_road_area": {"site_id": 3, "name": "Lodhi Road Area"},
    "anand_vihar": {"site_id": 3, "name": "Anand Vihar"},
    "central_delhi": {"site_id": 3, "name": "Central Delhi"},
    "narela": {"site_id": 4, "name": "Narela"},
    "okhla": {"site_id": 5, "name": "Okhla"},
    "jasola": {"site_id": 5, "name": "Jasola"},
    "sarita_vihar": {"site_id": 5, "name": "Sarita Vihar"},
    "bawana": {"site_id": 6, "name": "Bawana"},
    "wazirpur": {"site_id": 7, "name": "Wazirpur"},
}

FEATURE_COLUMNS = [
    "O3_forecast",
    "NO2_forecast",
    "T_forecast",
    "NO2_satellite",
    "HCHO_satellite",
    "site_id",
    "hour",
    "day_of_week",
    "is_weekend",
    "is_rush_hour",
]


class PollutionInput(BaseModel):
    O3_forecast: float = Field(..., description="Ozone forecast value")
    NO2_forecast: float = Field(..., description="Nitrogen dioxide forecast value")
    T_forecast: float = Field(..., description="Temperature forecast")
    NO2_satellite: float = Field(..., description="Sentinel-5P NO2 value")
    HCHO_satellite: float = Field(..., description="Sentinel-5P HCHO value")
    site_id: int = Field(..., ge=1, le=7)
    hour: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    is_weekend: int = Field(..., ge=0, le=1)
    is_rush_hour: int = Field(..., ge=0, le=1)


class PredictionOutput(BaseModel):
    O3_predicted: float
    NO2_predicted: float
    aqi: int
    aqi_status: str
    model_used: str
    status: str


class ForecastOutput(PredictionOutput):
    location: str
    location_key: str
    site_id: int
    hour: int
    period: str
    inputs: Dict[str, Union[float, int]]
    generated_at: str


def _load_model() -> Tuple[Optional[Any], Dict[str, Any]]:
    if joblib is None:
        return None, {
            "status": "unavailable",
            "reason": "Install backend requirements to enable ML model loading.",
        }

    try:
        bundle = joblib.load(MODEL_PATH)
        return bundle, {"status": "loaded"}
    except Exception as exc:
        return None, {"status": "unavailable", "reason": str(exc)}


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _load_hourly_defaults() -> Dict[Tuple[int, int], Dict[str, Union[float, int]]]:
    grouped: Dict[Tuple[int, int], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    if not DATA_PATH.exists():
        return {}

    with DATA_PATH.open(newline="") as file:
        for row in csv.DictReader(file):
            try:
                key = (int(float(row["site_id"])), int(float(row["hour"])))
                for col in FEATURE_COLUMNS:
                    grouped[key][col].append(float(row[col]))
                grouped[key]["O3_target"].append(float(row["O3_target"]))
                grouped[key]["NO2_target"].append(float(row["NO2_target"]))
            except (KeyError, TypeError, ValueError):
                continue

    defaults: Dict[Tuple[int, int], Dict[str, Union[float, int]]] = {}
    for key, columns in grouped.items():
        defaults[key] = {}
        for col, values in columns.items():
            value = _mean(values)
            if col in {"site_id", "hour", "day_of_week", "is_weekend", "is_rush_hour"}:
                defaults[key][col] = int(round(value))
            else:
                defaults[key][col] = round(value, 4)
    return defaults


MODEL_BUNDLE, MODEL_STATE = _load_model()
MODEL = MODEL_BUNDLE.get("model") if isinstance(MODEL_BUNDLE, dict) else None
MODEL_FEATURE_COLUMNS = (
    MODEL_BUNDLE.get("feature_columns", FEATURE_COLUMNS)
    if isinstance(MODEL_BUNDLE, dict)
    else FEATURE_COLUMNS
)
MODEL_NAME = (
    MODEL_BUNDLE.get("model_name", MODEL.__class__.__name__)
    if isinstance(MODEL_BUNDLE, dict) and MODEL is not None
    else "dataset-baseline"
)
MODEL_METRICS = MODEL_BUNDLE.get("metrics", {}) if isinstance(MODEL_BUNDLE, dict) else {}
HOURLY_DEFAULTS = _load_hourly_defaults()

app = FastAPI(
    title="AeroSat AI Pollution Forecast API",
    description="Short-term O3 and NO2 forecast service for Delhi locations.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def is_rush_hour(hour: int) -> int:
    return int(7 <= hour <= 10 or 17 <= hour <= 20)


def period_label(hour: int) -> str:
    if 0 <= hour <= 4:
        return "MIDNIGHT"
    if 5 <= hour <= 7:
        return "DAWN"
    if 7 <= hour <= 10 or 17 <= hour <= 20:
        return "RUSH HOUR"
    if 11 <= hour <= 13:
        return "LATE MORNING"
    if 14 <= hour <= 16:
        return "AFTERNOON"
    if hour in {21, 22}:
        return "EVENING"
    return "NIGHT"


def aqi_from_pollutants(o3: float, no2: float) -> int:
    return max(0, min(500, round(50 + (o3 * 0.8) + (no2 * 1.3))))


def aqi_status(aqi: int) -> str:
    if aqi > 300:
        return "Hazardous"
    if aqi > 200:
        return "Very Unhealthy"
    if aqi > 150:
        return "Unhealthy"
    if aqi > 100:
        return "Unhealthy for SG"
    if aqi > 50:
        return "Moderate"
    return "Good"


def baseline_prediction(data: PollutionInput) -> Tuple[float, float]:
    rush_boost = 1.12 if data.is_rush_hour else 0.95
    o3_hour_factor = 0.9 + 0.18 * math.sin(max(0.0, (data.hour - 6) / 16) * math.pi)
    no2_hour_factor = rush_boost + (0.03 if data.is_weekend else 0.0)

    o3 = (data.O3_forecast * o3_hour_factor) + (data.HCHO_satellite * 0.08)
    no2 = (data.NO2_forecast * no2_hour_factor) + (data.NO2_satellite * 0.12)
    return round(o3, 4), round(no2, 4)


def model_prediction(data: PollutionInput) -> Tuple[float, float, str]:
    if MODEL is None or np is None:
        o3, no2 = baseline_prediction(data)
        return o3, no2, MODEL_NAME

    feature_values = {
        "O3_forecast": data.O3_forecast,
        "NO2_forecast": data.NO2_forecast,
        "T_forecast": data.T_forecast,
        "NO2_satellite": data.NO2_satellite,
        "HCHO_satellite": data.HCHO_satellite,
        "site_id": data.site_id,
        "hour": data.hour,
        "day_of_week": data.day_of_week,
        "is_weekend": data.is_weekend,
        "is_rush_hour": data.is_rush_hour,
    }
    ordered_values = [[feature_values[column] for column in MODEL_FEATURE_COLUMNS]]
    features = (
        pd.DataFrame(ordered_values, columns=MODEL_FEATURE_COLUMNS)
        if pd is not None
        else np.array(ordered_values)
    )

    raw = MODEL.predict(features)[0]
    if isinstance(raw, (list, tuple, np.ndarray)) and len(raw) >= 2:
        return round(float(raw[0]), 4), round(float(raw[1]), 4), MODEL_NAME

    o3, _ = baseline_prediction(data)
    return o3, round(float(raw), 4), MODEL_NAME


def default_inputs(site_id: int, hour: int) -> PollutionInput:
    today = datetime.now()
    row = HOURLY_DEFAULTS.get((site_id, hour), {})

    return PollutionInput(
        O3_forecast=float(row.get("O3_forecast", 45 + hour * 1.6)),
        NO2_forecast=float(row.get("NO2_forecast", 35 + (20 if is_rush_hour(hour) else 4))),
        T_forecast=float(row.get("T_forecast", 293 + min(hour, 14) * 0.8)),
        NO2_satellite=float(row.get("NO2_satellite", 4.8 + (1.2 if is_rush_hour(hour) else 0.0))),
        HCHO_satellite=float(row.get("HCHO_satellite", 2.2)),
        site_id=site_id,
        hour=hour,
        day_of_week=int(row.get("day_of_week", today.weekday())),
        is_weekend=int(row.get("is_weekend", today.weekday() >= 5)),
        is_rush_hour=int(row.get("is_rush_hour", is_rush_hour(hour))),
    )


def build_prediction(data: PollutionInput) -> PredictionOutput:
    o3_predicted, no2_predicted, model_used = model_prediction(data)
    aqi = aqi_from_pollutants(o3_predicted, no2_predicted)
    return PredictionOutput(
        O3_predicted=o3_predicted,
        NO2_predicted=no2_predicted,
        aqi=aqi,
        aqi_status=aqi_status(aqi),
        model_used=model_used,
        status="success",
    )


def as_dict(model: BaseModel) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "status": "AeroSat AI API is running",
        "version": app.version,
        "docs": "/docs",
        "model": MODEL_STATE,
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "model_loaded": MODEL is not None,
        "dataset_loaded": bool(HOURLY_DEFAULTS),
    }


@app.get("/locations")
def locations() -> Dict[str, List[Dict[str, Union[str, int]]]]:
    return {
        "locations": [
            {"key": key, "site_id": int(value["site_id"]), "name": str(value["name"])}
            for key, value in LOCATION_MAP.items()
        ]
    }


@app.get("/model-info")
def model_info() -> Dict[str, Any]:
    return {
        "model_name": MODEL_NAME,
        "feature_columns": MODEL_FEATURE_COLUMNS,
        "metrics": MODEL_METRICS,
        "model_state": MODEL_STATE,
        "dataset_rows_grouped": len(HOURLY_DEFAULTS),
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(data: PollutionInput) -> PredictionOutput:
    try:
        return build_prediction(data)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/forecast/{location_key}", response_model=ForecastOutput)
def forecast(
    location_key: str,
    hour: int = Query(14, ge=0, le=23),
) -> ForecastOutput:
    key = location_key.lower().strip()
    if key not in LOCATION_MAP:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown location '{location_key}'. Use /locations for valid keys.",
        )

    location = LOCATION_MAP[key]
    data = default_inputs(int(location["site_id"]), hour)
    prediction = build_prediction(data)

    return ForecastOutput(
        **as_dict(prediction),
        location=str(location["name"]),
        location_key=key,
        site_id=int(location["site_id"]),
        hour=hour,
        period=period_label(hour),
        inputs=as_dict(data),
        generated_at=datetime.now().isoformat(timespec="seconds"),
    )


@app.get("/forecast/{location_key}/daily")
def daily_forecast(location_key: str) -> Dict[str, Any]:
    key = location_key.lower().strip()
    if key not in LOCATION_MAP:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown location '{location_key}'. Use /locations for valid keys.",
        )

    return {
        "location": LOCATION_MAP[key]["name"],
        "location_key": key,
        "hours": [forecast(key, hour) for hour in range(24)],
    }
