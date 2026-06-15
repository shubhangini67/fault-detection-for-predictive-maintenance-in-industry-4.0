"""FastAPI backend for industrial fault detection."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.services.fault_detection_service import FaultDetectionService

app = FastAPI(
    title="Industrial Fault Detection API",
    description="Edge AI predictive maintenance — train, predict, quantize",
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

svc = FaultDetectionService()


class PredictRequest(BaseModel):
    scenario: str = Field(..., example="IR - 7")
    engine: str = Field("Random Forest", example="Random Forest")
    offset: int = Field(0, ge=0)
    buffer_size: int = Field(400, ge=50, le=2000)


class TrainRequest(BaseModel):
    sample_size: int = Field(2000, ge=500, le=10000)


@app.get("/api/v1/health")
def health():
    return {"status": "ok", "service": "fault-detection-api"}


@app.get("/api/v1/system")
def system_status():
    return svc.system_status()


@app.post("/api/v1/predict")
def predict(body: PredictRequest):
    try:
        result = svc.predict(body.scenario, body.engine, body.offset, body.buffer_size)
        return result.to_dict()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/v1/telemetry/{scenario}")
def telemetry(scenario: str, offset: int = 0, size: int = 400):
    df = svc.get_telemetry(scenario, offset, size)
    return df.to_dict(orient="records")


@app.post("/api/v1/train/{scenario}")
def train_scenario(scenario: str, body: TrainRequest | None = None):
    size = body.sample_size if body else 2000
    try:
        return svc.train_scenario(scenario, size)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/v1/train-all")
def train_all(body: TrainRequest | None = None):
    size = body.sample_size if body else 2000
    return {"trained": svc.train_all_scenarios(size)}


@app.get("/api/v1/quantization/{scenario}")
def quantization(scenario: str):
    report = svc.quantization_report(scenario)
    if not report:
        raise HTTPException(status_code=404, detail="No quantization report")
    return report
