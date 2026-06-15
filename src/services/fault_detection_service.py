"""Core fault detection service — used by API and dashboard."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.bootstrap import ensure_deploy_artifacts
from src.config import load_config, resolve_path
from src.data_loader import list_fault_scenarios, load_stream_chunk
from src.model_registry import deployment_status, list_deployed_scenarios, models_exist
from src.scenario_utils import scenario_prefix


@dataclass
class PredictionResponse:
    scenario: str
    engine: str
    fault_detected: bool
    confidence: float
    model_type: str
    health_label: str
    reconstruction_error: float | None
    maintenance: dict
    engines: dict = field(default_factory=dict)
    buffer_stats: dict = field(default_factory=dict)
    analyzed_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class FaultDetectionService:
    """Single backend entry point for training, inference, and system status."""

    def __init__(self) -> None:
        self.cfg = load_config()
        ensure_deploy_artifacts()

    def system_status(self) -> dict:
        deployed = list_deployed_scenarios()
        profiles = []
        for key, prof in self.cfg["fault_profiles"].items():
            if key == "NB":
                continue
            profiles.append(
                {
                    "id": key,
                    "name": prof["display_name"],
                    "severity": prof["severity"],
                    "defect_in": prof["defect_size_in"],
                    "deployed": models_exist(key),
                }
            )
        return {
            "platform": self.cfg["project"]["name"],
            "version": self.cfg["project"]["version"],
            "edge_mode": self.cfg["edge_ai"]["inference_mode"],
            "deployed_count": len(deployed),
            "total_scenarios": len(list_fault_scenarios(self.cfg)),
            "scenarios": profiles,
            "ready": len(deployed) > 0,
        }

    def get_telemetry(self, scenario: str, offset: int = 0, size: int = 400) -> pd.DataFrame:
        return load_stream_chunk(scenario, offset=offset, chunk_size=size)

    def predict(
        self,
        scenario: str,
        engine: str = "Random Forest",
        offset: int = 0,
        buffer_size: int = 400,
    ) -> PredictionResponse:
        from src.edge_predictor import EdgeInferenceResult, EdgePredictor

        if not models_exist(scenario):
            raise FileNotFoundError(
                f"Models not deployed for '{scenario}'. Train or restore artifacts first."
            )

        chunk = self.get_telemetry(scenario, offset, buffer_size)
        predictor = EdgePredictor(scenario)
        results: dict[str, EdgeInferenceResult] = {}

        if engine in ("Random Forest", "Ensemble"):
            results["Random Forest"] = predictor.predict_random_forest(chunk)
        if engine in ("Autoencoder", "Ensemble"):
            results["Autoencoder"] = predictor.predict_autoencoder(chunk)
        if engine in ("TFLite FP16", "Ensemble"):
            try:
                results["TFLite FP16"] = predictor.predict_tflite(chunk, variant="fp16")
            except FileNotFoundError:
                pass

        primary_key = engine if engine != "Ensemble" else "Random Forest"
        primary = results.get(primary_key) or next(iter(results.values()), None)
        if primary is None:
            raise RuntimeError("No inference engine produced a result.")

        fault = primary.fault_detected
        severity = self.cfg["fault_profiles"][scenario]["severity"]
        if not fault:
            health = "Healthy"
        elif severity == "critical":
            health = "Critical"
        else:
            health = "Degraded"

        return PredictionResponse(
            scenario=scenario,
            engine=engine,
            fault_detected=fault,
            confidence=primary.confidence,
            model_type=primary.model_type,
            health_label=health,
            reconstruction_error=primary.reconstruction_error,
            maintenance=primary.maintenance,
            engines={
                name: {"fault": r.fault_detected, "confidence": round(r.confidence, 4)}
                for name, r in results.items()
            },
            buffer_stats={
                "samples": len(chunk),
                "de_mean": round(float(chunk["DE"].mean()), 4),
                "fe_mean": round(float(chunk["FE"].mean()), 4),
                "fault_ratio": round(float(chunk["Fault"].mean()), 4),
            },
            analyzed_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        )

    def train_scenario(self, scenario: str, sample_size: int = 2000) -> dict:
        from src.edge_quantization import quantize_autoencoder
        from src.train_models import train_all

        result = train_all(scenario, sample_size=sample_size)
        try:
            quantize_autoencoder(scenario, sample_size=min(sample_size, 1500))
        except FileNotFoundError:
            pass
        return {"scenario": scenario, "status": "trained", "metrics": result}

    def train_all_scenarios(self, sample_size: int = 2000) -> list[dict]:
        outputs = []
        for sc in list_fault_scenarios(self.cfg):
            outputs.append(self.train_scenario(sc, sample_size))
        return outputs

    def restore_artifacts(self) -> int:
        return ensure_deploy_artifacts()

    def quantization_report(self, scenario: str) -> dict | None:
        path = resolve_path("models") / f"quantization_{scenario_prefix(scenario)}.json"
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)
