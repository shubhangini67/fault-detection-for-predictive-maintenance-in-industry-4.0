"""Model deployment registry — track which edge scenarios have trained artifacts."""

from __future__ import annotations

from src.config import resolve_path
from src.data_loader import list_fault_scenarios
from src.train_models import _scenario_prefix


def rf_artifact_path(scenario: str):
    return resolve_path("models") / f"rf_{_scenario_prefix(scenario)}.joblib"


def models_exist(scenario: str) -> bool:
    return rf_artifact_path(scenario).exists()


def list_deployed_scenarios() -> list[str]:
    return [s for s in list_fault_scenarios() if models_exist(s)]


def deployment_status() -> dict[str, bool]:
    return {s: models_exist(s) for s in list_fault_scenarios()}
