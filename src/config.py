"""Configuration loader for the industrial fault detection system."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.platform import on_streamlit_cloud

ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "config" / "settings.yaml"
CLOUD_ARTIFACTS = ROOT_DIR / "deploy" / "artifacts"


def load_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or CONFIG_PATH
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(relative: str) -> Path:
    if relative == "models" and on_streamlit_cloud() and CLOUD_ARTIFACTS.is_dir():
        return CLOUD_ARTIFACTS
    return ROOT_DIR / relative
