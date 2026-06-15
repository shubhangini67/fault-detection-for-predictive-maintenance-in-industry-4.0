"""Configuration loader for the industrial fault detection system."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "config" / "settings.yaml"


def load_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or CONFIG_PATH
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(relative: str) -> Path:
    return ROOT_DIR / relative
