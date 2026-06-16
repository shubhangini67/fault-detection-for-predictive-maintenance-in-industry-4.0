"""Restore bundled deploy artifacts for cloud and container startup."""

from __future__ import annotations

import shutil
from pathlib import Path

from src.platform import on_streamlit_cloud

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "deploy" / "artifacts"
MODELS = ROOT / "models"


def ensure_deploy_artifacts() -> int:
    if on_streamlit_cloud() and ARTIFACTS.is_dir():
        return 0

    MODELS.mkdir(parents=True, exist_ok=True)
    if not ARTIFACTS.exists():
        return 0

    copied = 0
    for src in ARTIFACTS.iterdir():
        if not src.is_file():
            continue
        dest = MODELS / src.name
        if not dest.exists():
            shutil.copy2(src, dest)
            copied += 1
    return copied
