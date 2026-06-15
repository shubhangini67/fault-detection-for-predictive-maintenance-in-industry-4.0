#!/usr/bin/env python3
"""Copy trained models into deploy/artifacts for Docker and cloud deploy."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODELS = ROOT / "models"
ARTIFACTS = ROOT / "deploy" / "artifacts"


def main() -> int:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in MODELS.iterdir():
        if not src.is_file():
            continue
        if src.suffix in {".joblib", ".keras", ".tflite", ".json"}:
            shutil.copy2(src, ARTIFACTS / src.name)
            copied += 1
    print(f"Bundled {copied} artifacts to deploy/artifacts/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
