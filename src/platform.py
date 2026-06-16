"""Environment detection for cloud vs local."""

from __future__ import annotations

from pathlib import Path


def on_streamlit_cloud() -> bool:
    return Path("/mount/src").exists()
