"""Application entry — navigation shell."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Keep TensorFlow footprint small on Streamlit Cloud (1 GB RAM tier).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.bootstrap import ensure_deploy_artifacts
from dashboard.layout import inject_theme

ensure_deploy_artifacts()

st.set_page_config(
    page_title="Fault Detection Platform",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_theme()

pages = st.navigation(
    [
        st.Page(str(ROOT / "dashboard/pages/home.py"), title="Home", icon="🏠", default=True),
        st.Page(str(ROOT / "dashboard/pages/detect.py"), title="Detect Faults", icon="🔍"),
        st.Page(str(ROOT / "dashboard/pages/edge_ai.py"), title="Edge AI Lab", icon="⚡"),
    ]
)

try:
    pages.run()
except Exception as exc:
    st.error("The dashboard hit an error during startup.")
    st.exception(exc)
