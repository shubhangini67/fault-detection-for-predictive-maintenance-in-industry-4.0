"""Application entry — navigation shell."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dashboard.layout import inject_theme

st.set_page_config(
    page_title="Fault Detection Platform",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_theme()

pages = st.navigation(
    [
        st.Page("dashboard/pages/home.py", title="Home", icon="🏠", default=True),
        st.Page("dashboard/pages/detect.py", title="Detect Faults", icon="🔍"),
        st.Page("dashboard/pages/edge_ai.py", title="Edge AI Lab", icon="⚡"),
    ]
)
pages.run()
