"""Home — platform overview."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dashboard.layout import metrics_row, page_header, step_block
from src.services.fault_detection_service import FaultDetectionService

svc = FaultDetectionService()
status = svc.system_status()

page_header(
    "Industrial Fault Detection",
    "Edge AI platform for bearing fault detection — vibration analysis, ML inference, maintenance actions.",
)

c1, c2, c3 = st.columns(3, gap="medium")
with c1:
    step_block(1, "Select scenario", "Choose a fault type under Detect Faults.")
with c2:
    step_block(2, "Analyze", "Run ML on DE/FE vibration sensor data.")
with c3:
    step_block(3, "Maintain", "Review CMMS work orders and spare parts.")

st.divider()

metrics_row([
    ("Models ready", f"{status['deployed_count']} / {status['total_scenarios']}"),
    ("Version", status["version"]),
    ("Edge mode", status["edge_mode"].replace("_", " ").title()),
    ("Backend", "FastAPI + Python"),
])

st.subheader("Asset catalog")
table = pd.DataFrame(status["scenarios"]).rename(
    columns={"id": "Scenario", "name": "Description", "severity": "Severity", "defect_in": "Defect (in)", "deployed": "Status"}
)
table["Status"] = table["Status"].map({True: "Ready", False: "Not trained"})
st.dataframe(table, use_container_width=True, hide_index=True)

if status["deployed_count"] == 0:
    st.warning("No models deployed — open **Edge AI Lab** to train.")
else:
    st.info(f"Go to **Detect Faults** to analyze any of the {status['deployed_count']} ready scenarios.")

with st.expander("System architecture"):
    st.markdown(
        """
        | Layer | Technology |
        |-------|------------|
        | Sensors | DE / FE vibration (CWRU benchmark data) |
        | Models | Random Forest, Autoencoder, TFLite FP16 |
        | Backend | `FaultDetectionService` + FastAPI REST API |
        | Output | CMMS maintenance recommendations |
        """
    )
