"""Edge AI Lab — training and API."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dashboard.layout import page_header, styled_figure
from src.data_loader import list_fault_scenarios
from src.runtime import tensorflow_available
from src.services.fault_detection_service import FaultDetectionService

svc = FaultDetectionService()
scenarios = list_fault_scenarios(svc.cfg)

page_header("Edge AI Lab", "Train models, view quantization reports, and connect via REST API.")

tab_train, tab_quant, tab_api = st.tabs(["Train", "Quantization", "API"])

with tab_train:
    if not tensorflow_available():
        st.info(
            "Training is disabled on Streamlit Cloud. Bundled models are pre-loaded — "
            "use **Detect Faults** for inference. For training, run locally with `requirements-full.txt`."
        )
    left, right = st.columns(2, gap="large")
    with left:
        with st.container(border=True):
            st.markdown("**Train one scenario**")
            train_sc = st.selectbox("Scenario", scenarios)
            if st.button("Start training", type="primary", use_container_width=True, disabled=not tensorflow_available()):
                with st.spinner(f"Training {train_sc}…"):
                    svc.train_scenario(train_sc)
                st.success("Complete.")
                st.rerun()
    with right:
        with st.container(border=True):
            st.markdown("**Bulk operations**")
            if st.button("Train all scenarios", use_container_width=True, disabled=not tensorflow_available()):
                with st.spinner("Training…"):
                    svc.train_all_scenarios()
                st.success("All scenarios trained.")
                st.rerun()
            if st.button("Restore bundled models", use_container_width=True):
                n = svc.restore_artifacts()
                st.success(f"Restored {n} files." if n else "Already present.")
                st.rerun()

    st.subheader("Deployment status")
    df = pd.DataFrame(svc.system_status()["scenarios"]).rename(
        columns={"id": "Scenario", "name": "Name", "severity": "Severity", "deployed": "Ready"}
    )
    df["Ready"] = df["Ready"].map({True: "Yes", False: "No"})
    st.dataframe(df, use_container_width=True, hide_index=True)

with tab_quant:
    q_sc = st.selectbox("Scenario", scenarios, key="quant_sc")
    report = svc.quantization_report(q_sc)
    if not report:
        st.info("Train this scenario first to generate a quantization report.")
    else:
        rows = pd.DataFrame([
            {"Variant": k, "Size (KB)": v.get("size_kb"), "Latency (ms)": v.get("latency_ms")}
            for k, v in report.get("variants", {}).items() if "error" not in v
        ])
        st.dataframe(rows, use_container_width=True, hide_index=True)
        fig = go.Figure(go.Bar(x=rows["Variant"], y=rows["Size (KB)"], marker_color="#2563eb"))
        fig.update_layout(title="Model size by variant")
        st.plotly_chart(styled_figure(fig), use_container_width=True)

with tab_api:
    st.markdown("**Start server:** `uvicorn api.main:app --port 8000`")
    st.markdown("**Docs:** http://localhost:8000/docs")
    st.code(
        'curl -X POST http://localhost:8000/api/v1/predict \\\n'
        '  -H "Content-Type: application/json" \\\n'
        '  -d \'{"scenario": "IR - 7", "engine": "Random Forest"}\'',
        language="bash",
    )
