"""Detect Faults — analysis workflow."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dashboard.layout import maintenance_panel, metrics_row, page_header, styled_figure, verdict_banner
from src.services.fault_detection_service import FaultDetectionService

svc = FaultDetectionService()
deployed = [s["id"] for s in svc.system_status()["scenarios"] if s["deployed"]]

page_header("Detect Faults", "Configure parameters, run analysis, and review results below.")

if not deployed:
    st.error("No trained models. Train scenarios in **Edge AI Lab** first.")
    st.stop()

# ── Configuration ────────────────────────────────────────────────────────────

with st.container(border=True):
    st.markdown("**Analysis settings**")
    with st.form("detect_form", clear_on_submit=False):
        r1c1, r1c2 = st.columns(2, gap="large")
        with r1c1:
            scenario = st.selectbox(
                "Fault scenario",
                deployed,
                format_func=lambda x: svc.cfg["fault_profiles"][x]["display_name"],
            )
            engine = st.selectbox(
                "ML engine",
                ["Random Forest", "Autoencoder", "TFLite FP16", "Ensemble"],
            )
        with r1c2:
            buffer_size = st.slider("Sample count", 200, 800, 400, 50)
            offset = st.slider("Window offset", 0, 5000, 0, 100)
        submitted = st.form_submit_button("Run analysis", type="primary", use_container_width=True)

# ── Run or show cached ─────────────────────────────────────────────────────────

if submitted:
    try:
        with st.spinner("Running inference…"):
            result = svc.predict(scenario, engine, offset, buffer_size)
            chunk = svc.get_telemetry(scenario, offset, buffer_size)
        st.session_state["detect_result"] = (result, chunk)
        st.session_state["detect_params"] = (scenario, engine, offset, buffer_size)
    except Exception as e:
        st.error(str(e))
        st.stop()
elif "detect_result" not in st.session_state:
    st.info("Set options above and click **Run analysis**.")
    preview = svc.get_telemetry(deployed[0], 0, 400)
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=preview["DE"], name="DE", line=dict(color="#2563eb")))
    fig.add_trace(go.Scatter(y=preview["FE"], name="FE", line=dict(color="#7c3aed")))
    fig.update_layout(title="Telemetry preview")
    st.plotly_chart(styled_figure(fig), use_container_width=True)
    st.stop()

result, chunk = st.session_state["detect_result"]
params = st.session_state.get("detect_params")
current = (scenario, engine, offset, buffer_size)
if params and params != current and not submitted:
    st.warning("Settings changed — click **Run analysis** to refresh.")

# ── Results ────────────────────────────────────────────────────────────────────

st.caption(f"{result.analyzed_at} · {result.engine} · {result.model_type}")
verdict_banner(result.fault_detected, result.confidence)

metrics_row([
    ("Health", result.health_label),
    ("Priority", result.maintenance.get("priority", "—")),
    ("Downtime (h)", str(result.maintenance.get("estimated_downtime_hours", 0))),
    ("Samples", str(result.buffer_stats.get("samples", 0))),
])

st.divider()

chart_col, info_col = st.columns([3, 2], gap="large")

with chart_col:
    line = go.Figure()
    line.add_trace(go.Scatter(y=chunk["DE"], name="DE", line=dict(color="#2563eb", width=2)))
    line.add_trace(go.Scatter(y=chunk["FE"], name="FE", line=dict(color="#7c3aed", width=2)))
    line.update_layout(title="Vibration signals")
    st.plotly_chart(styled_figure(line), use_container_width=True)

    scatter = go.Figure(
        go.Scatter(
            x=chunk["DE"], y=chunk["FE"], mode="markers",
            marker=dict(color=chunk["Fault"].map({0: "#2563eb", 1: "#ef4444"}), size=6, opacity=0.55),
        )
    )
    scatter.update_layout(title="Feature space")
    st.plotly_chart(styled_figure(scatter), use_container_width=True)

with info_col:
    maintenance_panel(result.maintenance)

    if len(result.engines) > 1:
        st.markdown("**Engine comparison**")
        st.dataframe(
            pd.DataFrame([
                {"Engine": k, "Result": "Fault" if v["fault"] else "Normal", "Confidence": f"{v['confidence']:.0%}"}
                for k, v in result.engines.items()
            ]),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Technical notes"):
        for note in result.maintenance.get("technical_notes", []):
            st.markdown(f"- {note}")

    if result.reconstruction_error is not None:
        st.metric("Reconstruction error", f"{result.reconstruction_error:.4f}")
