"""Industrial Edge AI Fault Detection Dashboard — Streamlit application."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bootstrap import ensure_deploy_artifacts

ensure_deploy_artifacts()

from src.config import load_config, resolve_path
from src.data_loader import list_fault_scenarios, load_stream_chunk
from src.edge_predictor import EdgePredictor
from src.edge_quantization import quantize_autoencoder
from src.train_models import _scenario_prefix, train_all

st.set_page_config(
    page_title="Industrial Edge AI | Fault Detection",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

cfg = load_config()
FAULT_SCENARIOS = list_fault_scenarios(cfg)

PRIORITY_COLORS = {
    "ROUTINE": "#22c55e",
    "SCHEDULED": "#eab308",
    "URGENT": "#f97316",
    "CRITICAL": "#ef4444",
}


def models_exist(scenario: str) -> bool:
    models_dir = resolve_path("models")
    prefix = _scenario_prefix(scenario)
    return (models_dir / f"rf_{prefix}.joblib").exists()


def load_quantization_report(scenario: str) -> dict | None:
    prefix = _scenario_prefix(scenario)
    path = resolve_path("models") / f"quantization_{prefix}.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def render_edge_status():
    st.markdown("### Edge AI Operations Status")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Inference Mode", cfg["edge_ai"]["inference_mode"].replace("_", " ").title())
    c2.metric("Target Device", cfg["edge_ai"]["target_device"].replace("_", " ").title())
    c3.metric("Quantization", "Enabled" if cfg["edge_ai"]["quantization"]["enabled"] else "Off")
    c4.metric("Platform Version", cfg["project"]["version"])


def render_sensor_plots(df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=df["DE"], name="Drive End (DE)", line=dict(color="#3b82f6", width=1)))
    fig.add_trace(go.Scatter(y=df["FE"], name="Fan End (FE)", line=dict(color="#a855f7", width=1)))
    fig.update_layout(
        title="Vibration Telemetry Stream (Edge Buffer)",
        xaxis_title="Sample Index",
        yaxis_title="Normalized Amplitude",
        height=320,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_maintenance_card(maintenance: dict):
    priority = maintenance["priority"]
    color = PRIORITY_COLORS.get(priority, "#64748b")

    st.markdown(
        f"""
        <div style="border-left: 4px solid {color}; padding: 1rem 1.25rem;
                    background: rgba(255,255,255,0.04); border-radius: 0 8px 8px 0;">
            <span style="background:{color}; color:#000; padding:2px 10px;
                         border-radius:4px; font-size:0.75rem; font-weight:700;">
                {priority}
            </span>
            <h4 style="margin:0.75rem 0 0.5rem;">{maintenance['summary']}</h4>
            <p style="opacity:0.7; font-size:0.85rem;">
                CMMS Work Order: {maintenance['cmms_work_order_title']}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Recommended Actions**")
        for action in maintenance["actions"]:
            st.markdown(f"- `{action}`")
        st.metric("Est. Downtime", f"{maintenance['estimated_downtime_hours']} hrs")
    with col_b:
        st.markdown("**Spare Parts**")
        if maintenance["spare_parts"]:
            for part in maintenance["spare_parts"]:
                st.markdown(f"- {part}")
        else:
            st.markdown("_None required_")

    with st.expander("Technical Notes"):
        for note in maintenance["technical_notes"]:
            st.markdown(f"- {note}")


def render_quantization_section(scenario: str):
    st.markdown("### Edge AI Quantization Analysis")
    st.caption(
        "Post-training quantization reduces model footprint and latency for "
        "on-device inference at the industrial edge."
    )

    report = load_quantization_report(scenario)
    if not report:
        st.info("No quantization report found. Run **Quantize Models** from the sidebar.")
        return

    rows = []
    for name, variant in report.get("variants", {}).items():
        if "error" in variant:
            continue
        rows.append(
            {
                "Variant": name,
                "Size (KB)": variant.get("size_kb"),
                "Latency (ms)": variant.get("latency_ms"),
                "Quantization": variant.get("quantization"),
                "Edge Use Case": variant.get("edge_use_case", ""),
            }
        )

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        fig = px.bar(
            df,
            x="Variant",
            y="Size (KB)",
            color="Variant",
            title="Model Size by Quantization Variant",
            text="Size (KB)",
        )
        fig.update_traces(texttemplate="%{text:.1f} KB", textposition="outside")
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

        if "compression_ratio_fp16" in report:
            st.success(
                f"Float16 achieves **{report['compression_ratio_fp16']}×** size reduction "
                "vs float32 — suitable for GPU/NPU-accelerated edge gateways."
            )


with st.sidebar:
    st.title("Edge Control Panel")
    st.caption(cfg["project"]["name"])

    scenario = st.selectbox(
        "Fault Scenario",
        FAULT_SCENARIOS,
        index=FAULT_SCENARIOS.index(cfg["dashboard"]["default_scenario"])
        if cfg["dashboard"]["default_scenario"] in FAULT_SCENARIOS
        else 0,
    )

    profile = cfg["fault_profiles"][scenario]
    st.markdown(f"**Asset Profile:** {profile['display_name']}")
    st.markdown(f"**Severity Class:** `{profile['severity'].upper()}`")

    model_choice = st.radio(
        "Edge Inference Engine",
        ["Random Forest (CPU)", "Autoencoder (Keras)", "Autoencoder (TFLite FP16)"],
    )

    chunk_size = st.slider("Telemetry Buffer Size", 100, 2000, 500, step=100)
    stream_offset = st.slider("Stream Offset", 0, 10000, 0, step=500)

    st.divider()
    if st.button("Train Edge Models", use_container_width=True):
        with st.spinner("Training..."):
            train_all(scenario, sample_size=5000)
        st.success("Models saved to models/")

    if st.button("Quantize for Edge Deployment", use_container_width=True):
        with st.spinner("Quantizing..."):
            try:
                quantize_autoencoder(scenario, sample_size=3000)
                st.success("Quantization complete.")
            except FileNotFoundError as e:
                st.error(str(e))

    st.divider()
    st.markdown(
        """
        **Edge AI Glossary**
        - *Edge inference*: ML on-device, not in cloud
        - *Quantization*: FP32 → FP16/INT8 compression
        - *Telemetry buffer*: Sliding sensor window
        - *CMMS*: Maintenance management system
        """
    )

st.title("Industrial Equipment Fault Detection System")
st.markdown(
    "Real-time **Edge AI** predictive maintenance — vibration fault detection "
    "with on-device inference and CMMS-ready recommendations."
)

render_edge_status()
st.divider()

tab_monitor, tab_inference, tab_quant, tab_arch = st.tabs(
    ["Live Monitoring", "Edge Inference", "Quantization", "Architecture"]
)

with tab_monitor:
    st.subheader("Sensor Telemetry — Simulated Edge Stream")
    chunk = load_stream_chunk(scenario, offset=stream_offset, chunk_size=chunk_size)
    render_sensor_plots(chunk)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Buffer Samples", len(chunk))
    m2.metric("DE Mean", f"{chunk['DE'].mean():.4f}")
    m3.metric("FE Mean", f"{chunk['FE'].mean():.4f}")
    m4.metric("Fault Labels in Buffer", int(chunk["Fault"].sum()))

    scatter = px.scatter(
        chunk,
        x="DE",
        y="FE",
        color=chunk["Fault"].map({0: "Normal", 1: "Fault"}),
        title="DE vs FE Feature Space",
        opacity=0.6,
    )
    scatter.update_layout(height=380, legend_title="Class")
    st.plotly_chart(scatter, use_container_width=True)

with tab_inference:
    st.subheader("On-Device Fault Detection")

    if not models_exist(scenario):
        st.warning("Train edge models first (sidebar).")
    else:
        predictor = EdgePredictor(scenario)
        chunk = load_stream_chunk(scenario, offset=stream_offset, chunk_size=chunk_size)

        try:
            if model_choice.startswith("Random Forest"):
                result = predictor.predict_random_forest(chunk)
            elif model_choice.startswith("Autoencoder (Keras"):
                result = predictor.predict_autoencoder(chunk)
            else:
                result = predictor.predict_tflite(chunk, variant="fp16")

            status_color = "#ef4444" if result.fault_detected else "#22c55e"
            status_text = "FAULT DETECTED" if result.fault_detected else "NORMAL OPERATION"

            st.markdown(
                f"""
                <div style="text-align:center; padding:1.5rem; border-radius:12px;
                            border:1px solid {status_color};">
                    <div style="font-size:2rem; font-weight:800; color:{status_color};">
                        {status_text}
                    </div>
                    <div style="opacity:0.7; margin-top:0.5rem;">
                        Engine: {result.model_type} · Confidence: {result.confidence:.1%}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            ic1, ic2, ic3 = st.columns(3)
            ic1.metric("Model Type", result.model_type)
            ic2.metric("Confidence", f"{result.confidence:.1%}")
            if result.reconstruction_error is not None:
                ic3.metric("Reconstruction Error (MAE)", f"{result.reconstruction_error:.4f}")
            else:
                ic3.metric("Samples Analyzed", result.edge_metadata.get("samples_analyzed", "—"))

            st.divider()
            st.markdown("### Maintenance Recommendation Module")
            render_maintenance_card(result.maintenance)

        except FileNotFoundError as e:
            st.error(str(e))

with tab_quant:
    render_quantization_section(scenario)

with tab_arch:
    st.subheader("Edge AI System Architecture")
    st.markdown(
        """
        | Layer | Component | Edge AI Role |
        |-------|-----------|--------------|
        | **Acquisition** | DE/FE accelerometers | Raw telemetry at source |
        | **Edge Compute** | Industrial gateway / IPC | On-device inference, offline-capable |
        | **Models** | RF + Quantized Autoencoder | Lightweight, deterministic |
        | **Action** | Maintenance module | CMMS work orders, spare parts |
        | **MLOps** | Train → Quantize → Deploy | OTA model updates to edge fleet |

        Original research notebooks (`1.`–`9.` folders) remain for algorithm benchmarking.
        """
    )

    profiles_df = pd.DataFrame(
        [
            {
                "Scenario": k,
                "Display Name": v["display_name"],
                "Severity": v["severity"],
                "Defect (in)": v["defect_size_in"],
            }
            for k, v in cfg["fault_profiles"].items()
        ]
    )
    st.dataframe(profiles_df, use_container_width=True, hide_index=True)
