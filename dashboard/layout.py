"""Unified layout, theme, and chart styling."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

# Shared Plotly defaults
CHART = dict(
    font=dict(family="Inter, -apple-system, BlinkMacSystemFont, sans-serif", size=12, color="#334155"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#ffffff",
    margin=dict(l=48, r=24, t=48, b=40),
    xaxis=dict(showgrid=True, gridcolor="#f1f5f9", linecolor="#e2e8f0"),
    yaxis=dict(showgrid=True, gridcolor="#f1f5f9", linecolor="#e2e8f0"),
)


def inject_theme() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.25rem;
            padding-bottom: 2rem;
            max-width: 1100px;
        }
        [data-testid="stHorizontalBlock"] {
            gap: 1rem;
            align-items: stretch !important;
        }
        [data-testid="column"] {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }
        [data-testid="stMetric"] {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 0.65rem 0.85rem;
            min-height: 5.5rem;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.75rem !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: #64748b !important;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.35rem !important;
            font-weight: 700 !important;
            color: #0f172a !important;
        }
        [data-testid="stSidebar"] {
            background-color: #f8fafc;
            border-right: 1px solid #e2e8f0;
        }
        div[data-testid="stForm"] {
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 1.25rem;
            background: #fafafa;
        }
        h1 { font-size: 1.75rem !important; font-weight: 700 !important; color: #0f172a !important; }
        h2, h3 { color: #0f172a !important; }
        .page-subtitle { color: #64748b; font-size: 1rem; margin-top: -0.5rem; margin-bottom: 1.25rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.markdown(f'<p class="page-subtitle">{subtitle}</p>', unsafe_allow_html=True)


def step_block(number: int, title: str, description: str) -> None:
    with st.container(border=True):
        st.markdown(f"**Step {number} · {title}**")
        st.caption(description)


def metrics_row(items: list[tuple[str, str]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        col.metric(label, value)


def verdict_banner(fault: bool, confidence: float) -> None:
    if fault:
        st.error(f"**FAULT DETECTED** — {confidence:.0%} model confidence")
    else:
        st.success(f"**NORMAL OPERATION** — {confidence:.0%} model confidence")


def maintenance_panel(maintenance: dict) -> None:
    with st.container(border=True):
        st.markdown("#### Maintenance recommendation")
        st.markdown(f"**{maintenance.get('summary', '—')}**")
        st.caption(maintenance.get("cmms_work_order_title", ""))
        st.markdown("**Actions**")
        for action in maintenance.get("actions", []):
            st.markdown(f"- {action.replace('_', ' ').title()}")
        if maintenance.get("spare_parts"):
            st.markdown("**Spare parts**")
            for part in maintenance["spare_parts"]:
                st.markdown(f"- {part}")


def styled_figure(fig: go.Figure, height: int = 300) -> go.Figure:
    fig.update_layout(**CHART, height=height)
    return fig
