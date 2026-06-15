"""Shared UI theme, styles, and chart helpers for the industrial dashboard."""

from __future__ import annotations

import plotly.graph_objects as go

# Industrial command-center palette
C = {
    "bg": "#0b1120",
    "panel": "#111827",
    "panel_border": "#1e293b",
    "accent": "#38bdf8",
    "accent2": "#818cf8",
    "success": "#34d399",
    "warn": "#fbbf24",
    "danger": "#f87171",
    "text": "#e2e8f0",
    "muted": "#94a3b8",
    "grid": "#1e293b",
}

PRIORITY_COLORS = {
    "ROUTINE": C["success"],
    "SCHEDULED": C["warn"],
    "URGENT": "#fb923c",
    "CRITICAL": C["danger"],
}

CHART_DEFAULTS = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Sans, Inter, system-ui, sans-serif", size=11, color=C["muted"]),
    margin=dict(l=12, r=12, t=36, b=12),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
)


def inject_global_css() -> str:
    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
    #MainMenu, footer, header {{ visibility: hidden; }}
    .block-container {{ padding: 1rem 2rem 2rem; max-width: 1440px; }}
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #0f172a 0%, #0b1120 100%);
        border-right: 1px solid {C["panel_border"]};
    }}
    [data-testid="stSidebar"] * {{ color: {C["text"]} !important; }}
    div[data-testid="stTabs"] [data-baseweb="tab-list"] {{
        gap: 0.5rem; background: transparent; border-bottom: 1px solid {C["panel_border"]};
    }}
    div[data-testid="stTabs"] button {{
        font-weight: 600; font-size: 0.85rem; letter-spacing: 0.03em;
        color: {C["muted"]} !important; background: transparent !important;
    }}
    div[data-testid="stTabs"] button[aria-selected="true"] {{
        color: {C["accent"]} !important;
        border-bottom: 2px solid {C["accent"]} !important;
    }}
    [data-testid="stMetric"] {{
        background: {C["panel"]}; border: 1px solid {C["panel_border"]};
        border-radius: 10px; padding: 0.65rem 0.85rem;
    }}
    [data-testid="stMetric"] label {{ color: {C["muted"]} !important; font-size: 0.72rem !important;
        text-transform: uppercase; letter-spacing: 0.05em; }}
    [data-testid="stMetric"] [data-testid="stMetricValue"] {{
        color: {C["text"]} !important; font-weight: 700 !important; }}
    .top-bar {{
        display: flex; justify-content: space-between; align-items: flex-start;
        padding-bottom: 1rem; margin-bottom: 0.5rem;
        border-bottom: 1px solid {C["panel_border"]};
    }}
    .brand-block h1 {{
        font-family: 'IBM Plex Sans', sans-serif; font-size: 1.55rem; font-weight: 700;
        color: {C["text"]}; margin: 0; letter-spacing: -0.02em;
    }}
    .brand-block p {{ color: {C["muted"]}; font-size: 0.88rem; margin: 0.2rem 0 0 0; }}
    .brand-tag {{
        display: inline-block; margin-top: 0.45rem; padding: 0.15rem 0.55rem;
        border-radius: 4px; font-size: 0.68rem; font-weight: 700; letter-spacing: 0.06em;
        background: {C["accent"]}22; color: {C["accent"]}; border: 1px solid {C["accent"]}44;
    }}
    .live-pill {{
        display: inline-flex; align-items: center; gap: 0.4rem;
        padding: 0.35rem 0.75rem; border-radius: 999px; font-size: 0.75rem; font-weight: 600;
        background: #064e3b; color: {C["success"]}; border: 1px solid #065f46;
    }}
    .live-dot {{ width: 7px; height: 7px; border-radius: 50%; background: {C["success"]};
                 box-shadow: 0 0 8px {C["success"]}; animation: pulse 2s infinite; }}
    @keyframes pulse {{ 0%,100% {{ opacity:1; }} 50% {{ opacity:0.4; }} }}
    .panel-card {{
        background: {C["panel"]}; border: 1px solid {C["panel_border"]};
        border-radius: 12px; padding: 1rem 1.1rem; height: 100%;
    }}
    .panel-title {{
        font-size: 0.68rem; font-weight: 700; letter-spacing: 0.08em;
        text-transform: uppercase; color: {C["muted"]}; margin-bottom: 0.65rem;
    }}
    .status-hero {{
        border-radius: 12px; padding: 1.25rem; text-align: center;
        border: 1px solid {C["panel_border"]};
    }}
    .status-hero.ok {{ background: linear-gradient(135deg, #064e3b33, {C["panel"]});
                      border-color: #065f46; }}
    .status-hero.alert {{ background: linear-gradient(135deg, #7f1d1d33, {C["panel"]});
                          border-color: #991b1b; }}
    .status-hero .label {{ font-size: 0.7rem; font-weight: 700; letter-spacing: 0.1em;
                           color: {C["muted"]}; text-transform: uppercase; }}
    .status-hero .value {{ font-size: 1.35rem; font-weight: 800; margin: 0.4rem 0; }}
    .status-hero .value.ok {{ color: {C["success"]}; }}
    .status-hero .value.alert {{ color: {C["danger"]}; }}
    .wo-card {{
        background: {C["panel"]}; border: 1px solid {C["panel_border"]};
        border-left: 3px solid {C["accent"]}; border-radius: 0 10px 10px 0;
        padding: 1rem 1.15rem;
    }}
    .wo-id {{ font-size: 0.68rem; color: {C["muted"]}; font-weight: 600;
              letter-spacing: 0.05em; text-transform: uppercase; }}
    .wo-title {{ font-size: 0.95rem; font-weight: 600; color: {C["text"]}; margin: 0.35rem 0; }}
    .wo-body {{ font-size: 0.82rem; color: {C["muted"]}; line-height: 1.45; }}
    .chip-row {{ display: flex; flex-wrap: wrap; gap: 0.35rem; margin-top: 0.65rem; }}
    .chip {{
        font-size: 0.68rem; font-weight: 600; padding: 0.2rem 0.55rem; border-radius: 4px;
        background: {C["panel_border"]}; color: {C["text"]};
    }}
    .edge-node {{
        font-size: 0.78rem; color: {C["muted"]}; text-align: right;
    }}
    .edge-node strong {{ color: {C["accent"]}; }}
    </style>
    """


def style_figure(fig: go.Figure, height: int = 280, title: str | None = None) -> go.Figure:
    fig.update_layout(**CHART_DEFAULTS, height=height)
    if title:
        fig.update_layout(
            title=dict(text=title, font=dict(size=13, color=C["text"]), x=0, xanchor="left"),
        )
    fig.update_xaxes(gridcolor=C["grid"], linecolor=C["panel_border"], zeroline=False)
    fig.update_yaxes(gridcolor=C["grid"], linecolor=C["panel_border"], zeroline=False)
    return fig


def health_gauge(score: float, title: str = "Asset Health Index") -> go.Figure:
    color = C["success"] if score >= 75 else C["warn"] if score >= 45 else C["danger"]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number=dict(suffix="/100", font=dict(color=C["text"], size=28)),
            title=dict(text=title, font=dict(color=C["muted"], size=12)),
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor=C["muted"], dtick=25),
                bar=dict(color=color, thickness=0.75),
                bgcolor=C["panel"],
                borderwidth=0,
                steps=[
                    dict(range=[0, 45], color="#7f1d1d44"),
                    dict(range=[45, 75], color="#78350f44"),
                    dict(range=[75, 100], color="#064e3b44"),
                ],
                threshold=dict(line=dict(color=C["text"], width=2), value=score, thickness=0.8),
            ),
        )
    )
    return style_figure(fig, height=220)


def model_agreement_chart(labels: list[str], values: list[int]) -> go.Figure:
    colors = [C["success"] if v else C["danger"] for v in values]
    fig = go.Figure(go.Bar(x=labels, y=[1] * len(labels), marker_color=colors, text=labels, textposition="inside"))
    fig.update_layout(showlegend=False, yaxis_visible=False, yaxis_showticklabels=False)
    return style_figure(fig, height=120, title="Multi-Model Consensus")
