"""Visual theme for the dashboard."""

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,600;9..40,700&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', system-ui, sans-serif; }
.block-container { padding-top: 1.5rem; max-width: 1180px; }

.hero {
    background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 50%, #0ea5e9 100%);
    border-radius: 16px; padding: 2rem 2.25rem; color: white; margin-bottom: 1.5rem;
}
.hero h1 { font-size: 1.85rem; font-weight: 700; margin: 0 0 0.5rem 0; color: white !important; }
.hero p { font-size: 1.05rem; opacity: 0.92; margin: 0; color: #e0f2fe !important; }

.step-card {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px;
    padding: 1.25rem; height: 100%; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.step-num {
    display: inline-block; width: 28px; height: 28px; line-height: 28px;
    text-align: center; border-radius: 50%; background: #2563eb; color: white;
    font-weight: 700; font-size: 0.85rem; margin-bottom: 0.5rem;
}
.step-title { font-weight: 600; color: #0f172a; margin-bottom: 0.35rem; }
.step-desc { font-size: 0.88rem; color: #64748b; line-height: 1.45; }

.status-ok {
    background: #ecfdf5; border: 1px solid #6ee7b7; border-radius: 12px;
    padding: 1.25rem 1.5rem; text-align: center;
}
.status-bad {
    background: #fef2f2; border: 1px solid #fca5a5; border-radius: 12px;
    padding: 1.25rem 1.5rem; text-align: center;
}
.status-ok .big { font-size: 1.75rem; font-weight: 800; color: #059669; }
.status-bad .big { font-size: 1.75rem; font-weight: 800; color: #dc2626; }
.status-label { font-size: 0.75rem; font-weight: 600; letter-spacing: 0.06em;
                text-transform: uppercase; color: #64748b; }

.maint-box {
    background: #f8fafc; border-left: 4px solid #2563eb; border-radius: 0 10px 10px 0;
    padding: 1.15rem 1.35rem; margin-top: 0.5rem;
}
.maint-box h4 { margin: 0 0 0.5rem 0; color: #0f172a; font-size: 1rem; }
.maint-box p { margin: 0; color: #475569; font-size: 0.9rem; line-height: 1.5; }

[data-testid="stSidebar"] { background: #f8fafc; border-right: 1px solid #e2e8f0; }
div[data-testid="stMetric"] {
    background: white; border: 1px solid #e2e8f0; border-radius: 10px; padding: 0.75rem;
}
</style>
"""


def inject_theme() -> None:
    import streamlit as st
    st.markdown(THEME_CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str) -> None:
    import streamlit as st
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def step_card(num: int, title: str, desc: str) -> str:
    return f"""
    <div class="step-card">
        <div class="step-num">{num}</div>
        <div class="step-title">{title}</div>
        <div class="step-desc">{desc}</div>
    </div>
    """
