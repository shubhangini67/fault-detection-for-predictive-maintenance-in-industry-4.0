# Deployment Guide

Deploy the **Industrial Fault Detection Platform** (dashboard + FastAPI backend).

---

## Quick start (Docker)

```bash
docker compose up --build -d
open http://localhost:8501
```

Dashboard entry: `run_dashboard.py`  
API (optional): `uvicorn api.main:app --port 8000`

---

## Streamlit Community Cloud

1. Push repo to GitHub (includes `deploy/artifacts/`).
2. [share.streamlit.io](https://share.streamlit.io) → **New app**
3. **Repository:** `shubhangini67/fault-detection-for-predictive-maintenance-in-industry-4.0`
4. **Branch:** `master`
5. **Main file path:** `run_dashboard.py`
6. **Advanced settings → Python version:** `3.11` (required for TensorFlow)
7. Deploy (~5–10 min first build).

If dependency install fails, confirm Python **3.11** is selected (not 3.13), then **Reboot app** or redeploy.

---

## Render.com

1. Push repo to GitHub.
2. [render.com](https://render.com) → **New Blueprint** → connect repo (`render.yaml`).

---

## Local development

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/bundle_artifacts.py
python scripts/ensure_artifacts.py
streamlit run run_dashboard.py
```

Full stack (API + dashboard):

```bash
bash scripts/start_platform.sh
```

---

## Bundle models before deploy

```bash
python scripts/train.py --scenario "IR - 7"   # or train all via dashboard
python scripts/bundle_artifacts.py            # copy models/ → deploy/artifacts/
```

---

## Health check

```bash
curl http://localhost:8501/_stcore/health   # dashboard
curl http://localhost:8000/api/v1/health    # API
```
