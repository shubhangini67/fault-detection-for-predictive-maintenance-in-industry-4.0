# Deployment Guide

Deploy the **Industrial Equipment Fault Detection System** dashboard locally (Docker) or to the cloud (Streamlit Community Cloud, Render).

---

## Option 1: Docker (Recommended — Local or Any Host)

Pre-trained demo models are bundled in `deploy/artifacts/` and restored automatically on startup.

```bash
# Build and run
docker compose up --build -d

# Open dashboard
open http://localhost:8501

# View logs
docker compose logs -f

# Stop
docker compose down
```

Single-container alternative:

```bash
docker build -t industrial-fault-detection .
docker run -p 8501:8501 industrial-fault-detection
```

---

## Option 2: Streamlit Community Cloud (Free Hackathon Demo)

1. Push this repo to GitHub (include `deploy/artifacts/` — models are bundled there).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app** and configure:
   - **Repository:** `your-username/fault-detection-for-predictive-maintenance-in-industry-4.0`
   - **Branch:** `master`
   - **Main file path:** `dashboard/app.py`
4. Deploy. First build takes ~5–10 min (TensorFlow install).

The app restores bundled models from `deploy/artifacts/` on cold start via `scripts/ensure_artifacts.py`.

---

## Option 3: Render.com (Free Tier)

1. Push repo to GitHub.
2. Go to [render.com](https://render.com) → **New** → **Blueprint**.
3. Connect the repo — Render reads `render.yaml` automatically.
4. Deploy. URL will be `https://industrial-fault-detection.onrender.com` (name may vary).

Or manually: **New Web Service** → Docker → point to this repo.

> **Note:** Free tier spins down after inactivity; first request may take ~30s to wake.

---

## Option 4: Run Locally (Development)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/ensure_artifacts.py
streamlit run dashboard/app.py
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `STREAMLIT_SERVER_PORT` | `8501` | Dashboard port |
| `STREAMLIT_SERVER_ADDRESS` | `0.0.0.0` | Bind address (Docker) |
| `STREAMLIT_BROWSER_GATHER_USAGE_STATS` | `false` | Disable telemetry |

---

## Health Check

```bash
curl http://localhost:8501/_stcore/health
```

Returns `ok` when the dashboard is ready.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| **No models / inference fails** | Run `python scripts/ensure_artifacts.py` or train via sidebar |
| **Docker build slow** | TensorFlow is large (~500MB); first build is expected to take several minutes |
| **Render cold start timeout** | Retry after 30s; upgrade plan or use Docker on a VPS |
| **Streamlit Cloud memory limit** | Demo uses bundled IR-7 models only; avoid training all scenarios in cloud |
