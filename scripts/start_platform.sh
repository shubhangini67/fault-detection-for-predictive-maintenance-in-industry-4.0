#!/usr/bin/env bash
# Start full platform: FastAPI backend + Streamlit dashboard
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true

echo "Starting API on http://localhost:8000"
uvicorn api.main:app --host 0.0.0.0 --port 8000 &
API_PID=$!

echo "Starting dashboard on http://localhost:8501"
streamlit run run_dashboard.py --server.port 8501 --server.headless true

kill $API_PID 2>/dev/null || true
