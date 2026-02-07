#!/usr/bin/env bash
set -e

# Streamlit must bind to 0.0.0.0 inside containers so port mapping works
# Also, use $PORT if the platform provides it (many free hosts do).
PORT="${PORT:-8501}"

exec streamlit run app/streamlit_app.py \
  --server.address=0.0.0.0 \
  --server.port="${PORT}" \
  --server.headless=true \
  --browser.gatherUsageStats=false
