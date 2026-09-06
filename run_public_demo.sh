#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
mkdir -p .demo_logs

if ! command -v cloudflared >/dev/null 2>&1; then
  echo "cloudflared is not installed."
  echo
  echo "macOS:"
  echo "  brew install cloudflared"
  exit 1
fi

cleanup() {
  echo
  echo "Stopping public demo..."
  kill "${BACKEND_PID:-}" "${FRONTEND_PID:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Starting FastAPI on localhost:8000..."
python -m uvicorn service.gliner_app:app \
  --host 127.0.0.1 \
  --port 8000 \
  > .demo_logs/backend.log 2>&1 &
BACKEND_PID=$!

echo "Starting Streamlit on localhost:8501..."
python -m streamlit run streamlit_app.py \
  --server.address 127.0.0.1 \
  --server.port 8501 \
  > .demo_logs/frontend.log 2>&1 &
FRONTEND_PID=$!

echo "Waiting for frontend..."
sleep 5

echo
echo "A public URL will appear below."
echo "Send the https://....trycloudflare.com link to another computer."
echo
echo "The URL works while this terminal and your Mac stay running."
echo "Press Ctrl+C to stop."
echo

cloudflared tunnel --url http://127.0.0.1:8501
