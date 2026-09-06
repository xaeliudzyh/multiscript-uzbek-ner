#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

mkdir -p .demo_logs

if [ -x ".venv/bin/python" ]; then
  PYTHON=".venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

cleanup() {
  echo
  echo "Stopping LAN demo..."
  kill "${BACKEND_PID:-}" "${FRONTEND_PID:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

"$PYTHON" -m uvicorn service.app:app \
  --host 127.0.0.1 \
  --port 8000 \
  > .demo_logs/backend.log 2>&1 &
BACKEND_PID=$!

echo "Waiting for model/backend..."
"$PYTHON" - <<'PY'
import time
import urllib.request

for _ in range(600):
    try:
        with urllib.request.urlopen(
            "http://127.0.0.1:8000/health",
            timeout=3,
        ) as response:
            if response.status == 200:
                break
    except Exception:
        pass
    time.sleep(2)
else:
    raise SystemExit(
        "Backend did not start. "
        "See .demo_logs/backend.log"
    )
PY

"$PYTHON" -m streamlit run streamlit_app.py \
  --server.address 0.0.0.0 \
  --server.port 8501 \
  > .demo_logs/frontend.log 2>&1 &
FRONTEND_PID=$!

sleep 3

IP="$(ipconfig getifaddr en0 2>/dev/null || true)"
if [ -z "$IP" ]; then
  IP="$(ipconfig getifaddr en1 2>/dev/null || true)"
fi

echo
if [ -n "$IP" ]; then
  echo "Open on another device in the same Wi-Fi:"
  echo "  http://${IP}:8501"
else
  echo "Could not determine Wi-Fi IP."
  echo "Run: ipconfig getifaddr en0"
fi
echo
echo "Press Ctrl+C to stop."

wait
