#!/usr/bin/env bash
set -euo pipefail

cd /app

cleanup() {
  kill "${BACKEND_PID:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[container] Starting backend..."
python -m uvicorn service.app:app \
  --host 127.0.0.1 \
  --port 8000 \
  --workers 1 &
BACKEND_PID=$!

echo "[container] Waiting for model download/load..."
python - <<'PY'
import time
import urllib.request

url = "http://127.0.0.1:8000/health"

for _ in range(900):
    try:
        with urllib.request.urlopen(url, timeout=4) as response:
            if response.status == 200:
                print("[container] Backend ready.")
                break
    except Exception:
        pass
    time.sleep(2)
else:
    raise SystemExit(
        "Backend did not become healthy."
    )
PY

echo "[container] Starting Streamlit..."
exec python -m streamlit run streamlit_app.py \
  --server.address 0.0.0.0 \
  --server.port 8501 \
  --server.headless true
