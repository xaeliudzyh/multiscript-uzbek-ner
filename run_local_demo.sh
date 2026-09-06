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
  echo "Stopping demo..."
  kill "${BACKEND_PID:-}" "${FRONTEND_PID:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Python: $PYTHON"
echo "Starting FastAPI..."
"$PYTHON" -m uvicorn service.app:app \
  --host 127.0.0.1 \
  --port 8000 \
  > .demo_logs/backend.log 2>&1 &
BACKEND_PID=$!

echo "Waiting for model/backend..."
"$PYTHON" - <<'PY'
import time
import urllib.request

url = "http://127.0.0.1:8000/health"

for _ in range(600):
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            if response.status == 200:
                print("Backend ready.")
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

echo "Starting Streamlit..."
"$PYTHON" -m streamlit run streamlit_app.py \
  --server.address 127.0.0.1 \
  --server.port 8501 \
  > .demo_logs/frontend.log 2>&1 &
FRONTEND_PID=$!

echo
echo "Frontend:     http://127.0.0.1:8501"
echo "Backend docs: http://127.0.0.1:8000/docs"
echo
echo "First run downloads the model from Hugging Face."
echo "Logs:"
echo "  .demo_logs/backend.log"
echo "  .demo_logs/frontend.log"
echo
echo "Press Ctrl+C to stop."

wait
