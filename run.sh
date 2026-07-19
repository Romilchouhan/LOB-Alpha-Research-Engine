#!/usr/bin/env bash
# One-command launcher for the Micro-Price-LOB Streamlit app.
#   ./run.sh
# Creates the venv + installs deps on first run, then starts Streamlit and
# opens the browser. Stop with Ctrl-C.
set -euo pipefail
cd "$(dirname "$0")"

VENV=".venv"
[ -d "$VENV" ] || python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# Install deps only if streamlit isn't already present in the venv.
python -c "import streamlit" 2>/dev/null || pip install -q -r requirements.txt

PORT="${PORT:-8501}"
URL="http://localhost:$PORT"
( sleep 3; command -v open >/dev/null && open "$URL" ) &   # auto-open browser (macOS)
echo "→ Micro-Price-LOB running at $URL   (Ctrl-C to stop)"
exec streamlit run streamlit_app.py --server.port "$PORT"
