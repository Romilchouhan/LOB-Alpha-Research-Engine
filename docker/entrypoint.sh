#!/usr/bin/env bash
# Generate CSV feature matrices (once) then launch the dashboard.
set -euo pipefail

DATA_DIR="${LOB_DATA_DIR:-/app/data}"
mkdir -p "$DATA_DIR"

# Always provide a synthetic dataset — needs no input files.
if [ ! -f "$DATA_DIR/synthetic.csv" ]; then
    echo "[entrypoint] Generating synthetic feature matrix…"
    lob_engine --synthetic --synthetic-n 5000 --dump-csv "$DATA_DIR/synthetic.csv" >/dev/null
fi

# If a real FI-2010 binary is mounted, derive its feature matrix too.
if [ -f "$DATA_DIR/lob.bin" ] && [ ! -f "$DATA_DIR/fi2010.csv" ]; then
    echo "[entrypoint] Found lob.bin — generating FI-2010 feature matrix…"
    lob_engine --file "$DATA_DIR/lob.bin" --dump-csv "$DATA_DIR/fi2010.csv" >/dev/null
fi

echo "[entrypoint] Launching dashboard on http://localhost:8501"
exec streamlit run dashboard/app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --browser.gatherUsageStats=false
