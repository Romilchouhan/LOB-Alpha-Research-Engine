#!/usr/bin/env python3
"""Micro-Price-LOB — Streamlit entry point.

Single-page native app (no embedded HTML): the FI-2010 Benchmark, rendered from
reports/metrics.json with real Streamlit + Plotly charts. Ready to extend to live
data later.

Run:  streamlit run streamlit_app.py   (or ./run.sh)
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Micro-Price-LOB — FI-2010 Benchmark",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Light chrome polish on top of .streamlit/config.toml (dark, monospace).
st.markdown(
    """
    <style>
      .block-container { padding-top: 2.2rem; max-width: 1180px; }
      [data-testid="stMetricValue"] { font-size: 1.55rem; }
      [data-testid="stMetricLabel"] { opacity: 0.72; }
      h1 { letter-spacing: -0.01em; font-weight: 650; }
      h2, h3 { letter-spacing: -0.005em; }
      [data-testid="stHeader"] { background: transparent; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Make `dashboard` importable regardless of launch cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dashboard.benchmark import render  # noqa: E402

render()
