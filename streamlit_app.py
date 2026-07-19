#!/usr/bin/env python3
"""Micro-Price-LOB — Streamlit entry point.

Native multi-page app (no embedded HTML), so every chart is real Streamlit +
Plotly and the surface is ready to extend to live data:

  • Benchmark        — the honest FI-2010 result, rendered from reports/metrics.json.
  • Feature Explorer — interactive exploration of an engine-dumped feature matrix.

``st.set_page_config`` is called once here; the page modules must not call it again.

Run locally:  streamlit run streamlit_app.py
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Micro-Price-LOB",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
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

benchmark = st.Page("dashboard/benchmark.py", title="Benchmark",
                    icon="📊", default=True)
explorer = st.Page("dashboard/app.py", title="Feature Explorer", icon="🔬")

st.navigation([benchmark, explorer]).run()
