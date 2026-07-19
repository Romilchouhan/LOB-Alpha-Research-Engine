#!/usr/bin/env python3
"""
Micro-Price-LOB — Streamlit entry point (Streamlit Community Cloud default).

Two pages, wired with st.navigation so the sidebar reads "Explainer" /
"Feature Dashboard" (not the entry filename):

  • Explainer — the interactive micro-price site (web/index.html), embedded with
    its measured FI-2010 stats injected so it runs inside Streamlit's sandboxed
    iframe (which cannot fetch local files), dark-matched to the chrome.
  • Feature Dashboard — the existing dashboard/app.py, run as-is.

Only the active page's code runs per rerun (pg.run()), so each page owning its
own st.set_page_config never collides.

Run locally:  streamlit run streamlit_app.py
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

WEB = Path(__file__).parent / "web"


def _explainer_html() -> str:
    """index.html with measured stats injected + dark theme defaulted."""
    html = (WEB / "index.html").read_text()
    stats = WEB / "signal_stats.json"
    data = stats.read_text() if stats.exists() else "null"
    boot = (
        "<script>"
        f"window.__SIGNAL_STATS__ = {data};"
        "document.documentElement.setAttribute('data-theme','dark');"
        "</script>\n"
    )
    return boot + html


def explainer_page() -> None:
    st.set_page_config(page_title="Micro-Price · Explainer",
                       page_icon="📈", layout="wide")
    st.markdown(
        "<style>.block-container{padding:0 !important;max-width:100% !important;}</style>",
        unsafe_allow_html=True)
    components.html(_explainer_html(), height=4100, scrolling=True)


explainer = st.Page(explainer_page, title="Explainer", icon="📈", default=True)
dashboard = st.Page("dashboard/app.py", title="Feature Dashboard", icon="🔬")

st.navigation([explainer, dashboard]).run()
