#!/usr/bin/env python3
"""
Micro-Price-LOB — Streamlit entry point (Streamlit Community Cloud default).

This home page embeds the interactive micro-price explainer (web/index.html)
with its measured FI-2010 statistics injected, so it runs inside Streamlit's
sandboxed iframe (which cannot fetch local files). The data-exploration app
lives on the "Feature Dashboard" page (see pages/).

Run locally:  streamlit run streamlit_app.py
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Micro-Price · Explainer",
                   page_icon="📈", layout="wide")

WEB = Path(__file__).parent / "web"


def explainer_html() -> str:
    """index.html with measured stats injected as window.__SIGNAL_STATS__."""
    html = (WEB / "index.html").read_text()
    stats = WEB / "signal_stats.json"
    data = stats.read_text() if stats.exists() else "null"
    # Inject measured stats + default to dark to match the Streamlit chrome
    # (the site's ◐ theme button still toggles). Runs before the site's script.
    boot = (
        "<script>"
        f"window.__SIGNAL_STATS__ = {data};"
        "document.documentElement.setAttribute('data-theme','dark');"
        "</script>\n"
    )
    return boot + html


# Full-bleed: strip Streamlit's default page padding so the site owns the canvas.
st.markdown(
    "<style>.block-container{padding:0 !important;max-width:100% !important;}</style>",
    unsafe_allow_html=True)

components.html(explainer_html(), height=4100, scrolling=True)
