#!/usr/bin/env python3
"""Feature Dashboard page — runs the existing dashboard/app.py in-place.

Kept as a thin wrapper so dashboard/app.py stays runnable standalone
(`streamlit run dashboard/app.py`) while also appearing as a page here.
"""
from pathlib import Path

_APP = Path(__file__).resolve().parent.parent / "dashboard" / "app.py"
exec(compile(_APP.read_text(), str(_APP), "exec"))  # noqa: S102
