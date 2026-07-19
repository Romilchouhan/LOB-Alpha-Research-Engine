#!/usr/bin/env python3
"""
Micro-Price-LOB — interactive research dashboard.

Reads the CSV feature matrix emitted by `lob_engine --dump-csv` and provides
interactive exploration: tick-range zoom, feature overlays, and a lightweight
information-coefficient (IC) panel that previews the Phase-1 signal study.

Run:  streamlit run dashboard/app.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Directory the engine writes CSVs into (overridable in the container).
import os
# Anchor to the repo's data/ regardless of launch cwd (env var overrides).
_DEFAULT_DATA = Path(__file__).resolve().parent.parent / "data"
DATA_DIR = Path(os.environ.get("LOB_DATA_DIR", str(_DEFAULT_DATA)))

# Feature columns that may appear in the CSV, with human labels.
FEATURE_LABELS = {
    "obi":            "Order-book imbalance",
    "ofi":            "Order-flow imbalance (per-event)",
    "ofi_rolling":    "OFI (rolling sum)",
    "bid_slope":      "Bid-side book slope",
    "ask_slope":      "Ask-side book slope",
    "rv_50":          "Realized vol (50)",
    "rv_200":         "Realized vol (200)",
    "accel":          "Mid acceleration",
    "depth_imb_flow": "Depth-imbalance flow",
}

# Page config is set once in streamlit_app.py (the st.navigation entry point).


# ── Data loading ─────────────────────────────────────────────────────────────
# An engine-dumped feature matrix starts with this exact header prefix. Used to
# distinguish our CSVs from raw input files (e.g. the 592 MB FI2010_train.csv).
_ENGINE_HEADER_PREFIX = "tick,mid_price,micro_price"


def _is_engine_csv(path: Path) -> bool:
    try:
        with path.open("r") as fh:
            return fh.readline().startswith(_ENGINE_HEADER_PREFIX)
    except OSError:
        return False


def list_datasets() -> dict[str, Path]:
    if not DATA_DIR.exists():
        return {}
    return {p.stem: p for p in sorted(DATA_DIR.glob("*.csv")) if _is_engine_csv(p)}


@st.cache_data(show_spinner=False)
def load_csv(path: str, mtime: float) -> pd.DataFrame:
    # mtime is part of the cache key so edits invalidate the cache.
    return pd.read_csv(path)


def downsample(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    stride = int(np.ceil(len(df) / max_points))
    return df.iloc[::stride]


# ── IC analysis ──────────────────────────────────────────────────────────────
def forward_return(mid: pd.Series, k: int) -> pd.Series:
    return mid.shift(-k) / mid - 1.0


def ic_table(df: pd.DataFrame, features: list[str], k: int) -> pd.DataFrame:
    """Pearson IC + Spearman rank-IC of each feature vs forward mid return."""
    fwd = forward_return(df["mid_price"], k)
    # micro-price edge as a derived signal, too
    signals = {f: df[f] for f in features if f in df.columns}
    if "micro_price" in df.columns and "mid_price" in df.columns:
        signals["micro − mid"] = df["micro_price"] - df["mid_price"]

    rows = []
    for name, sig in signals.items():
        pair = pd.concat([sig, fwd], axis=1).dropna()
        pair = pair[np.isfinite(pair).all(axis=1)]
        if len(pair) < 30 or pair.iloc[:, 0].nunique() < 2:
            ic = rank_ic = np.nan
        else:
            ic = pair.iloc[:, 0].corr(pair.iloc[:, 1])
            rank_ic = pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman")
        rows.append({"signal": name, "IC (Pearson)": ic,
                     "rank-IC (Spearman)": rank_ic, "n": len(pair)})
    return pd.DataFrame(rows).sort_values(
        "rank-IC (Spearman)", key=lambda s: s.abs(), ascending=False)


# ── Charts ───────────────────────────────────────────────────────────────────
def price_chart(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28],
        vertical_spacing=0.04,
        subplot_titles=("Micro-price vs mid-price", "Micro − mid (signal)"))
    fig.add_trace(go.Scatter(x=df["tick"], y=df["mid_price"], name="Mid",
                             line=dict(width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["tick"], y=df["micro_price"], name="Micro",
                             line=dict(width=1)), row=1, col=1)
    diff = df["micro_price"] - df["mid_price"]
    fig.add_trace(go.Scatter(x=df["tick"], y=diff, name="micro − mid",
                             line=dict(width=1), fill="tozeroy",
                             showlegend=False), row=2, col=1)
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=1.08),
                      hovermode="x unified")
    return fig


def feature_chart(df: pd.DataFrame, features: list[str], normalize: bool) -> go.Figure:
    fig = go.Figure()
    for f in features:
        if f not in df.columns:
            continue
        y = df[f]
        if normalize:
            sd = y.std()
            y = (y - y.mean()) / sd if sd and np.isfinite(sd) else y * 0
        fig.add_trace(go.Scatter(x=df["tick"], y=y,
                                 name=FEATURE_LABELS.get(f, f),
                                 line=dict(width=1)))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.12),
                      hovermode="x unified",
                      yaxis_title="z-score" if normalize else "value")
    return fig


# ── App ──────────────────────────────────────────────────────────────────────
st.title("Feature Explorer")
st.caption("Interactive exploration of an engine-dumped feature matrix. Zoom a "
           "tick range, overlay features, and preview each signal's in-sample "
           "information coefficient. For the headline result, see the "
           "**Benchmark** page.")

datasets = list_datasets()
if not datasets:
    st.error(f"No CSV files found in `{DATA_DIR}/`. Run the engine with "
             f"`--dump-csv {DATA_DIR}/<name>.csv` first.")
    st.stop()

with st.sidebar:
    st.header("Controls")
    ds_name = st.selectbox("Dataset", list(datasets.keys()))
    path = datasets[ds_name]
    df = load_csv(str(path), path.stat().st_mtime)

    n = len(df)
    st.metric("Ticks", f"{n:,}")

    tick_lo, tick_hi = st.slider("Tick range", 0, n - 1, (0, n - 1))
    view = df.iloc[tick_lo:tick_hi + 1]

    avail = [f for f in FEATURE_LABELS if f in df.columns]
    default = [f for f in ("obi", "ofi_rolling", "depth_imb_flow") if f in avail]
    features = st.multiselect("Features to plot",
                              avail, default=default,
                              format_func=lambda f: FEATURE_LABELS.get(f, f))
    normalize = st.checkbox("Normalize (z-score)", value=True)

    max_points = st.select_slider("Max plotted points",
                                  options=[2000, 5000, 10000, 25000],
                                  value=5000)
    horizon = st.select_slider("IC forward horizon k (ticks)",
                               options=[1, 5, 10, 20, 50, 100], value=10)

plot_df = downsample(view, max_points)
if len(view) > len(plot_df):
    st.caption(f"Showing {len(plot_df):,} of {len(view):,} ticks "
               f"(downsampled for rendering; IC uses all ticks).")

st.plotly_chart(price_chart(plot_df), use_container_width=True)

if features:
    st.subheader("Feature overlay")
    st.plotly_chart(feature_chart(plot_df, features, normalize),
                    use_container_width=True)

# ── IC panel ────────────────────────────────────────────────────────────────
st.subheader(f"Information coefficient — signal vs forward mid-return (k={horizon})")
st.caption("IC = correlation of each signal at tick *t* with the mid-price "
           "return over the next *k* ticks. rank-IC is the Spearman version "
           "(robust to outliers). **In-sample, single series — a preview, not "
           "the Phase-1 walk-forward result.**")
ic = ic_table(view, features if features else avail, horizon)
st.dataframe(
    ic.style.format({"IC (Pearson)": "{:+.4f}", "rank-IC (Spearman)": "{:+.4f}",
                     "n": "{:,}"})
      .background_gradient(cmap="RdBu", subset=["rank-IC (Spearman)"],
                           vmin=-0.1, vmax=0.1),
    use_container_width=True, hide_index=True)

with st.expander("Raw data (current tick range)"):
    st.dataframe(view.head(1000), use_container_width=True, hide_index=True)
    st.download_button("Download filtered CSV",
                       view.to_csv(index=False).encode(),
                       file_name=f"{ds_name}_{tick_lo}-{tick_hi}.csv",
                       mime="text/csv")
