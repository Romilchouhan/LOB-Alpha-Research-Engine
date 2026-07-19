#!/usr/bin/env python3
"""Micro-Price-LOB — FI-2010 benchmark page.

Renders the honest Phase-1 result directly from ``reports/metrics.json``:
mid-price-direction classification (LogReg + LightGBM) evaluated walk-forward
with an embargo and 95% stationary-bootstrap CIs, compared against the published
DeepLOB deep-learning baseline. No raw prices are needed — every number and chart
comes from the metrics file, so the page is a faithful view of the committed result.

Design: colour encodes model identity in a fixed, CVD-safe order (LightGBM blue,
LogReg amber); the published DeepLOB baseline is a neutral grey reference, not a
competing hue. One y-axis per chart. Legends always present; CIs shown as error bars.
"""
from __future__ import annotations

import json
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

_REPO = Path(__file__).resolve().parent.parent
_METRICS = _REPO / "reports" / "metrics.json"

# Fixed, CVD-safe categorical hues (dataviz dark ramp). Identity, never rank.
C_LGBM = "#3987e5"   # blue   — LightGBM (headline model)
C_LOGREG = "#c98500"  # amber  — Logistic regression
C_REF = "#8b90a0"    # grey   — published DeepLOB baseline (reference, not a series)
C_IC = ["#3987e5", "#c98500", "#199e70", "#9085e9", "#d55181"]  # up to 5 features
_INK = "#c3c2b7"     # recessive axis/grid ink on the dark surface

MODEL_STYLE = {
    "lightgbm": ("LightGBM", C_LGBM),
    "logreg": ("Logistic reg.", C_LOGREG),
}


@st.cache_data(show_spinner=False)
def load_metrics(path: str, mtime: float) -> dict:
    return json.loads(Path(path).read_text())


def _base_layout(fig: go.Figure, height: int, ytitle: str) -> go.Figure:
    fig.update_layout(
        height=height,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="ui-monospace, monospace", color=_INK, size=13),
        margin=dict(l=60, r=20, t=30, b=50),
        legend=dict(orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
    )
    fig.update_xaxes(
        title_text="Prediction horizon (order-book events ahead)",
        type="category", showgrid=False, zeroline=False,
        linecolor="#2a2f3a", ticks="outside", tickcolor="#2a2f3a",
    )
    fig.update_yaxes(
        title_text=ytitle, gridcolor="#1b2029", zeroline=False,
        linecolor="#2a2f3a",
    )
    return fig


def _deeplob_points(pub: list[dict], metric: str) -> tuple[list[str], list[float]]:
    """Setup-2 (anchored, our protocol) DeepLOB points where reported."""
    key = "macro_f1" if metric == "macro_f1" else "accuracy"
    xs, ys = [], []
    for r in pub:
        if r["source"].startswith("DeepLOB 2019 (Setup 2") and r.get(key) is not None:
            xs.append(str(r["k"]))
            ys.append(r[key])
    return xs, ys


def model_vs_horizon(m: dict, metric: str, ytitle: str) -> go.Figure:
    """metric in {'macro_f1','acc'}. CI error bars only apply to macro_f1."""
    fig = go.Figure()
    xcats = [str(k) for k in m["horizon_events"]]

    for model_key, (label, colour) in MODEL_STYLE.items():
        rows = m["models"][model_key]
        x = [str(r["k_events"]) for r in rows]
        y = [r[metric] for r in rows]
        err = None
        if metric == "macro_f1":
            lo = [r[metric] - r["f1_ci95"][0] for r in rows]
            hi = [r["f1_ci95"][1] - r[metric] for r in rows]
            err = dict(type="data", symmetric=False, array=hi, arrayminus=lo,
                       thickness=1.4, width=4, color=colour)
        fig.add_trace(go.Scatter(
            x=x, y=y, name=label, mode="lines+markers",
            line=dict(color=colour, width=2.4),
            marker=dict(color=colour, size=9, line=dict(color="#0b0e14", width=1.5)),
            error_y=err,
            hovertemplate=f"{label}<br>%{{x}} events · {ytitle}=%{{y:.3f}}<extra></extra>",
        ))

    dx, dy = _deeplob_points(m["published_reference"], metric)
    if dx:
        fig.add_trace(go.Scatter(
            x=dx, y=dy, name="DeepLOB 2019 (published)", mode="markers",
            marker=dict(color=C_REF, size=13, symbol="diamond",
                        line=dict(color="#0b0e14", width=1.5)),
            hovertemplate="DeepLOB (published)<br>%{x} events · "
                          f"{ytitle}=%{{y:.3f}}<extra></extra>",
        ))
    # Random 3-class floor for macro-F1 context (imbalanced → ~0.33 upper ref).
    if metric == "macro_f1":
        fig.add_hline(y=1 / 3, line=dict(color="#3a4150", width=1, dash="dot"),
                      annotation_text="≈ random (3-class)",
                      annotation_position="bottom right",
                      annotation_font_color="#6b7280")
    _base_layout(fig, 430, ytitle)
    fig.update_xaxes(categoryorder="array", categoryarray=xcats)
    return fig


def ic_decay_fig(m: dict) -> go.Figure:
    fig = go.Figure()
    feats: dict[str, list] = {}
    for r in m["ic_decay"]:
        feats.setdefault(r["feature"], []).append(r)
    for i, (feat, rows) in enumerate(sorted(feats.items())):
        rows = sorted(rows, key=lambda r: r["k_events"])
        fig.add_trace(go.Scatter(
            x=[str(r["k_events"]) for r in rows],
            y=[r["rank_ic"] for r in rows],
            name=feat, mode="lines+markers",
            line=dict(color=C_IC[i % len(C_IC)], width=2),
            marker=dict(size=7, line=dict(color="#0b0e14", width=1)),
            hovertemplate=f"{feat}<br>%{{x}} events · rank-IC=%{{y:+.4f}}<extra></extra>",
        ))
    fig.add_hline(y=0, line=dict(color="#3a4150", width=1))
    _base_layout(fig, 380, "Out-of-sample rank-IC (Spearman)")
    return fig


# ── Page ─────────────────────────────────────────────────────────────────────
def render() -> None:
    if not _METRICS.exists():
        st.error(f"`{_METRICS}` not found. Run "
                 "`python -m python.fi2010.run_benchmark --out reports/metrics.json`.")
        st.stop()
    m = load_metrics(str(_METRICS), _METRICS.stat().st_mtime)

    h = m["headline"]
    ref = h["vs_deeplob_setup2_same_horizon"]
    gap = ref["macro_f1"] - h["macro_f1"]
    ci = h["f1_ci95"]

    st.title("FI-2010 mid-price-direction benchmark")
    st.caption(
        "Predicting the next move of the mid-price (down / flat / up) on the "
        "FI-2010 limit-order-book dataset — walk-forward, embargoed, with "
        "bootstrap confidence intervals, measured against a published deep-learning "
        "baseline.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best model", "LightGBM",
              help="Gradient-boosted trees on the 144 canonical FI-2010 features.")
    c2.metric(f"Macro-F1 @ {h['k_events']} events", f"{h['macro_f1']:.3f}",
              help=f"95% stationary-bootstrap CI [{ci[0]:.3f}, {ci[1]:.3f}].")
    c3.metric("Gap to DeepLOB", f"−{gap:.3f} F1", delta=f"{-gap:.3f}",
              delta_color="off",
              help=f"Published DeepLOB macro-F1 {ref['macro_f1']:.3f} at the same "
                   "20-event horizon (Setup 2, anchored).")
    c4.metric("Dataset", "362,400 ticks",
              help=m["dataset"])

    st.markdown(
        f"**A laptop-trainable, interpretable LightGBM reaches macro-F1 "
        f"{h['macro_f1']:.3f} (95% CI [{ci[0]:.3f}, {ci[1]:.3f}]) at the "
        f"{h['k_events']}-event horizon — about {gap:.2f} F1 below the published "
        f"DeepLOB deep network ({ref['macro_f1']:.3f}), not beating it but landing "
        f"in the same ballpark from a model you can inspect.**")

    st.subheader("Macro-F1 vs prediction horizon")
    st.caption("Error bars are 95% stationary-bootstrap CIs (Politis–Romano, "
               "auto block length). DeepLOB diamonds are the published Setup-2 "
               "numbers at the horizons that paper reports (10 / 20 / 50 events).")
    st.plotly_chart(model_vs_horizon(m, "macro_f1", "Macro-F1"),
                    use_container_width=True)

    st.subheader("Accuracy vs prediction horizon")
    st.plotly_chart(model_vs_horizon(m, "acc", "Accuracy"),
                    use_container_width=True)

    st.subheader("Signal decay — rank-IC vs horizon")
    st.caption("Out-of-sample Spearman rank-IC of each interpretable feature "
               "against the forward mid-return. Values are small and fade with "
               "horizon: the naive micro-price tilt has essentially no standalone "
               "edge on this normalized data — the predictive power the models "
               "find is in the joint 144-feature structure, not a single feature.")
    st.plotly_chart(ic_decay_fig(m), use_container_width=True)

    with st.expander("Per-(model, horizon) results table"):
        import pandas as pd
        rows = []
        for mk, (label, _) in MODEL_STYLE.items():
            for r in m["models"][mk]:
                rows.append({
                    "model": label, "horizon (events)": r["k_events"],
                    "accuracy": r["acc"], "macro-F1": r["macro_f1"],
                    "F1 95% CI": f"[{r['f1_ci95'][0]:.3f}, {r['f1_ci95'][1]:.3f}]",
                })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True,
                     column_config={
                         "accuracy": st.column_config.NumberColumn(format="%.3f"),
                         "macro-F1": st.column_config.NumberColumn(format="%.3f"),
                     })

    with st.expander("Method, the loader correction, and honest limitations"):
        st.markdown(
            "**Method.** Labels are the canonical Ntakaris FI-2010 direction labels "
            "at event horizons {10, 20, 30, 50, 100}. Evaluation is anchored "
            f"walk-forward ({m['n_folds']} expanding folds) with an embargo of "
            f"{m['embargo']} events between train and test to prevent label leakage; "
            "train/test index sets are asserted disjoint. Models: multinomial "
            "logistic regression (standardized) and LightGBM. Reported accuracy / "
            "macro-F1 are pooled out-of-sample; CIs use the Politis–Romano "
            "stationary bootstrap. Seed 42, reproducible from a clean checkout.\n\n"
            "**The loader correction.** The raw FI-2010 order-book columns are "
            "per-level *interleaved* `[ask_price, ask_vol, bid_price, bid_vol]`, not "
            "the block layout the original code assumed. That bug had inflated an "
            "earlier '52% edge' claim; on correctly-mapped data the naive "
            "micro-price signal shows no standalone directional edge (~44% "
            "next-event).\n\n"
            "**Limitations.** Single instrument, single normalized variant "
            "(z-scored — so per-tick price/volume *levels* are not reconstructable, "
            "which is why there is no live price/volume panel here yet). This is a "
            "classification benchmark: no fills, queue position, costs, or PnL. "
            "Un-normalized (DecPre) prices for an intuitive micro-vs-mid + volume "
            "panel are planned next.")


if __name__ == "__main__":
    # Allows `streamlit run dashboard/benchmark.py` directly as well.
    render()
