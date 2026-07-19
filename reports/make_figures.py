#!/usr/bin/env python3
"""Generate the FI-2010 benchmark figures from ``reports/metrics.json``.

Reproduces every figure embedded in ``reports/fi2010_benchmark.md`` directly
from the Workstream-A metrics artifact — no raw data required. Uses the repo's
dark plot palette (mirrors ``scripts/visualize.py``).

Usage
-----
    source .venv/bin/activate
    python reports/make_figures.py                       # -> reports/*.png + *.svg
    python reports/make_figures.py --metrics reports/metrics.json --out reports

Figures
-------
1. ``fig_f1_vs_horizon``   — macro-F1 vs EVENT horizon, LogReg / LightGBM (with
   stationary-bootstrap 95% CI error bars) vs published DeepLOB.
2. ``fig_acc_vs_horizon``  — accuracy vs EVENT horizon, same models vs DeepLOB.
3. ``fig_ic_decay``        — rank-IC decay of the interpretable ``micro_minus_mid``
   signal across event horizons, with ±1 fold-std band.

IMPORTANT: axes use ``k_events`` (true horizons {10,20,30,50,100}), never the
column index ``k``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# --- dark palette (mirrors scripts/visualize.py COLORS_DARK) ------------------
C = {
    "bg": "#0d1117",
    "text": "#e6edf3",
    "grid": "#21262d",
    "logreg": "#d29922",   # amber
    "lightgbm": "#58a6ff",  # blue
    "deeplob": "#3fb950",   # green (published SOTA)
    "baseline": "#8b949e",  # grey (classical baselines / chance)
    "ic": "#bc8cff",        # purple
}


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": C["bg"],
        "axes.facecolor": C["bg"],
        "savefig.facecolor": C["bg"],
        "text.color": C["text"],
        "axes.labelcolor": C["text"],
        "xtick.color": C["text"],
        "ytick.color": C["text"],
        "axes.edgecolor": C["grid"],
        "grid.color": C["grid"],
        "grid.alpha": 0.5,
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "legend.fontsize": 10.5,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def _events(rows):
    return [r["k_events"] for r in rows]


def _f1(rows):
    return [r["macro_f1"] for r in rows]


def _acc(rows):
    return [r["acc"] for r in rows]


def _f1_err(rows):
    """Return asymmetric [lo, hi] error offsets from f1_ci95."""
    lo = [r["macro_f1"] - r["f1_ci95"][0] for r in rows]
    hi = [r["f1_ci95"][1] - r["macro_f1"] for r in rows]
    return [lo, hi]


def _deeplob_by_horizon(published, key):
    """Best (Setup-2 preferred) published DeepLOB value per event horizon."""
    out = {}
    for row in published:
        if "DeepLOB" not in row["source"]:
            continue
        val = row.get(key)
        if val is None:
            continue
        ke = row["k"]  # published rows use event horizon directly in 'k'
        # prefer Setup 2 (anchored) when both present
        prefer = "Setup 2" in row["source"]
        if ke not in out or (prefer and not out[ke][1]):
            out[ke] = (val, prefer)
    return {k: v[0] for k, v in out.items()}


def _save(fig, out: Path, name: str) -> None:
    for ext in ("png", "svg"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    print(f"wrote {out / (name + '.png')}  (+ .svg)")


def fig_metric_vs_horizon(m, out: Path, *, metric: str) -> None:
    logreg = m["models"]["logreg"]
    lgbm = m["models"]["lightgbm"]
    getter = _f1 if metric == "macro_f1" else _acc
    label = "macro-F1" if metric == "macro_f1" else "accuracy"

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.grid(True, linewidth=0.6)

    if metric == "macro_f1":
        ax.errorbar(_events(lgbm), _f1(lgbm), yerr=_f1_err(lgbm),
                    color=C["lightgbm"], marker="o", lw=2.2, capsize=4,
                    label="LightGBM (144 feat) · 95% CI")
        ax.errorbar(_events(logreg), _f1(logreg), yerr=_f1_err(logreg),
                    color=C["logreg"], marker="s", lw=2.2, capsize=4,
                    label="LogReg (144 feat) · 95% CI")
    else:
        ax.plot(_events(lgbm), _acc(lgbm), color=C["lightgbm"], marker="o",
                lw=2.2, label="LightGBM (144 feat)")
        ax.plot(_events(logreg), _acc(logreg), color=C["logreg"], marker="s",
                lw=2.2, label="LogReg (144 feat)")

    dl = _deeplob_by_horizon(m["published_reference"],
                             "macro_f1" if metric == "macro_f1" else "accuracy")
    if dl:
        xs = sorted(dl)
        ax.plot(xs, [dl[x] for x in xs], color=C["deeplob"], marker="D",
                lw=2.2, ls="--", label="DeepLOB 2019 (published)")

    ax.set_xscale("log")
    ax.set_xticks([10, 20, 30, 50, 100])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("prediction horizon (events ahead)")
    ax.set_ylabel(label)
    ax.set_title(f"FI-2010 mid-direction: {label} vs horizon", pad=12)
    if metric == "macro_f1":
        ax.axhline(1 / 3, color=C["baseline"], lw=1.0, ls=":",
                   label="3-class chance (0.33)")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    name = "fig_f1_vs_horizon" if metric == "macro_f1" else "fig_acc_vs_horizon"
    _save(fig, out, name)


def fig_ic_decay(m, out: Path) -> None:
    rows = [r for r in m["ic_decay"] if r["feature"] == "micro_minus_mid"]
    rows.sort(key=lambda r: r["k_events"])
    x = [r["k_events"] for r in rows]
    y = [r["rank_ic"] for r in rows]
    sd = [r.get("rank_ic_std", 0.0) for r in rows]

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.grid(True, linewidth=0.6)
    ax.axhline(0, color=C["baseline"], lw=1.0, ls=":")
    ax.fill_between(x, [a - b for a, b in zip(y, sd)],
                    [a + b for a, b in zip(y, sd)],
                    color=C["ic"], alpha=0.15, label="±1 fold-std")
    ax.plot(x, y, color=C["ic"], marker="o", lw=2.2,
            label="rank-IC · micro_minus_mid (corrected loader)")
    ax.set_xscale("log")
    ax.set_xticks([10, 20, 30, 50, 100])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("prediction horizon (events ahead)")
    ax.set_ylabel("Spearman rank-IC")
    ax.set_title("IC decay: interpretable micro-price tilt has ~zero edge", pad=12)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    _save(fig, out, "fig_ic_decay")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metrics", default=str(Path(__file__).parent / "metrics.json"))
    ap.add_argument("--out", default=str(Path(__file__).parent))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    m = json.loads(Path(args.metrics).read_text())

    _style()
    import matplotlib.ticker  # noqa: F401 (registered lazily for formatter above)
    fig_metric_vs_horizon(m, out, metric="macro_f1")
    fig_metric_vs_horizon(m, out, metric="acc")
    fig_ic_decay(m, out)


if __name__ == "__main__":
    import matplotlib.ticker  # ensure available
    main()
