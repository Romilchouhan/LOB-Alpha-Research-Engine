#!/usr/bin/env python3
"""
visualize.py — Generate presentation-quality plots from LOB engine CSV output.

Usage:
    # First, run the engine with --dump-csv
    ./build/lob_engine --synthetic --synthetic-n 2000 --dump-csv data/results.csv

    # Then plot
    python scripts/visualize.py --input data/results.csv --output plots/
    python scripts/visualize.py --input data/results.csv --output plots/ --dark
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless rendering

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# ── Style constants ──────────────────────────────────────────────────────────

COLORS_LIGHT = {
    "bg":       "#FFFFFF",
    "text":     "#1a1a2e",
    "grid":     "#e0e0e0",
    "mid":      "#3498db",
    "micro":    "#e74c3c",
    "pnl_pos":  "#27ae60",
    "pnl_neg":  "#e74c3c",
    "depth_imb_flow":     "#8e44ad",
    "thresh":   "#e74c3c",
    "obi_pos":  "#2ecc71",
    "obi_neg":  "#e67e22",
    "fill_buy": "#27ae60",
    "fill_sell":"#e74c3c",
    "abort":    "#f39c12",
    "spread":   "#1abc9c",
}

COLORS_DARK = {
    "bg":       "#0d1117",
    "text":     "#e6edf3",
    "grid":     "#21262d",
    "mid":      "#58a6ff",
    "micro":    "#f97583",
    "pnl_pos":  "#3fb950",
    "pnl_neg":  "#f85149",
    "depth_imb_flow":     "#bc8cff",
    "thresh":   "#f85149",
    "obi_pos":  "#3fb950",
    "obi_neg":  "#d29922",
    "fill_buy": "#3fb950",
    "fill_sell":"#f85149",
    "abort":    "#d29922",
    "spread":   "#39d353",
}


def setup_style(dark: bool):
    """Configure matplotlib for presentation-quality output."""
    c = COLORS_DARK if dark else COLORS_LIGHT
    plt.rcParams.update({
        "figure.facecolor":   c["bg"],
        "axes.facecolor":     c["bg"],
        "text.color":         c["text"],
        "axes.labelcolor":    c["text"],
        "xtick.color":        c["text"],
        "ytick.color":        c["text"],
        "axes.edgecolor":     c["grid"],
        "grid.color":         c["grid"],
        "grid.alpha":         0.5,
        "font.family":        "sans-serif",
        "font.size":          12,
        "axes.titlesize":     16,
        "axes.labelsize":     13,
        "legend.fontsize":    11,
        "figure.dpi":         150,
        "savefig.dpi":        200,
        "savefig.bbox":       "tight",
        "savefig.facecolor":  c["bg"],
    })
    return c


def plot_micro_vs_mid(df: pd.DataFrame, c: dict, out: Path):
    """Plot 1: Micro-Price vs Mid-Price with spread band."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1],
                                    sharex=True)

    t = df["tick"].values
    mid = df["mid_price"].values
    micro = df["micro_price"].values
    diff = micro - mid

    # Top: price series
    ax1.plot(t, mid,   color=c["mid"],   linewidth=1.0, alpha=0.8, label="Mid-Price")
    ax1.plot(t, micro, color=c["micro"], linewidth=1.0, alpha=0.8, label="Micro-Price")
    ax1.fill_between(t, mid, micro, alpha=0.15, color=c["micro"])
    ax1.set_ylabel("Price")
    ax1.set_title("Micro-Price vs Mid-Price — Stoikov Estimator", fontweight="bold")
    ax1.legend(loc="upper left", framealpha=0.7)
    ax1.grid(True, alpha=0.3)

    # Bottom: difference (alpha signal)
    ax2.fill_between(t, diff, 0,
                     where=diff > 0, color=c["obi_pos"], alpha=0.6, label="μ > mid (Buy)")
    ax2.fill_between(t, diff, 0,
                     where=diff < 0, color=c["obi_neg"], alpha=0.6, label="μ < mid (Sell)")
    ax2.axhline(0, color=c["text"], linewidth=0.5, alpha=0.5)
    ax2.set_ylabel("μ − Mid")
    ax2.set_xlabel("Tick")
    ax2.legend(loc="upper left", framealpha=0.7)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out / "micro_vs_mid.png")
    plt.close()
    print(f"  ✓ micro_vs_mid.png")


def plot_pnl(df: pd.DataFrame, c: dict, out: Path):
    """Plot 2: PnL curve with trade markers."""
    fig, ax = plt.subplots(figsize=(14, 6))

    t = df["tick"].values
    pnl = df["pnl"].values

    # PnL line — colored by sign
    ax.fill_between(t, pnl, 0,
                    where=pnl >= 0, color=c["pnl_pos"], alpha=0.3)
    ax.fill_between(t, pnl, 0,
                    where=pnl < 0, color=c["pnl_neg"], alpha=0.3)
    ax.plot(t, pnl, color=c["mid"], linewidth=1.2)

    # Mark aborts
    aborts = df[df["action"].str.contains("ABORT", na=False)]
    if not aborts.empty:
        ax.scatter(aborts["tick"], aborts["pnl"],
                   color=c["abort"], marker="x", s=40, zorder=5,
                   label=f"ABORT ({len(aborts)})", alpha=0.8)

    ax.axhline(0, color=c["text"], linewidth=0.5, alpha=0.5)
    ax.set_title("SimulatedTrader PnL — NOT tradable (no fill/queue/cost model)", fontweight="bold")
    ax.set_xlabel("Tick")
    ax.set_ylabel("Cumulative PnL")
    ax.legend(loc="upper left", framealpha=0.7)
    ax.grid(True, alpha=0.3)

    # Annotate final PnL
    final_pnl = pnl[-1]
    color = c["pnl_pos"] if final_pnl >= 0 else c["pnl_neg"]
    ax.annotate(f"Final PnL: {final_pnl:+.4f}",
                xy=(t[-1], final_pnl), fontsize=12, fontweight="bold",
                color=color, ha="right",
                xytext=(-80, 20 if final_pnl >= 0 else -30),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5))

    plt.tight_layout()
    fig.savefig(out / "pnl_curve.png")
    plt.close()
    print(f"  ✓ pnl_curve.png")


def plot_dif(df: pd.DataFrame, c: dict, out: Path):
    """Plot 3: depth-imbalance-flow time-series with 90th-pct band.

    NOTE: this is the VPIN bucket formula repurposed as a plain feature.
    FI-2010 has no trade prints, so it is NOT a flow-toxicity signal.
    """
    dif = df["depth_imb_flow"].values
    # Filter out zeros (value is NaN/0 before warm-up)
    valid = dif > 0
    if valid.sum() < 10:
        print("  ⚠ Not enough depth-imbalance-flow data for plot")
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    t = df["tick"].values

    ax.plot(t[valid], dif[valid], color=c["depth_imb_flow"], linewidth=1.0, alpha=0.8,
            label="Depth-imbalance flow")

    # 90th percentile band
    p90 = np.percentile(dif[valid], 90)
    ax.axhline(p90, color=c["thresh"], linewidth=1.5, linestyle="--",
               alpha=0.8, label=f"90th Pct ({p90:.4f})")

    # Shade elevated regions
    elevated = valid & (dif >= p90)
    ax.fill_between(t, 0, dif, where=elevated,
                    color=c["thresh"], alpha=0.2, label="Elevated")

    ax.set_title("Depth-imbalance flow (VPIN formula, repurposed as a feature — not toxicity)",
                 fontweight="bold")
    ax.set_xlabel("Tick")
    ax.set_ylabel("Depth-imbalance flow")
    ax.legend(loc="upper right", framealpha=0.7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    fig.savefig(out / "dif_timeseries.png")
    plt.close()
    print(f"  ✓ dif_timeseries.png")


def plot_obi(df: pd.DataFrame, c: dict, out: Path):
    """Plot 4: Order Book Imbalance heatmap-style bar chart."""
    fig, ax = plt.subplots(figsize=(14, 4))

    t = df["tick"].values
    obi = df["obi"].values

    ax.fill_between(t, obi, 0,
                    where=obi >= 0, color=c["obi_pos"], alpha=0.6, label="Buy Pressure")
    ax.fill_between(t, obi, 0,
                    where=obi < 0, color=c["obi_neg"], alpha=0.6, label="Sell Pressure")
    ax.axhline(0, color=c["text"], linewidth=0.5, alpha=0.5)

    ax.set_title("Order Book Imbalance (OBI) — 10 Levels", fontweight="bold")
    ax.set_xlabel("Tick")
    ax.set_ylabel("OBI  ∈ [-1, +1]")
    ax.set_ylim(-1.05, 1.05)
    ax.legend(loc="upper right", framealpha=0.7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out / "obi.png")
    plt.close()
    print(f"  ✓ obi.png")


def plot_dashboard(df: pd.DataFrame, c: dict, out: Path):
    """Plot 5: Combined 4-panel research dashboard."""
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("Micro-Price-LOB — FI-2010 Feature Dashboard",
                 fontsize=20, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.25)

    t = df["tick"].values
    mid = df["mid_price"].values
    micro = df["micro_price"].values
    obi = df["obi"].values
    pnl = df["pnl"].values
    dif = df["depth_imb_flow"].values

    # Panel 1: Micro vs Mid (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, mid, color=c["mid"], linewidth=0.8, alpha=0.8, label="Mid")
    ax1.plot(t, micro, color=c["micro"], linewidth=0.8, alpha=0.8, label="Micro")
    ax1.set_title("Micro-Price vs Mid-Price")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 2: OBI (top-right)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.fill_between(t, obi, 0, where=obi >= 0, color=c["obi_pos"], alpha=0.5)
    ax2.fill_between(t, obi, 0, where=obi < 0, color=c["obi_neg"], alpha=0.5)
    ax2.axhline(0, color=c["text"], linewidth=0.5, alpha=0.3)
    ax2.set_title("Order Book Imbalance")
    ax2.set_ylim(-1.05, 1.05)
    ax2.grid(True, alpha=0.3)

    # Panel 3: PnL (middle, full width)
    ax3 = fig.add_subplot(gs[1, :])
    ax3.fill_between(t, pnl, 0, where=pnl >= 0, color=c["pnl_pos"], alpha=0.3)
    ax3.fill_between(t, pnl, 0, where=pnl < 0, color=c["pnl_neg"], alpha=0.3)
    ax3.plot(t, pnl, color=c["mid"], linewidth=1.0)
    aborts = df[df["action"].str.contains("ABORT", na=False)]
    if not aborts.empty:
        ax3.scatter(aborts["tick"], aborts["pnl"],
                    color=c["abort"], marker="x", s=30, zorder=5, label="ABORT")
    ax3.axhline(0, color=c["text"], linewidth=0.5, alpha=0.3)
    ax3.set_title(f"SimulatedTrader PnL — NOT tradable (no fill/queue/cost model)  (Final: {pnl[-1]:+.4f})")
    ax3.set_ylabel("PnL (arbitrary units)")
    ax3.legend(loc="upper left", fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Panel 4: depth-imbalance flow (bottom, full width)
    valid = dif > 0
    ax4 = fig.add_subplot(gs[2, :])
    if valid.sum() > 10:
        ax4.plot(t[valid], dif[valid], color=c["depth_imb_flow"], linewidth=0.8, alpha=0.8)
        p90 = np.percentile(dif[valid], 90)
        ax4.axhline(p90, color=c["thresh"], linewidth=1.2, linestyle="--", alpha=0.7,
                    label=f"90th Pct ({p90:.4f})")
        elevated = valid & (dif >= p90)
        ax4.fill_between(t, 0, dif, where=elevated, color=c["thresh"], alpha=0.15)
    ax4.set_title("Depth-imbalance flow (feature, not toxicity)")
    ax4.set_xlabel("Tick")
    ax4.set_ylabel("Depth-imbalance flow")
    ax4.legend(loc="upper right", fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(bottom=0)

    fig.savefig(out / "dashboard.png")
    plt.close()
    print(f"  ✓ dashboard.png")


def main():
    parser = argparse.ArgumentParser(
        description="Generate presentation-quality plots from LOB engine output"
    )
    parser.add_argument("--input",  "-i", required=True, help="CSV from --dump-csv")
    parser.add_argument("--output", "-o", default="plots", help="Output directory")
    parser.add_argument("--dark",   action="store_true", help="Use dark theme")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    c = setup_style(args.dark)
    df = pd.read_csv(args.input)

    print(f"\n[visualize] Generating plots ({len(df)} ticks) → {out}/\n")

    # NOTE: plot_pnl() is intentionally NOT called by default. The bundled
    # SimulatedTrader crosses the spread and has no fill/queue/cost model, so
    # its PnL curve is a meaningless unit and must not read as a result.
    plot_micro_vs_mid(df, c, out)
    plot_dif(df, c, out)
    plot_obi(df, c, out)
    plot_dashboard(df, c, out)

    n = len(list(out.glob("*.png")))
    print(f"\n[visualize] Done — {n} plots saved to {out}/\n")


if __name__ == "__main__":
    main()
