#!/usr/bin/env python3
"""
Compute the micro-price signal statistics shown on the explainer site
(web/index.html) directly from an engine-dumped feature matrix.

Signal  = micro_price - mid_price at tick t.
Outcome = sign of the mid-price change over the next k ticks.
Hit-rate = share of ticks where sign(signal) == sign(forward mid change),
           counted only over ticks where BOTH are non-zero.

Emits web/signal_stats.json so the site loads measured numbers instead of
hand-typed constants. Run:

    python3 scripts/compute_signal_stats.py --input data/fi2010.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HORIZONS = [1, 2, 5, 10, 20, 50, 100]
DECILES = 10


def hit_rate(signal: np.ndarray, fwd: np.ndarray) -> tuple[float, int]:
    """Directional agreement between signal sign and forward-change sign."""
    ss, fs = np.sign(signal), np.sign(fwd)
    mask = (ss != 0) & (fs != 0) & np.isfinite(signal) & np.isfinite(fwd)
    if mask.sum() == 0:
        return float("nan"), 0
    return float((ss[mask] == fs[mask]).mean() * 100.0), int(mask.sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/fi2010.csv")
    ap.add_argument("--output", default="web/signal_stats.json")
    args = ap.parse_args()

    df = pd.read_csv(args.input, usecols=["mid_price", "micro_price"])
    mid = df["mid_price"].to_numpy()
    signal = df["micro_price"].to_numpy() - mid
    n = len(df)

    # Hit-rate vs forward horizon k.
    horizon = []
    for k in HORIZONS:
        fwd = np.empty(n)
        fwd[:] = np.nan
        fwd[: n - k] = mid[k:] - mid[: n - k]
        hr, cnt = hit_rate(signal, fwd)
        horizon.append({"k": k, "hit_rate": round(hr, 2), "n": cnt})

    # Decile analysis at k=1: bin by |signal| strength, hit-rate per decile.
    fwd1 = np.empty(n)
    fwd1[:] = np.nan
    fwd1[: n - 1] = mid[1:] - mid[:-1]
    absig = np.abs(signal)
    valid = (np.sign(signal) != 0) & (np.sign(fwd1) != 0) & np.isfinite(fwd1)
    a, s, f = absig[valid], signal[valid], fwd1[valid]
    order = np.argsort(a, kind="stable")
    a, s, f = a[order], s[order], f[order]
    decile = []
    for d in range(DECILES):
        lo, hi = d * len(a) // DECILES, (d + 1) * len(a) // DECILES
        hr, cnt = hit_rate(s[lo:hi], f[lo:hi])
        decile.append({"decile": d + 1, "hit_rate": round(hr, 2), "n": cnt})

    # Real replay slice for the hero animation: pick the 240-tick window with
    # the largest micro-mid activity, so the site animates recorded data
    # instead of a random walk. Emitted as compact rounded arrays.
    WIN = 240
    if n > WIN:
        # Pick a window where the micro-mid signal is large *relative to* the
        # local price drift — a choppy, sideways stretch — so the animated tape
        # actually shows the micro-price leading the mid instead of both lines
        # riding one big directional move. Score = std(signal) / (std(mid)+eps),
        # scanned on a coarse stride for speed.
        # Maximise how often the micro-price crosses the mid (the visible "lead")
        # while keeping the mid's own travel in a gentle band — it should move,
        # but not run away, so the tape auto-scale reveals the oscillation.
        flips_all = (np.diff(np.sign(signal)) != 0).astype(float)
        cflip = np.concatenate([[0.0], np.nancumsum(flips_all)])
        best_score, start = -1.0, 0
        for s in range(0, n - WIN, 60):
            w = mid[s:s + WIN]
            flips = cflip[s + WIN - 1] - cflip[s]
            rel = (np.nanmax(w) - np.nanmin(w)) / (np.nanmean(w) + 1e-12)  # fractional travel
            if not (0.001 <= rel <= 0.015):   # ~0.1%–1.5% of price
                continue
            if flips > best_score:
                best_score, start = flips, s
    else:
        start = 0
    sl = slice(start, start + WIN)
    replay = {
        "start": start,
        "mid": [round(float(x), 7) for x in mid[sl]],
        "micro": [round(float(x), 7) for x in df["micro_price"].to_numpy()[sl]],
    }

    next_tick = horizon[0]["hit_rate"]
    best = max(decile, key=lambda x: x["hit_rate"])
    worst = min(decile, key=lambda x: x["hit_rate"])
    # first horizon at/below coin-flip = where the edge has decayed
    decayed = next((h["k"] for h in horizon if h["hit_rate"] <= 50.0), HORIZONS[-1])

    out = {
        "source": Path(args.input).name,
        "n_ticks": n,
        "next_tick_hit_rate": next_tick,
        "best_decile": best,
        "worst_decile": worst,
        "decay_horizon": decayed,
        "horizon": horizon,
        "decile": [d["hit_rate"] for d in decile],
        "decile_detail": decile,
        "replay": replay,
    }
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
