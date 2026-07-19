#!/usr/bin/env python3
"""
verify_layout.py — regression guard for the FI-2010 PER-LEVEL INTERLEAVED
column layout used by scripts/preprocess.py and the C++ loader.

Raw layout (first 40 cols), per level L (0-based, levels 1..10):
    raw[4L+0] = P_ask   raw[4L+1] = V_ask
    raw[4L+2] = P_bid   raw[4L+3] = V_bid

Checks, on a *known* raw row of the Zscore variant:
  1. even raw indices decode to prices (>0 on this row), odd to volumes;
  2. reorder_to_interleaved() maps raw slots to the C++ LOBSnapshot binary
     layout with the documented identity (raw[0]->out[0], raw[2]->out[20], …);
  3. the reordered best-level spread is positive (ask.px > bid.px).

It also reports the *population* share of positive-spread rows: on Zscore data
each column is standardized independently, so a positive spread is NOT a global
invariant (≈42% of rows). That is the documented normalization caveat — full
physical correctness needs the un-normalized DecPre variant (Phase 1c).

Usage:
    python scripts/verify_layout.py --input data/FI2010_train.csv
Exit code 0 on success, 1 on any assertion failure.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess import LOB_COLS, reorder_to_interleaved  # noqa: E402


def load_rows(path: Path, nrows: int) -> np.ndarray:
    import pandas as pd

    # The canonical file has a header row (",0,1,…,148"); let pandas infer it.
    df = pd.read_csv(path, nrows=nrows)
    # Drop the unnamed integer row-index column if present (col 0 == 0,1,2,…).
    col0 = df.iloc[:, 0].to_numpy()
    if col0.dtype.kind in "if" and np.all(np.diff(col0) == 1) and col0[0] <= 1:
        df = df.iloc[:, 1:]
    return df.to_numpy(dtype=np.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/FI2010_train.csv")
    ap.add_argument("--nrows", type=int, default=2000)
    args = ap.parse_args()

    data = load_rows(Path(args.input), args.nrows)
    lob = data[:, :LOB_COLS]
    row0 = lob[0]

    # 1. even = prices (>0 on row 0), odd = volumes (finite).
    prices = row0[0::2]
    volumes = row0[1::2]
    assert np.all(prices > 0.0), "row0 even indices (prices) must be > 0"
    assert np.all(np.isfinite(volumes)), "row0 odd indices (volumes) must be finite"

    # 2. reorder identity: raw[4L,4L+1,4L+2,4L+3] -> out[2L,2L+1,20+2L,20+2L+1].
    out = reorder_to_interleaved(row0)
    for L in range(10):
        assert out[2 * L] == row0[4 * L + 0], f"ask price map broke at L={L}"
        assert out[2 * L + 1] == row0[4 * L + 1], f"ask vol map broke at L={L}"
        assert out[20 + 2 * L] == row0[4 * L + 2], f"bid price map broke at L={L}"
        assert out[20 + 2 * L + 1] == row0[4 * L + 3], f"bid vol map broke at L={L}"

    # 3. positive best-level spread on this known row.
    best_ask_px, best_bid_px = out[0], out[20]
    spread = best_ask_px - best_bid_px
    assert spread > 0.0, f"row0 spread must be positive, got {spread}"

    # Population caveat (informational, not an assertion).
    pa1, pb1 = lob[:, 0], lob[:, 2]
    pos_frac = float((pa1 > pb1).mean())

    print("[verify_layout] PASS — interleaved layout confirmed")
    print(f"  row0 best ask={best_ask_px:.6f}  best bid={best_bid_px:.6f}  "
          f"spread={spread:.6f}")
    print(f"  reorder identity holds for all 10 levels")
    print(f"  Zscore caveat: positive-spread share over {len(lob)} rows = "
          f"{pos_frac:.1%} (NOT a global invariant on standardized columns; "
          f"physical spread needs the DecPre variant)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
