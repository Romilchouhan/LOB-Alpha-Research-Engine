#!/usr/bin/env python3
"""
preprocess.py — Convert FI-2010 raw .txt / .csv files into a high-speed
binary format for the LOB Alpha Research Engine.

Binary layout
─────────────
Each snapshot = 40 × float64 (little-endian) = 320 bytes, identical to the
C++ LOBSnapshot struct:

  asks[0].price, asks[0].volume,  …  asks[9].price, asks[9].volume,
  bids[0].price, bids[0].volume,  …  bids[9].price, bids[9].volume

Usage
─────
  python scripts/preprocess.py --input data/fi2010.csv --output data/lob.bin
  python scripts/preprocess.py --input data/fi2010.txt --output data/lob.bin --delimiter \\t
  python scripts/preprocess.py --input data/fi2010.csv --output data/lob.bin --stats
"""

import argparse
import struct
import sys
import time
from pathlib import Path

import numpy as np

# ── FI-2010 Column Mapping ───────────────────────────────────────────────────
#
# The canonical FI-2010 dataset has 144 feature columns per row.  The FIRST 40
# are the raw 10-level LOB data that we care about, laid out PER-LEVEL
# INTERLEAVED (NOT block-by-field). For level L (0-based, levels 1..10):
#
#   raw[4L + 0] : ask price  (P_ask, level L+1) — positive, ~0.30–0.32 (Zscore)
#   raw[4L + 1] : ask volume (V_ask, level L+1) — z-scored, can be negative
#   raw[4L + 2] : bid price  (P_bid, level L+1) — positive
#   raw[4L + 3] : bid volume (V_bid, level L+1) — z-scored, can be negative
#
# i.e. raw = [Pa1,Va1,Pb1,Vb1, Pa2,Va2,Pb2,Vb2, …, Pa10,Va10,Pb10,Vb10].
# VERIFIED against data/FI2010_train.csv: even raw indices are prices (all
# positive), odd raw indices are volumes (all negative under Zscore norm).
#
# NOTE: the earlier assumption of a "block" layout (ask_px[0:10],
# ask_vol[10:20], bid_px[20:30], bid_vol[30:40]) was WRONG and scrambled every
# derived feature. See reorder_to_interleaved() for the correct mapping.
#
# Some distributions transpose the matrix (features × time), so we auto-detect
# orientation and transpose if needed.
# ─────────────────────────────────────────────────────────────────────────────

LOB_COLS = 40  # number of LOB columns we extract
SNAPSHOT_FMT = f"<{LOB_COLS}d"  # little-endian, 40 doubles
SNAPSHOT_SIZE = struct.calcsize(SNAPSHOT_FMT)  # should be 320

assert SNAPSHOT_SIZE == 320, f"Unexpected snapshot size: {SNAPSHOT_SIZE}"


def load_raw(path: Path, delimiter: str = ",", skip_header: int = 0) -> np.ndarray:
    """Load raw FI-2010 data from CSV or TXT using pandas (robust)."""
    print(f"[preprocess] Loading {path} …")
    t0 = time.time()

    try:
        import pandas as pd
        # Use pandas for robust loading (handles headers, empty cells, index cols)
        df = pd.read_csv(path, sep=delimiter, skiprows=skip_header, header=None if skip_header > 0 else "infer")
        
        # If the first column looks like an index (0, 1, 2...), drop it
        if df.iloc[:, 0].dtype in (np.int64, np.float64) and np.all(np.diff(df.iloc[:, 0].values) == 1):
             if df.iloc[0, 0] <= 1: # Usually index starts at 0 or 1
                print(f"[preprocess] Dropping column 0 (detected as index)")
                df = df.iloc[:, 1:]

        data = df.to_numpy(dtype=np.float64)
    except ImportError:
        # Fallback to numpy if pandas not available
        print("[preprocess] pandas not found, falling back to np.genfromtxt (slower)")
        data = np.genfromtxt(
            path,
            delimiter=delimiter,
            skip_header=skip_header,
            dtype=np.float64,
            filling_values=0.0
        )

    elapsed = time.time() - t0
    print(f"[preprocess] Loaded shape {data.shape} in {elapsed:.2f}s")

    # ── Auto-detect orientation ──────────────────────────────────────────
    # If rows < columns and columns >= 40, the data is probably transposed
    # (features × time, as in the original FI-2010 .txt release).
    if data.ndim == 2 and data.shape[0] < data.shape[1] and data.shape[0] >= LOB_COLS:
        print("[preprocess] Detected transposed layout — transposing to (time × features)")
        data = data.T

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] < LOB_COLS:
        sys.exit(f"[ERROR] Expected at least {LOB_COLS} columns, got {data.shape[1]}")

    return data


def reorder_to_interleaved(row: np.ndarray) -> np.ndarray:
    """
    Map the raw FI-2010 first-40 columns (per-level interleaved
    [P_ask, V_ask, P_bid, V_bid] × 10 levels) onto the C++ LOBSnapshot
    binary layout, which groups the asks block then the bids block:

      out = [ask[0].price, ask[0].vol, …, ask[9].price, ask[9].vol,
             bid[0].price, bid[0].vol, …, bid[9].price, bid[9].vol]

    For level L (0-based):
      raw[4L + 0] = P_ask  → out[2L]        (ask[L].price)
      raw[4L + 1] = V_ask  → out[2L + 1]    (ask[L].vol)
      raw[4L + 2] = P_bid  → out[20 + 2L]   (bid[L].price)
      raw[4L + 3] = V_bid  → out[20 + 2L + 1] (bid[L].vol)
    """
    interleaved = np.empty(LOB_COLS, dtype=np.float64)
    for L in range(10):
        interleaved[2 * L]          = row[4 * L + 0]   # ask price
        interleaved[2 * L + 1]      = row[4 * L + 1]   # ask volume
        interleaved[20 + 2 * L]     = row[4 * L + 2]   # bid price
        interleaved[20 + 2 * L + 1] = row[4 * L + 3]   # bid volume

    return interleaved


def write_binary(data: np.ndarray, out_path: Path) -> int:
    """Write interleaved snapshots as packed binary."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(out_path, "wb") as f:
        for row in data:
            snap = reorder_to_interleaved(row[:LOB_COLS])
            f.write(struct.pack(SNAPSHOT_FMT, *snap))
            count += 1
    return count


def print_stats(data: np.ndarray) -> None:
    """Print basic statistics for quick sanity checking."""
    lob = data[:, :LOB_COLS]
    # Raw columns are per-level INTERLEAVED: [P_ask, V_ask, P_bid, V_bid] × 10.
    labels = []
    for L in range(10):
        labels += [
            f"ask_px_{L+1}", f"ask_vol_{L+1}",
            f"bid_px_{L+1}", f"bid_vol_{L+1}",
        ]
    print(f"\n{'Column':<14} {'Min':>12} {'Max':>12} {'Mean':>12} {'Std':>12}")
    print("─" * 64)
    for j, label in enumerate(labels):
        col = lob[:, j]
        print(
            f"{label:<14} {col.min():>12.4f} {col.max():>12.4f} "
            f"{col.mean():>12.4f} {col.std():>12.4f}"
        )

    # Best ask = raw col 0 (P_ask level 1), best bid = raw col 2 (P_bid level 1).
    mid = (lob[:, 0] + lob[:, 2]) / 2.0
    spread = lob[:, 0] - lob[:, 2]
    print(f"\n{'Mid-Price':<14} {mid.min():>12.4f} {mid.max():>12.4f} "
          f"{mid.mean():>12.4f} {mid.std():>12.4f}")
    print(f"{'Spread':<14} {spread.min():>12.4f} {spread.max():>12.4f} "
          f"{spread.mean():>12.4f} {spread.std():>12.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert FI-2010 CSV/TXT to binary format for LOB engine"
    )
    parser.add_argument("--input", "-i", required=True, help="Input CSV/TXT path")
    parser.add_argument("--output", "-o", required=True, help="Output .bin path")
    parser.add_argument(
        "--delimiter", "-d", default=",",
        help="Column delimiter (default: comma). Use '\\t' for tab."
    )
    parser.add_argument(
        "--skip-header", type=int, default=0,
        help="Number of header rows to skip"
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Print column-level statistics"
    )
    args = parser.parse_args()

    # Handle escaped tab
    delimiter = "\t" if args.delimiter in ("\\t", "tab") else args.delimiter

    data = load_raw(Path(args.input), delimiter=delimiter, skip_header=args.skip_header)

    if args.stats:
        print_stats(data)

    n = write_binary(data, Path(args.output))
    file_size_mb = Path(args.output).stat().st_size / (1024 * 1024)
    print(f"\n[preprocess] Wrote {n} snapshots → {args.output}"
          f"  ({file_size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
