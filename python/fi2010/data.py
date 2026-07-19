"""FI-2010 data loader + engineered micro-price signal.

Reads the canonical **Zscore-normalised** FI-2010 matrix
``data/FI2010_train.csv`` (362,401 lines = 1 header + 362,400 rows, 150
comma-separated columns):

- column 0 is an *unnamed integer row index* -> dropped.
- canonical columns ``0..143`` = **144 features**.
- canonical columns ``144..148`` = the **5 Ntakaris horizon labels**
  (integer classes ``{1,2,3}`` = down / stationary / up).

Column layout of the first 40 features (VERIFIED against the raw file and the
DeepLOB paper, arXiv:1808.03668 sec. III: ``x_t = [P_ask, V_ask, P_bid, V_bid]``
per level ``i=1..10``):  the LOB is **INTERLEAVED per level**, i.e. level ``L``
(1-indexed ``L=l+1``) occupies raw feature indices::

    raw[4*l + 0] = P_ask(L)   raw[4*l + 1] = V_ask(L)
    raw[4*l + 2] = P_bid(L)   raw[4*l + 3] = V_bid(L)

Prices are z-scored but stay positive (~0.30-0.32); volumes are z-scored and are
therefore frequently negative. This is why the naive "block layout" assumption in
the C++/preprocess path (fixed by Workstream B) produced scrambled features.

**Label-horizon note (honesty).** The 5 label columns are the canonical FI-2010
labels whose true prediction horizons are *event counts* ``{10,20,30,50,100}``
(verified here: the stationary-class fraction decreases monotonically
0.639 -> 0.245 across the columns, i.e. longer horizons move more). The Phase-1
plan indexes the same 5 columns as ``k in {1,2,3,5,10}``; we keep that index as
the JSON key but always carry the *true event horizon* alongside so the DeepLOB
comparison is apples-to-apples. See :data:`HORIZON_INDEX` / :data:`HORIZON_EVENTS`.

The engineered :func:`micro_minus_mid` signal is computed from the **correctly
interleaved** level-1 columns so the IC study can include an interpretable
microstructure feature. On Zscore data the volume-weighted micro-price is only
*qualitatively* meaningful (standardised volumes can be negative, so the weight
is not a true fraction) -- this caveat is surfaced in the report and metrics.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Repo root: python/fi2010/data.py -> parents[2] == repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = Path(os.environ.get("FI2010_CSV", _REPO_ROOT / "data" / "FI2010_train.csv"))

N_FEATURES = 144
N_LABELS = 5
N_ROWS = 362_400

# Plan's column-index label -> canonical FI-2010 event horizon for that column.
HORIZON_INDEX = [1, 2, 3, 5, 10]          # plan indexing (kept as JSON key "k")
HORIZON_EVENTS = [10, 20, 30, 50, 100]    # TRUE canonical event horizons
# label column j (0..4) == canonical feature 144+j.


def feature_names() -> list[str]:
    """Return the 144 canonical feature names.

    First 40 get descriptive interleaved names (``P_ask_1``, ``V_ask_1``,
    ``P_bid_1``, ``V_bid_1``, ... level 10); the remaining 104 keep their
    canonical integer index as ``feat_40 .. feat_143``.
    """
    roles = ["P_ask", "V_ask", "P_bid", "V_bid"]
    names: list[str] = []
    for i in range(N_FEATURES):
        if i < 40:
            level = i // 4 + 1
            names.append(f"{roles[i % 4]}_{level}")
        else:
            names.append(f"feat_{i}")
    return names


@dataclass
class FI2010:
    """Loaded FI-2010 matrix.

    Attributes
    ----------
    X : np.ndarray, shape (n, 144), float32
        Feature matrix (Zscore-normalised).
    y : np.ndarray, shape (n, 5), int8
        The 5 horizon labels, classes in ``{1,2,3}``.
    feature_names : list[str]
        Length-144 names (see :func:`feature_names`).
    """

    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]

    @property
    def n(self) -> int:
        return self.X.shape[0]

    def labels_for(self, k_index: int) -> np.ndarray:
        """Return the label column for plan horizon index ``k_index``
        (one of :data:`HORIZON_INDEX`)."""
        j = HORIZON_INDEX.index(k_index)
        return self.y[:, j]

    def level1(self) -> dict[str, np.ndarray]:
        """Return level-1 interleaved columns: P_ask, V_ask, P_bid, V_bid."""
        return {
            "P_ask": self.X[:, 0],
            "V_ask": self.X[:, 1],
            "P_bid": self.X[:, 2],
            "V_bid": self.X[:, 3],
        }

    def mid(self) -> np.ndarray:
        """Level-1 mid price = (P_ask_1 + P_bid_1) / 2 (well-defined on Zscore)."""
        c = self.level1()
        return 0.5 * (c["P_ask"] + c["P_bid"])

    def micro_minus_mid(self) -> np.ndarray:
        """Engineered interpretable signal on the correctly-interleaved level 1.

        Volume-weighted micro-price minus mid:

            micro - mid = (V_bid - V_ask) / (2 (V_ask + V_bid)) * (P_ask - P_bid)

        which is the classic Stoikov micro-price displacement (imbalance-weighted
        half-spread). On Zscore-normalised volumes the denominator can be small or
        negative, so we guard ``|V_ask+V_bid| < eps`` -> 0 and replace any
        non-finite result with 0. Rank-IC (Spearman) downstream is robust to the
        residual scale distortion.
        """
        return micro_minus_mid_from(self.level1())


def micro_minus_mid_from(c: dict[str, np.ndarray], eps: float = 1e-6) -> np.ndarray:
    """Compute ``micro - mid`` from a dict with P_ask/V_ask/P_bid/V_bid arrays."""
    denom = c["V_ask"] + c["V_bid"]
    half_spread = 0.5 * (c["P_ask"] - c["P_bid"])
    with np.errstate(divide="ignore", invalid="ignore"):
        sig = (c["V_bid"] - c["V_ask"]) / denom * half_spread
    sig = np.where(np.abs(denom) < eps, 0.0, sig)
    sig[~np.isfinite(sig)] = 0.0
    return sig.astype(np.float64)


def forward_return(mid, k: int):
    """Forward mid-return over the next ``k`` steps: ``mid[t+k]/mid[t] - 1``.

    Reuses the shape of ``dashboard/app.py::forward_return`` (a Series shift).
    Accepts a pandas Series or a 1-D numpy array; returns the same type. The last
    ``k`` entries are NaN (no future observation) and must be dropped by callers.
    """
    if isinstance(mid, pd.Series):
        return mid.shift(-k) / mid - 1.0
    mid = np.asarray(mid, dtype=np.float64)
    out = np.full_like(mid, np.nan)
    out[:-k] = mid[k:] / mid[:-k] - 1.0
    return out


def load_fi2010(path: str | Path = DEFAULT_CSV, nrows: int | None = None) -> FI2010:
    """Load the FI-2010 matrix, dropping the unnamed index column 0.

    Parameters
    ----------
    path : str | Path
        CSV path (default: ``data/FI2010_train.csv``).
    nrows : int | None
        If given, read only the first ``nrows`` data rows (for smoke tests).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"FI-2010 CSV not found at {path}. Set $FI2010_CSV or place the file "
            f"under data/."
        )
    # Column 0 is the unnamed integer index -> usecols skips it entirely.
    df = pd.read_csv(path, header=0, nrows=nrows, usecols=range(1, 1 + N_FEATURES + N_LABELS))
    arr = df.to_numpy()
    X = arr[:, :N_FEATURES].astype(np.float32)
    y = arr[:, N_FEATURES:].astype(np.int8)
    if nrows is None and X.shape[0] != N_ROWS:
        # Not fatal, but worth flagging: the canonical file is exactly 362,400 rows.
        pass
    return FI2010(X=X, y=y, feature_names=feature_names())
