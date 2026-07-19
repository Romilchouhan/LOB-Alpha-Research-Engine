"""Out-of-sample Spearman rank-IC per feature (the IC-decay table).

Generalises ``dashboard/app.py::ic_table`` (which computes an *in-sample*,
single-series Pearson + Spearman IC) to an **out-of-sample, per-fold** rank-IC
across horizons:

For each walk-forward fold, on the fold's **test block only**, we compute the
forward mid-return ``r_{t,k} = mid[t+k]/mid[t] - 1`` (event horizon ``k``) and the
Spearman rank correlation between each feature at ``t`` and ``r_{t,k}``. The last
``k`` rows of each block are dropped (no future observation -> no cross-block
leak). We average the per-fold rank-ICs to get the reported OOS rank-IC.

Spearman (rank) IC is used -- as in the dashboard -- because it is robust to the
heavy tails and the Zscore-volume scale distortion of the engineered
``micro_minus_mid`` signal.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from .data import FI2010, HORIZON_EVENTS, HORIZON_INDEX, forward_return
from .split import Fold


def _rank_ic(sig: np.ndarray, fwd: np.ndarray) -> float:
    """Spearman rank-IC of a signal vs a forward return, dropping NaN/inf."""
    m = np.isfinite(sig) & np.isfinite(fwd)
    if m.sum() < 30:
        return float("nan")
    s = sig[m]
    f = fwd[m]
    if np.unique(s).size < 2 or np.unique(f).size < 2:
        return float("nan")
    rho, _ = spearmanr(s, f)
    return float(rho)


def ic_decay(
    fi: FI2010,
    folds: list[Fold],
    features: dict[str, np.ndarray],
    horizons: list[int] | None = None,
) -> list[dict]:
    """Out-of-sample rank-IC per (feature, horizon), averaged across folds.

    Parameters
    ----------
    fi : FI2010
        Loaded matrix (provides the level-1 mid for the forward return).
    folds : list[Fold]
        Walk-forward folds (rank-IC computed on each test block).
    features : dict[str, np.ndarray]
        Named signals (length-n arrays) to score, e.g. ``micro_minus_mid`` and
        the level-1 raw columns.
    horizons : list[int] | None
        Plan horizon indices to report (default all of :data:`HORIZON_INDEX`).

    Returns
    -------
    list[dict]
        Rows ``{"feature", "k", "k_events", "rank_ic", "rank_ic_std"}``.
    """
    horizons = horizons or HORIZON_INDEX
    mid = fi.mid()
    rows: list[dict] = []

    for k_index in horizons:
        k_events = HORIZON_EVENTS[HORIZON_INDEX.index(k_index)]
        for name, sig in features.items():
            per_fold: list[float] = []
            for fold in folds:
                te = fold.test
                fwd = forward_return(mid[te], k_events)  # within-block, tail NaN
                per_fold.append(_rank_ic(sig[te], fwd))
            arr = np.array(per_fold, dtype=float)
            arr = arr[np.isfinite(arr)]
            rows.append(
                {
                    "feature": name,
                    "k": k_index,
                    "k_events": k_events,
                    "rank_ic": float(np.mean(arr)) if arr.size else float("nan"),
                    "rank_ic_std": float(np.std(arr)) if arr.size else float("nan"),
                }
            )
    return rows
