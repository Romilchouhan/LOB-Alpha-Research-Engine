"""Politis-Romano stationary bootstrap with automatic block length.

Headline metrics (macro-F1) are not simple means, and the out-of-sample
prediction series is **serially dependent** (adjacent LOB ticks share state), so a
naive i.i.d. bootstrap understates the variance. We use the **stationary
bootstrap** of Politis & Romano (1994): resample *geometric-length blocks* of the
aligned ``(y_true, y_pred)`` series (mean block length ``L``, wrap-around) and
recompute the statistic on each resample.

The mean block length is chosen **automatically** by the Politis & White (2009)
data-driven rule (with the Patton, Politis & White 2009 correction), applied to
the per-observation *correctness indicator* ``1[y_true == y_pred]`` -- i.e. the
block length adapts to the actual autocorrelation of the model's errors.

95% CI = the 2.5 / 97.5 percentiles of the bootstrap distribution. Fixed
``seed=42``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import f1_score

from . import SEED


def _flat_top_lambda(t: np.ndarray) -> np.ndarray:
    """Politis-White flat-top (trapezoidal) lag window."""
    at = np.abs(t)
    out = np.zeros_like(at, dtype=float)
    out[at <= 0.5] = 1.0
    mid = (at > 0.5) & (at <= 1.0)
    out[mid] = 2.0 * (1.0 - at[mid])
    return out


def politis_white_block_length(x: np.ndarray) -> float:
    """Automatic optimal mean block length for the stationary bootstrap.

    Implements Politis & White (2009), "Automatic Block-Length Selection for the
    Dependent Bootstrap" (with the 2009 correction). Returns the stationary-
    bootstrap optimal mean block length ``b*_SB`` (a positive float).
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 8:
        return max(1.0, n ** (1.0 / 3.0))
    x = x - x.mean()

    # Correlogram threshold and search horizon (Politis-White).
    kn = max(5, int(np.ceil(np.sqrt(np.log10(n)))))
    m_max = int(np.ceil(np.sqrt(n))) + kn
    c = 2.0 * np.sqrt(np.log10(n) / n)  # implied 95% test bound on rho(k)

    var = np.dot(x, x) / n
    if var <= 0:
        return max(1.0, n ** (1.0 / 3.0))

    # Sample autocorrelations rho(1..m_max).
    rho = np.empty(m_max + 1)
    rho[0] = 1.0
    for k in range(1, m_max + 1):
        rho[k] = np.dot(x[: n - k], x[k:]) / n / var

    # mhat: smallest m s.t. |rho(m+j)| < c for j=1..kn (kn consecutive lags).
    mhat = None
    for m in range(1, m_max - kn + 1):
        if np.all(np.abs(rho[m + 1 : m + 1 + kn]) < c):
            mhat = m
            break
    if mhat is None:
        mhat = 1
    M = min(2 * mhat, m_max)
    M = max(M, 1)

    # Autocovariances R(k) = var * rho(k), symmetric; apply flat-top window.
    kk = np.arange(-M, M + 1)
    lam = _flat_top_lambda(kk / M)
    acv = var * rho[np.abs(kk)]

    g_hat = float(np.sum(lam * np.abs(kk) * acv))
    d_sb = 2.0 * (float(np.sum(lam * acv))) ** 2

    if d_sb <= 0:
        return max(1.0, n ** (1.0 / 3.0))

    b_star = ((2.0 * g_hat**2) / d_sb) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
    b_max = np.ceil(min(3.0 * np.sqrt(n), n / 3.0))
    return float(min(max(b_star, 1.0), b_max))


def _stationary_indices(n: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """Generate a length-n stationary-bootstrap index path (wrap-around).

    Vectorised: draw geometric block lengths (mean ``mean_block``) and uniform
    block starts, then lay contiguous wrapped runs until length ``n`` is covered.
    Equivalent in distribution to the per-tick Politis-Romano recursion but O(n).
    """
    p = 1.0 / max(mean_block, 1.0)
    est = int(np.ceil(n / max(mean_block, 1.0))) + 16
    lengths = None
    starts = None
    total = 0
    while total < n:
        lengths = rng.geometric(p, size=est)  # support >= 1
        starts = rng.integers(0, n, size=est)
        total = int(lengths.sum())
        est *= 2  # grow if we undershot (rare)
    block_start_cum = np.cumsum(lengths) - lengths  # start offset of each block
    within = np.arange(total) - np.repeat(block_start_cum, lengths)
    idx = (np.repeat(starts, lengths) + within) % n
    return idx[:n]


@dataclass
class BootstrapCI:
    point: float
    lo: float
    hi: float
    mean_block: float
    n_boot: int


def macro_f1_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = 1000,
    seed: int = SEED,
    mean_block: float | None = None,
) -> BootstrapCI:
    """95% stationary-bootstrap CI for pooled OOS macro-F1.

    ``mean_block`` defaults to the Politis-White automatic length on the
    correctness indicator ``1[y_true == y_pred]``.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = y_true.size
    point = float(f1_score(y_true, y_pred, average="macro"))

    if mean_block is None:
        correct = (y_true == y_pred).astype(float)
        mean_block = politis_white_block_length(correct)

    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = _stationary_indices(n, mean_block, rng)
        stats[b] = f1_score(y_true[idx], y_pred[idx], average="macro")
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return BootstrapCI(
        point=point,
        lo=float(lo),
        hi=float(hi),
        mean_block=float(mean_block),
        n_boot=int(n_boot),
    )
