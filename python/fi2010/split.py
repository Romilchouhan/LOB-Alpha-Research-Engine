"""Anchored walk-forward cross-validation with an embargo gap.

Time-series discipline (hard rule): **never shuffle across time**. We use an
*anchored / expanding* walk-forward scheme:

- Partition the ``n`` ticks into ``n_folds + 1`` equal, contiguous, time-ordered
  chunks. Chunk 0 is the initial anchor; chunks ``1..n_folds`` are the sequential
  forward test folds.
- Fold ``i`` (1-indexed) trains on **all ticks before the start of test chunk i**
  (expanding window: the train set grows each fold) and tests on chunk ``i``.
- An **embargo gap** of ``embargo`` ticks (>= the max prediction horizon, 100
  events here) is removed from the *end* of each train window so that a training
  label ``y_t = sign(mid[t+h] - mid[t])`` cannot peek into the test block.

We assert, per fold, that (a) train and test index sets are **disjoint** and
(b) every test index is strictly greater than every train index (test follows
train in time).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Fold:
    """One walk-forward fold."""

    idx: int  # 1-indexed fold number
    train: np.ndarray  # sorted integer indices
    test: np.ndarray  # sorted integer indices


def walk_forward_splits(n: int, n_folds: int = 5, embargo: int = 100) -> list[Fold]:
    """Return ``n_folds`` anchored, expanding-window folds with an embargo gap.

    Parameters
    ----------
    n : int
        Total number of ticks (rows).
    n_folds : int
        Number of sequential forward test folds.
    embargo : int
        Ticks dropped from the end of each train window (>= max horizon = 100).
    """
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    if embargo < 0:
        raise ValueError("embargo must be >= 0")

    n_chunks = n_folds + 1
    # Chunk boundaries: n_chunks contiguous blocks covering [0, n).
    edges = np.linspace(0, n, n_chunks + 1, dtype=np.int64)

    folds: list[Fold] = []
    for i in range(1, n_chunks):  # test chunk index 1..n_folds
        test_start, test_end = int(edges[i]), int(edges[i + 1])
        train_end = max(0, test_start - embargo)  # embargo gap before test
        train = np.arange(0, train_end, dtype=np.int64)
        test = np.arange(test_start, test_end, dtype=np.int64)

        # --- leakage / ordering asserts (hard rule) ---
        assert train.size > 0, f"fold {i}: empty train window"
        assert test.size > 0, f"fold {i}: empty test window"
        assert np.intersect1d(train, test).size == 0, (
            f"fold {i}: train/test indices overlap"
        )
        assert train.max() < test.min(), (
            f"fold {i}: test must follow train in time "
            f"(train.max={train.max()} >= test.min={test.min()})"
        )
        assert test.min() - train.max() - 1 >= embargo, (
            f"fold {i}: embargo gap {test.min() - train.max() - 1} < {embargo}"
        )
        folds.append(Fold(idx=i, train=train, test=test))
    return folds
