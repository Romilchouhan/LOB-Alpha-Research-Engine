"""Per-horizon classifiers: multinomial LogReg + LightGBM.

For each horizon we fit two models on every walk-forward fold and evaluate on the
fold's out-of-sample test block:

- **Multinomial logistic regression** (``sklearn``), on **standardised** features
  (``StandardScaler`` fit on train only -- anchored normalisation, no leakage).
- **LightGBM** gradient-boosted trees (multiclass), on raw features (trees are
  scale-invariant).

Labels ``{1,2,3}`` are mapped to ``{0,1,2}`` for the estimators and mapped back
implicitly (metrics are label-invariant). Metrics: **accuracy** and **macro-F1**
(unweighted mean of per-class F1 -- the right metric for the imbalanced
3-class FI-2010 task). We report, per (model, horizon), the mean across folds and
the per-fold spread (std), plus the pooled out-of-sample predictions (used by the
bootstrap CI). Fixed ``seed=42`` everywhere.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from . import SEED
from .split import Fold

try:
    import lightgbm as lgb

    _HAS_LGBM = True
except Exception:  # pragma: no cover - env guard
    _HAS_LGBM = False


@dataclass
class ModelResult:
    """Aggregated result for one (model, horizon)."""

    model: str
    k_index: int          # plan horizon index (JSON key "k")
    k_events: int         # true canonical event horizon
    acc_mean: float
    acc_std: float
    f1_mean: float
    f1_std: float
    fold_acc: list[float] = field(default_factory=list)
    fold_f1: list[float] = field(default_factory=list)
    # Pooled OOS predictions across folds (ordered by fold, then within-block).
    y_true_pooled: np.ndarray = field(default_factory=lambda: np.array([]))
    y_pred_pooled: np.ndarray = field(default_factory=lambda: np.array([]))


def _make_logreg() -> LogisticRegression:
    # Multinomial softmax; lbfgs handles multiclass natively. Deterministic.
    return LogisticRegression(
        C=1.0,
        max_iter=200,
        tol=1e-3,
        solver="lbfgs",
        random_state=SEED,
    )


def _make_lgbm(n_estimators: int) -> "lgb.LGBMClassifier":
    return lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=n_estimators,
        num_leaves=31,
        learning_rate=0.05,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=SEED,
        verbose=-1,
    )


def _subsample_train(train: np.ndarray, cap: int | None, seed: int) -> np.ndarray:
    """Cap the train window size while preserving time order (contiguous tail).

    We keep the most-recent ``cap`` ticks (the tail nearest the test block) rather
    than a random subset, so the anchored/expanding, no-shuffle discipline holds.
    """
    if cap is None or train.size <= cap:
        return train
    return train[-cap:]


def evaluate_horizon(
    X: np.ndarray,
    y_k: np.ndarray,
    folds: list[Fold],
    model: str,
    k_index: int,
    k_events: int,
    lgbm_estimators: int = 200,
    max_train_per_fold: int | None = None,
) -> ModelResult:
    """Fit + evaluate one model for one horizon across all folds."""
    if model == "lightgbm" and not _HAS_LGBM:
        raise RuntimeError("lightgbm not importable (missing libomp?)")

    y0 = (y_k.astype(np.int64) - 1)  # {1,2,3} -> {0,1,2}

    fold_acc: list[float] = []
    fold_f1: list[float] = []
    pooled_true: list[np.ndarray] = []
    pooled_pred: list[np.ndarray] = []

    for fold in folds:
        tr = _subsample_train(fold.train, max_train_per_fold, SEED)
        te = fold.test
        Xtr, ytr = X[tr], y0[tr]
        Xte, yte = X[te], y0[te]

        if model == "logreg":
            scaler = StandardScaler().fit(Xtr)  # anchored: train-only stats
            clf = _make_logreg()
            clf.fit(scaler.transform(Xtr), ytr)
            pred = clf.predict(scaler.transform(Xte))
        elif model == "lightgbm":
            clf = _make_lgbm(lgbm_estimators)
            clf.fit(Xtr, ytr)
            pred = clf.predict(Xte)
        else:
            raise ValueError(f"unknown model {model!r}")

        fold_acc.append(float(accuracy_score(yte, pred)))
        fold_f1.append(float(f1_score(yte, pred, average="macro")))
        pooled_true.append(yte)
        pooled_pred.append(np.asarray(pred))

    return ModelResult(
        model=model,
        k_index=k_index,
        k_events=k_events,
        acc_mean=float(np.mean(fold_acc)),
        acc_std=float(np.std(fold_acc)),
        f1_mean=float(np.mean(fold_f1)),
        f1_std=float(np.std(fold_f1)),
        fold_acc=fold_acc,
        fold_f1=fold_f1,
        y_true_pooled=np.concatenate(pooled_true),
        y_pred_pooled=np.concatenate(pooled_pred),
    )
