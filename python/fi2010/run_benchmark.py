"""Orchestrate the FI-2010 benchmark and write ``reports/metrics.json``.

Pipeline
--------
1. Load ``data/FI2010_train.csv`` (:mod:`fi2010.data`).
2. Build anchored walk-forward folds with an embargo gap (:mod:`fi2010.split`).
3. For each horizon, fit multinomial LogReg + LightGBM per fold; collect
   accuracy, macro-F1, and pooled OOS predictions (:mod:`fi2010.models`).
4. Stationary-bootstrap 95% CIs on pooled macro-F1 (:mod:`fi2010.bootstrap`).
5. Out-of-sample rank-IC decay per feature (:mod:`fi2010.ic`).
6. Write ``reports/metrics.json`` (interface contract for Workstream C) and print
   a summary table.

Published comparison figures were verified against DeepLOB (Zhang, Zohren &
Roberts 2019, IEEE TSP; arXiv:1808.03668) -- see ``PUBLISHED_REFERENCE`` below.

Run:  ``python python/fi2010/run_benchmark.py``  (add ``--nrows N`` for a smoke
test). Deterministic under ``seed=42``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import SEED
from .bootstrap import macro_f1_ci
from .data import (
    DEFAULT_CSV,
    HORIZON_EVENTS,
    HORIZON_INDEX,
    load_fi2010,
)
from .ic import ic_decay
from .models import evaluate_horizon
from .split import walk_forward_splits

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = _REPO_ROOT / "reports" / "metrics.json"

# Verified against arXiv:1808.03668 (DeepLOB, Zhang/Zohren/Roberts 2019).
# FI-2010 horizons are EVENT counts {10,20,30,50,100}. Setup 2 is the anchored
# (train-early / test-late) protocol closest to our walk-forward; Setup 1 is the
# 9-fold CV. Classical-baseline rows are as tabulated in the same paper.
PUBLISHED_REFERENCE = [
    {"source": "DeepLOB 2019 (Setup 2, anchored)", "k": 10, "macro_f1": 0.8340,
     "accuracy": 0.8447,
     "setup": "FI-2010 Zscore, horizon=10 events, train-early/test-late (Table II)"},
    {"source": "DeepLOB 2019 (Setup 2, anchored)", "k": 20, "macro_f1": 0.7282,
     "accuracy": 0.7485,
     "setup": "FI-2010 Zscore, horizon=20 events, train-early/test-late"},
    {"source": "DeepLOB 2019 (Setup 2, anchored)", "k": 50, "macro_f1": 0.8035,
     "accuracy": 0.8051,
     "setup": "FI-2010 Zscore, horizon=50 events, train-early/test-late"},
    {"source": "DeepLOB 2019 (Setup 1, 9-fold CV)", "k": 10, "macro_f1": 0.7766,
     "accuracy": 0.7891,
     "setup": "FI-2010 Zscore, horizon=10 events, anchored 9-fold CV (Table I)"},
    {"source": "DeepLOB 2019 (Setup 1, 9-fold CV)", "k": 100, "macro_f1": 0.7658,
     "accuracy": 0.7666,
     "setup": "FI-2010 Zscore, horizon=100 events, anchored 9-fold CV"},
    {"source": "BoF baseline (Ntakaris et al., via DeepLOB Table I)", "k": 10,
     "macro_f1": 0.3628, "accuracy": 0.5759,
     "setup": "FI-2010, horizon=10 events, classical bag-of-features (Setup 1)"},
    {"source": "MLP baseline (via DeepLOB Table II)", "k": 20, "macro_f1": 0.5112,
     "accuracy": None,
     "setup": "FI-2010, horizon=20 events, MLP on handcrafted features (Setup 2)"},
]

HORIZON_NOTE = (
    "The 5 label columns are the canonical Ntakaris FI-2010 labels. Their TRUE "
    "prediction horizons are event counts {10,20,30,50,100} (verified: stationary-"
    "class fraction falls 0.639->0.245 across columns). The JSON key 'k' is the "
    "Phase-1 plan's 1..10 column index; 'k_events' is the true horizon used for "
    "the DeepLOB comparison. Column j -> event horizon: "
    "1->10, 2->20, 3->30, 5->50, 10->100."
)


def _fmt(x: float) -> str:
    return "  nan " if x != x else f"{x:6.4f}"


def run(args: argparse.Namespace) -> dict:
    t0 = time.time()
    print(f"[fi2010] loading {args.csv} (nrows={args.nrows}) ...", flush=True)
    fi = load_fi2010(args.csv, nrows=args.nrows)
    n = fi.n
    print(f"[fi2010] X={fi.X.shape} y={fi.y.shape} n={n:,}", flush=True)

    folds = walk_forward_splits(n, n_folds=args.n_folds, embargo=args.embargo)
    for f in folds:
        print(f"  fold {f.idx}: train[0:{f.train.max()+1}] "
              f"({f.train.size:,})  test[{f.test.min()}:{f.test.max()+1}] "
              f"({f.test.size:,})", flush=True)

    models_out: dict[str, list[dict]] = {"logreg": [], "lightgbm": []}
    pooled: dict[tuple[str, int], tuple] = {}

    for model in ("logreg", "lightgbm"):
        for k_index, k_events in zip(HORIZON_INDEX, HORIZON_EVENTS):
            tk = time.time()
            y_k = fi.labels_for(k_index)
            res = evaluate_horizon(
                fi.X, y_k, folds, model=model,
                k_index=k_index, k_events=k_events,
                lgbm_estimators=args.lgbm_estimators,
                max_train_per_fold=args.max_train_per_fold,
            )
            ci = macro_f1_ci(
                res.y_true_pooled, res.y_pred_pooled,
                n_boot=args.n_boot, seed=SEED,
            )
            # Report the POOLED out-of-sample point estimates (macro-F1 = ci.point,
            # accuracy over the concatenated OOS predictions). This is the quantity
            # the bootstrap CI is built on, so the CI always brackets it. Fold
            # mean/std are kept separately as cross-fold-stability context.
            acc_pooled = float((res.y_true_pooled == res.y_pred_pooled).mean())
            models_out[model].append({
                "k": k_index,
                "k_events": k_events,
                "acc": round(acc_pooled, 6),
                "macro_f1": round(ci.point, 6),
                "f1_ci95": [round(ci.lo, 6), round(ci.hi, 6)],
                "acc_foldmean": round(res.acc_mean, 6),
                "acc_foldstd": round(res.acc_std, 6),
                "macro_f1_foldmean": round(res.f1_mean, 6),
                "macro_f1_foldstd": round(res.f1_std, 6),
                "boot_mean_block": round(ci.mean_block, 3),
                "fold_f1": [round(x, 6) for x in res.fold_f1],
            })
            pooled[(model, k_index)] = (ci.point, ci)
            print(f"[{model} k={k_index} (h={k_events}ev)] "
                  f"acc={_fmt(res.acc_mean)} f1={_fmt(res.f1_mean)} "
                  f"ci95=[{ci.lo:.4f},{ci.hi:.4f}] "
                  f"({time.time()-tk:.1f}s)", flush=True)

    # --- IC decay: engineered micro_minus_mid + level-1 raw cols ---
    lvl = fi.level1()
    features = {
        "micro_minus_mid": fi.micro_minus_mid(),
        "P_ask_1": lvl["P_ask"], "V_ask_1": lvl["V_ask"],
        "P_bid_1": lvl["P_bid"], "V_bid_1": lvl["V_bid"],
    }
    ic_rows = ic_decay(fi, folds, features)

    # --- headline: strongest LightGBM horizon by mean macro-F1 ---
    best_k = max(HORIZON_INDEX, key=lambda kk: pooled[("lightgbm", kk)][0])
    best_f1, best_ci = pooled[("lightgbm", best_k)]
    best_events = HORIZON_EVENTS[HORIZON_INDEX.index(best_k)]
    headline = {
        "model": "lightgbm",
        "k": best_k,
        "k_events": best_events,
        "macro_f1": round(best_f1, 6),
        "f1_ci95": [round(best_ci.lo, 6), round(best_ci.hi, 6)],
        "vs_deeplob_setup2_same_horizon": _deeplob_at(best_events),
    }

    caps = None
    if args.max_train_per_fold is not None:
        caps = (f"train window capped at {args.max_train_per_fold:,} most-recent "
                f"ticks per fold (time-ordered tail; anchored discipline preserved)")

    metrics = {
        "dataset": f"FI2010_train.csv (Zscore, {n} ticks)",
        "seed": SEED,
        "n_folds": args.n_folds,
        "embargo": args.embargo,
        "horizons": HORIZON_INDEX,
        "horizon_events": HORIZON_EVENTS,
        "horizon_note": HORIZON_NOTE,
        "subsample_note": caps,
        "config": {
            "lgbm_estimators": args.lgbm_estimators,
            "n_boot": args.n_boot,
            "max_train_per_fold": args.max_train_per_fold,
            "metric": "macro-F1 (unweighted 3-class); CI = stationary bootstrap "
                      "(Politis-Romano, Politis-White auto block length)",
        },
        "models": models_out,
        "ic_decay": ic_rows,
        "published_reference": PUBLISHED_REFERENCE,
        "headline": headline,
        "runtime_sec": round(time.time() - t0, 1),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    print(f"\n[fi2010] wrote {out} ({time.time()-t0:.1f}s total)", flush=True)
    _print_summary(metrics)
    return metrics


def _deeplob_at(k_events: int):
    for r in PUBLISHED_REFERENCE:
        if r["source"].startswith("DeepLOB 2019 (Setup 2") and r["k"] == k_events:
            return {"source": r["source"], "macro_f1": r["macro_f1"]}
    return None


def _print_summary(m: dict) -> None:
    print("\n================ FI-2010 benchmark summary ================")
    print(f"dataset: {m['dataset']}  seed={m['seed']}  folds={m['n_folds']}  "
          f"embargo={m['embargo']}")
    if m.get("subsample_note"):
        print(f"NOTE: {m['subsample_note']}")
    print(f"\n{'model':9} {'k':>3} {'h(ev)':>6} {'acc':>7} {'macroF1':>8} "
          f"{'CI95_lo':>8} {'CI95_hi':>8}")
    for model in ("logreg", "lightgbm"):
        for row in m["models"][model]:
            lo, hi = row["f1_ci95"]
            print(f"{model:9} {row['k']:>3} {row['k_events']:>6} "
                  f"{row['acc']:>7.4f} {row['macro_f1']:>8.4f} "
                  f"{lo:>8.4f} {hi:>8.4f}")
    h = m["headline"]
    print(f"\nHEADLINE: {h['model']} @ k-index {h['k']} (={h['k_events']} events) "
          f"macro-F1={h['macro_f1']:.4f} CI95={h['f1_ci95']}")
    ref = h.get("vs_deeplob_setup2_same_horizon")
    if ref:
        print(f"  vs {ref['source']}: macro-F1={ref['macro_f1']:.4f} "
              f"(published deep baseline, same event horizon)")
    print("\nTop rank-IC features (|IC| desc, across horizons):")
    top = sorted(m["ic_decay"], key=lambda r: -abs(r["rank_ic"])
                 if r["rank_ic"] == r["rank_ic"] else 0)[:8]
    for r in top:
        print(f"  {r['feature']:16} k={r['k']:>2} (h={r['k_events']:>3}ev) "
              f"rank_IC={r['rank_ic']:+.4f}")
    print("===========================================================")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FI-2010 mid-price-direction benchmark")
    p.add_argument("--csv", default=str(DEFAULT_CSV), help="FI-2010 CSV path")
    p.add_argument("--nrows", type=int, default=None,
                   help="read only first N rows (smoke test)")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--embargo", type=int, default=100,
                   help="embargo gap >= max horizon (100 events)")
    p.add_argument("--lgbm-estimators", type=int, default=200)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--max-train-per-fold", type=int, default=None,
                   help="optional cap on train-window size per fold (logged)")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    return p


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
