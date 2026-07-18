# Micro-Price-LOB — LOB Feature Engineering & Prediction Study (FI-2010)

> A **C++20 feature-engineering pipeline** benchmarking limit-order-book
> mid-price-direction prediction on the **FI-2010** benchmark dataset:
> hand-crafted microstructure features vs. DeepLOB baselines, with a
> deterministic C++ hot path exporting a Parquet feature matrix into a
> Python research layer for walk-forward validation, ablation, and
> block-bootstrap confidence intervals.

---

## What this is / what it deliberately does NOT claim

This repo started as an "HFT engine" and was deliberately **reframed** after an
honest microstructure audit. The self-critique is the point — read it first.

**What it is:**
- A correctness-tested C++20 pipeline that parses FI-2010 L2 snapshots and
  computes microstructure features on a fast, look-ahead-safe hot path.
- An exporter to a **Parquet/Arrow feature matrix** for downstream ML research.
- A research harness for measuring whether those features **predict** anything
  (Phase 1, in progress).

**What it deliberately does NOT claim:**
- ❌ **Not a trading engine, and no tradable PnL.** There is no queue model, no
  fill probability, no fees, no latency. The bundled `SimulatedTrader` crosses
  the spread on "fills" — its PnL is a meaningless unit and is not a result.
- ❌ **No VPIN toxicity signal.** FI-2010 is L2 snapshots with **no trade prints**.
  VPIN needs executed trades; fed snapshot volume it is mis-specified. The math is
  retained as a depth-imbalance feature (`DepthImbalanceFlow`) with the
  toxicity claim dropped.
- ❌ **Not production / distributed / co-located.** Single-process research harness.
  (The old Hazelcast layer was removed as cargo-cult architecture.)

> The one honest predictive number today is the **micro-price vs. mid RMSE by
> horizon** below. Phase 1 replaces it with proper directional metrics (IC,
> rank-IC, hit rate, AUC on the canonical 3-class label).

---

## Current results (honest baseline)

Micro-price (Stoikov) vs. realized mid, RMSE by forward horizon, FI-2010
(362,400 snapshots, decimal-precision binary):

| Horizon (ticks) | RMSE     |
|-----------------|----------|
| 10              | 0.01361  |
| 50              | 0.03047  |
| 100             | 0.04311  |

Hot-path throughput: ~12M ticks/s on Apple M2 (feature compute only).

> ⚠️ RMSE here is dominated by price drift, not signal — it is a smoke test, not a
> predictive claim. Phase 1 replaces it with an IC/rank-IC analysis that
> actually measures prediction.

---

## Features (C++ hot path)

| Feature                | File                                | Well-defined on L2? |
|------------------------|-------------------------------------|---------------------|
| Micro-price (Stoikov)  | `include/features/micro_price.hpp`  | ✅ |
| Order-book imbalance   | `include/features/obi.hpp`          | ✅ |
| Order-flow imbalance   | `include/features/ofi.hpp`          | ✅ (Cont-Kukanov-Stoikov 2014) |
| Queue imbalance        | `include/features/queue_imbalance.hpp` | ✅ |
| Book slope             | `include/features/book_slope.hpp`   | ✅ |
| Realized volatility    | `include/features/realized_vol.hpp` | ✅ |
| Price acceleration     | `include/features/accel.hpp`        | ✅ |
| Depth-imbalance flow   | `include/features/depth_imbalance_flow.hpp` | ⚠️ depth proxy, **not** VPIN toxicity |

---

## Quick start (Docker — one command)

The whole pipeline — build the C++ engine, generate the feature matrix, serve an
interactive dashboard — in one command. No CMake, no Python setup.

```bash
docker compose up            # first run builds the image (~2-3 min)
```

Then open **http://localhost:8501**. The dashboard lets you:
- zoom a tick range, overlay features (OBI, OFI, realized-vol, depth-imbalance flow),
- inspect an **information-coefficient (IC) table** — each signal vs forward mid-return,
  a preview of the Phase-1 study,
- download the filtered feature matrix as CSV.

A **synthetic** dataset is generated inside the container automatically. To also
analyse **real FI-2010**, drop the preprocessed binary at `data/lob.bin` (see
[Data preprocessing](#data-preprocessing)) before `docker compose up` — it is
picked up via the mounted `./data` volume.

---

## Build (native)

Requires CMake 3.22+. Targets Apple Silicon (arm64, `-mcpu=apple-m2`); builds
portably elsewhere.

```bash
# Minimal build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(sysctl -n hw.ncpu)

# With optional features
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DLOB_ENABLE_BOOST=ON \
      -DLOB_ENABLE_PLOTTING=ON
cmake --build build -j$(sysctl -n hw.ncpu)
```

Optional flags (default OFF, fetched via FetchContent): `LOB_ENABLE_BOOST`
(Boost.Accumulators), `LOB_ENABLE_PLOTTING` (Matplot++).

### Tests

32 GoogleTest cases (LOB invariants, micro-price zero-volume edge cases,
OFI/feature correctness).

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(sysctl -n hw.ncpu)
ctest --test-dir build --output-on-failure
```

---

## Run

```bash
# Synthetic data — no files needed
./build/lob_engine --synthetic --synthetic-n 5000

# From pre-processed binary
./build/lob_engine --file data/lob.bin

# From raw FI-2010 CSV
./build/lob_engine --csv data/FI-2010.csv

# Export a tick-level CSV for Python visualization
./build/lob_engine --file data/lob.bin --dump-csv data/results.csv

# Export the Parquet feature matrix for the research layer
./build/lob_engine --file data/lob.bin --emit-parquet data/features.parquet
```

Tunables: `--spread-offset` (0.0001), `--synthetic-n <n>`.

### Data preprocessing

```bash
# Convert FI-2010 CSV → binary (run once)
python3 scripts/preprocess.py --input data/FI-2010.csv --output data/lob.bin --stats

# Python visualization
python3 scripts/visualize.py --input data/results.csv --output plots/fi2010/ --dark
```

---

## Architecture

```
Input (CSV / binary / synthetic)
  → FI2010Parser        parses 40-col snapshots (10 bid + 10 ask levels)
  → LimitOrderBook      current 10-level depth snapshot
  → Feature extraction  micro-price, OBI, OFI, queue-imbalance, book-slope,
                        realized-vol, accel, depth-imbalance-flow
  → Parquet feature matrix  (Arrow) → Python research layer
  → RMSE smoke test / optional plots
```

## Key algorithms

**Micro-price (Stoikov 2018)**
```
μ = P_ask · (V_bid / (V_bid + V_ask))  +  P_bid · (V_ask / (V_bid + V_ask))
```

**Order-book imbalance**
```
OBI = (Σ V_bid − Σ V_ask) / (Σ V_bid + Σ V_ask)   ∈ [−1, +1]
```

**Order-flow imbalance (Cont, Kukanov & Stoikov 2014)** — the L2-correct
replacement for VPIN; computed from best-level changes across consecutive
snapshots.

---

## Roadmap

Phase 0 (honesty pass) done; Phase 1 builds the defensible
research layer (canonical Day 1-7/8-10 split, 3-class labels, IC-decay curve,
LogReg + LightGBM baselines vs. published DeepLOB, block-bootstrap CIs +
leave-one-feature-out ablation). Phase 2 is an LLM analyst agent that drives the
engine as a tool backend, with an eval harness.
