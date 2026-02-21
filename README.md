# LOB Alpha Research Engine — FI-2010

> **A C++20 HFT research engine for Limit Order Book alpha generation,
> built on the FI-2010 benchmark dataset and optimised for Apple Silicon.**

---

## Front-Office Use Case

In a modern electronic market-making desk, the ability to estimate the
**true price** of an instrument — milliseconds before the rest of the market
converges — is the single most valuable edge.

This engine implements the **Stoikov Micro-Price** estimator, which
weights the mid-price by the volume imbalance at the best bid/ask.
When `micro_price > mid_price + ε`, the engine signals that the fair
value has shifted upward, and a **passive limit buy** at the bid is
likely to get filled and immediately be in-the-money.

Simultaneously, the engine monitors market **toxicity** via
**VPIN** (Volume-Synchronized Probability of Informed Trading).
When toxic flow exceeds the 90th percentile of its historical
distribution, the engine **aborts** all new orders and flattens
existing inventory to avoid adverse selection.

| Signal            | Condition                                   | Action        |
|-------------------|---------------------------------------------|---------------|
| Micro-Price Alpha | `μ > mid + spread_offset`                   | Passive Fill  |
| Toxicity Abort    | `VPIN > 90th-percentile`                    | Flatten + Hold|
| Neutral           | Neither condition met                       | Hold          |

This is the exact decision framework used on real FI desks — the
difference being that production systems operate on co-located hardware
with sub-microsecond latencies. This engine is the **research harness**
that validates the signal *before* it gets promoted to production.

---

## Architecture

```
                     ┌─────────────────────────────────────────────┐
                     │              main.cpp  (Orchestrator)       │
                     │                                             │
  CSV / Binary ──►   │  FI2010Parser  ──►  LimitOrderBook          │
                     │       │                    │                │
                     │       ▼                    ▼                │
                     │  MicroPrice    OBI    VPIN                  │
                     │       │         │      │                    │
                     │       └────┬────┘      │                    │
                     │            ▼           ▼                    │
                     │      SimulatedTrader                        │
                     │       │           │                         │
                     │       ▼           ▼                         │
                     │  Trade_Signals  Inventory_Risk  ──► Hazelcast│
                     │       │                                     │
                     │       ▼                                     │
                     │  RMSE Analysis  ──►  Console / Matplot      │
                     └─────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

| Dependency       | Required? | Install (macOS)                              |
|------------------|-----------|----------------------------------------------|
| CMake ≥ 3.22     | ✅        | `brew install cmake`                         |
| C++20 compiler   | ✅        | Xcode Command Line Tools (Apple Clang 15+)   |
| Boost            | Optional  | `brew install boost`                         |
| Hazelcast        | Optional  | Docker: `docker run hazelcast/hazelcast`     |
| Matplotplusplus  | Optional  | Auto-fetched by CMake                        |
| Python 3 + NumPy | For `preprocess.py` | `pip install numpy`              |

### Build

```bash
# Minimal build (no optional dependencies)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(sysctl -n hw.ncpu)

# Full build (all features)
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DLOB_ENABLE_BOOST=ON \
      -DLOB_ENABLE_HAZELCAST=ON \
      -DLOB_ENABLE_PLOTTING=ON
cmake --build build -j$(sysctl -n hw.ncpu)
```

### Preprocess FI-2010 Data

```bash
# Download the FI-2010 dataset from:
# https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649

# Convert to binary
python scripts/preprocess.py \
    --input data/fi2010_raw.csv \
    --output data/lob.bin \
    --stats
```

### Run

```bash
# With binary data
./build/lob_engine --file data/lob.bin

# With raw CSV (slower parsing)
./build/lob_engine --csv data/fi2010_raw.csv

# Synthetic smoke test (no data files needed)
./build/lob_engine --synthetic

# Full run with visualization
./build/lob_engine --file data/lob.bin --plot

# With Hazelcast
./build/lob_engine --file data/lob.bin --hazelcast
```

### CLI Options

| Flag              | Description                                  | Default   |
|-------------------|----------------------------------------------|-----------|
| `--csv <path>`    | Load FI-2010 CSV file                        | —         |
| `--file <path>`   | Load pre-processed binary file               | —         |
| `--synthetic`     | Generate 500-tick synthetic LOB              | —         |
| `--hazelcast`     | Enable Hazelcast distributed store           | off       |
| `--plot`          | Show Matplotplusplus charts                  | off       |
| `--spread-offset` | Micro-Price threshold offset                 | 0.0001    |
| `--vpin-bucket`   | VPIN bucket volume                           | 1000      |
| `--vpin-window`   | VPIN rolling window (# buckets)              | 50        |

---

## Research Output

The engine prints three research blocks to stdout:

1. **Execution Summary** — tick count, throughput, trade stats, PnL
2. **RMSE Table** — Micro-Price predictive accuracy at 10 / 50 / 100 tick horizons
3. **VPIN Summary** — distribution statistics + toxicity threshold

---

## Project Structure

```
Micro-Price-LOB/
├── CMakeLists.txt
├── README.md
├── include/
│   ├── features/
│   │   ├── micro_price.hpp      # Stoikov Micro-Price
│   │   ├── obi.hpp              # Order Book Imbalance
│   │   └── vpin.hpp             # VPIN toxic flow detector
│   ├── infra/
│   │   └── hazelcast_store.hpp  # Distributed store interface
│   ├── lob/
│   │   ├── fi2010_parser.hpp    # CSV & binary parser
│   │   ├── order_book.hpp       # LimitOrderBook class
│   │   └── price_level.hpp      # PriceLevel struct
│   ├── stats/
│   │   └── rmse.hpp             # RMSE & statistics
│   └── trading/
│       └── simulated_trader.hpp # Passive Fill + VPIN abort
├── scripts/
│   └── preprocess.py            # FI-2010 → binary converter
└── src/
    ├── fi2010_parser.cpp
    ├── hazelcast_store.cpp
    ├── main.cpp
    ├── simulated_trader.cpp
    ├── visualization.cpp
    └── vpin.cpp
```

---

## Key Algorithms

### Micro-Price (Stoikov 2018)

```
μ = P_ask × (V_bid / (V_bid + V_ask))  +  P_bid × (V_ask / (V_bid + V_ask))
```

### Order Book Imbalance

```
OBI = (Σ V_bid − Σ V_ask) / (Σ V_bid + Σ V_ask)     ∈ [−1, +1]
```

### VPIN (Easley, López de Prado & O'Hara 2012)

```
VPIN = Σ|V_buy(n) − V_sell(n)| / (N × V_bucket)
```

Volume classification uses the **tick rule** on mid-price direction.

---

## License

Research / educational use.  Not for production trading without
appropriate risk management, compliance review, and regulatory approval.
