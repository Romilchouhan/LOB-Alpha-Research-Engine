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

### 1. Environment Setup

The engine optional features (Hazelcast, Plotting) require specific external services or libraries.

**Docker & Hazelcast**
Hazelcast is used for distributed storage of trade signals and risk metrics.
```bash
# Start Docker (macOS)
open -a Docker

# Run Hazelcast instance
docker run -d --name hazelcast -p 5701:5701 hazelcast/hazelcast

# Run Hazelcast Management Center (UI for monitoring)
# Access at http://localhost:8080
docker run -d --name hz-mc -p 8080:8080 hazelcast/management-center

# Hazelcast Cluster Definition
# Default Name: dev
# Default Port: 5701
# Verification: docker logs hazelcast | grep "Cluster Name"
```

**System Dependencies (macOS)**
```bash
brew install cmake boost
```

### 2. Build the Engine

The engine supports multiple build modes. Enabling optional features will automatically fetch dependencies via `FetchContent`.

| Feature | CMake Flag | Description |
| :--- | :--- | :--- |
| **Minimal** | (Default) | Core LOB logic only. |
| **Boost** | `-DLOB_ENABLE_BOOST=ON` | Uses Boost.Accumulators for high-precision stats. |
| **Hazelcast** | `-DLOB_ENABLE_HAZELCAST=ON` | Enables distributed signal/risk emission. |
| **Plotting** | `-DLOB_ENABLE_PLOTTING=ON` | Enables inline C++ plotting via Matplotplusplus. |

**Standard Build (Recommended)**
```bash
# Configure with all features enabled
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DLOB_ENABLE_BOOST=ON \
      -DLOB_ENABLE_HAZELCAST=ON \
      -DLOB_ENABLE_PLOTTING=ON

# Build using all CPU cores
cmake --build build -j$(sysctl -n hw.ncpu)
```

### 3. Run and Generate Plots

#### Preprocess Data
Download the FI-2010 dataset and convert it to the optimized binary format:
```bash
python3 scripts/preprocess.py --input data/FI-2010.csv --output data/lob.bin --stats
```

#### Run Engine
```bash
# Run with Hazelcast & Real-time Plotting (if enabled in build)
./build/lob_engine --file data/lob.bin --hazelcast --plot

# Generate CSV data for advanced Python visualization
./build/lob_engine --file data/lob.bin --dump-csv data/results.csv
```

#### Advanced Visualizations
Use the Python suite for high-fidelity research charts:
```bash
# Generate dashboard, pnl, and alpha plots
python3 scripts/visualize.py --input data/results.csv --output plots/fi2010/ --dark
```

| Plot Type | Description |
|-----------|-------------|
| **Dashboard** | 4-panel overview (Price, OBI, PnL, VPIN) |
| **Micro-vs-Mid** | Stoikov estimator vs standard mid-price |
| **PnL Curve** | Strategy returns with toxicity abort markers |
| **VPIN** | Real-time toxic flow detection |

---

## Project Structure

```
Micro-Price-LOB/
├── CMakeLists.txt           # Build system (C++20, Apple M2)
├── README.md                # Research engine overview
├── plots/                   # Stored visualizations (Synthetic & FI-2010)
│   ├── synthetic/           # Test run charts
│   └── fi2010/              # Research benchmark charts
├── include/
│   ├── features/
│   │   ├── micro_price.hpp   # Stoikov estimator
│   │   ├── obi.hpp           # Order Book Imbalance
│   │   └── vpin.hpp          # VPIN monitor
│   ├── infra/
│   │   └── hazelcast_store.hpp
│   ├── lob/
│   │   ├── fi2010_parser.hpp # CSV/Binary engine
│   │   ├── order_book.hpp    # LOB data container
│   │   └── price_level.hpp
│   ├── stats/
│   │   └── rmse.hpp          # Predictive accuracy
│   └── trading/
│       └── simulated_trader.hpp
├── scripts/
│   ├── preprocess.py        # FI-2010 data cleaner
│   └── visualize.py         # Advanced plot generator
└── src/
    ├── fi2010_parser.cpp     # Data engine implementation
    ├── hazelcast_store.cpp   # Distributed client logic
    ├── main.cpp              # Orchestrator
    ├── simulated_trader.cpp  # Strategy logic
    ├── visualization.cpp     # C++ plotting wrappers
    └── vpin.cpp              # Toxicity detector
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
