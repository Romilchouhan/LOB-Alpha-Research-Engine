# Learning Curriculum — Micro-Price-LOB

Study order mirrors the project phases. Goal: defend every line of this repo in an interview.

## Level 0 — Market structure basics (do this first)

**Concepts to nail before anything else:** what an exchange matching engine does; limit vs market orders; the limit order book (LOB); bid/ask, spread, mid-price; market data levels:
- **L1** = best bid/ask only (top of book)
- **L2** = aggregated depth per price level (this is what FI-2010 gives us — 10 levels each side, volume per level, **no individual orders, no trades**)
- **L3** = every individual order + full event stream (adds/cancels/executions)

The single most important fact for this repo: **FI-2010 is L2 snapshots — we never see trades.** Every design decision (dropping VPIN-as-toxicity, adopting OFI) follows from that.

**Videos:**
- Search YouTube: **"How exchanges work matching engine order book"** — any of the CME Group / Interactive Brokers explainer series; watch 2–3, they're short.
- ["Trading Theory: Market Microstructure and Limit Order Book"](https://www.youtube.com/watch?v=pALcn40A_ec) — beginner walkthrough.
- ["Stochastic Market Microstructure Models of Limit Order Books"](https://www.youtube.com/watch?v=XoBjQqMmKoM) — Columbia/Oxford talk; watch after the basics, it bridges to the math.
- Search YouTube: **"Hudson River Trading how the markets work"** and **"Jane Street what is market making"** — practitioner intros to why the book looks the way it does.
- Interactive reading: [Market Microstructure: Order Books & Execution Mechanics](https://mbrenndoerfer.com/writing/market-microstructure-order-book-mechanics) — work the examples.
- [QuantStart HFT I: Introduction to Market Microstructure](https://www.quantstart.com/articles/high-frequency-trading-i-introduction-to-market-microstructure/) — short, free.

**Checkpoint:** you can explain to a friend why the "price" of a stock is actually two prices, and what a 10-level L2 snapshot row (40 numbers) means.

## Level 1 — The features in this repo (P1/P2 code)

Read each paper *next to the header that implements it*.

| Paper | Implements | Priority |
|---|---|---|
| Stoikov (2018), *The Micro-Price* | `include/features/micro_price.hpp` | Read first. The formula is 5 lines; understand why imbalance predicts the next move. |
| Cont, Kukanov, Stoikov (2014), *The Price Impact of Order Book Events* | `include/features/ofi.hpp` | Core. Hand-derive the 4-term OFI formula; the unit tests' comments walk each case. |
| Easley, López de Prado, O'Hara (2012), *Flow Toxicity and Liquidity in a High-Frequency World* | `include/features/vpin.hpp` | Read to understand why VPIN **needs trades** — so you can defend why this repo demoted it to a pedagogical footnote. |

Videos: search YouTube **"order flow imbalance explained"** and **"VPIN flow toxicity"** (QuantInsti and university seminar recordings cover both). For micro-price, Stoikov himself has given recorded talks — search **"Stoikov micro-price talk"**.

Books (reference, don't read cover-to-cover yet):
- Bouchaud, Bonart, Donier, Gould — *Trades, Quotes and Prices*. The modern LOB bible. Chapters on order flow and queue dynamics matter for P6.
- O'Hara — *Market Microstructure Theory*. Skim inventory/information-model chapters.

**Checkpoint:** whiteboard the micro-price and OFI formulas from memory; explain why OFI works on L2 data and VPIN doesn't.

## Level 2 — FI-2010 dataset + labels (P3)

- Ntakaris et al. (2018), *Benchmark Dataset for Mid-Price Forecasting of Limit Order Book Data* — the dataset paper. Memorize the label spec: smoothed-mid 3-class, horizons k ∈ {10,20,50,100}, threshold α.
- Zhang, Zohren, Roberts (2019), *DeepLOB* — the benchmark we reproduce in P4. Know its architecture and reported accuracy per k. Search YouTube: **"DeepLOB Zohren"** — Oxford ML in finance seminars cover it.

## Level 3 — Statistical rigor (P5)

- Grinold & Kahn, *Active Portfolio Management* — Information Coefficient, IC decay, breadth.
- Politis & Romano — stationary/block bootstrap (answer to "how did you pick block length?").
- Newey & West (1987) — HAC standard errors.
- López de Prado, *Advances in Financial Machine Learning* — labeling, CV in finance, why leakage kills backtests. Search YouTube: **"López de Prado 7 reasons backtests fail"** — his recorded lectures are excellent.

## Level 4 — C++/systems (defending the engineering)

- Meyers, *Effective Modern C++* — the idioms in these headers (`[[nodiscard]]`, move semantics, `constexpr`).
- CppCon talks: search **"CppCon low latency trading"** (Carl Cook's "When a Microsecond Is an Eternity" is the classic) — cache lines, `alignas(64)`, why the hot loop avoids heap allocation.
- GoogleTest primer + CTest docs — the test infra added in P1.
- Apache Arrow/Parquet C++ docs — the `include/io/` writer.

## Suggested weekly cadence

1. **Week 1:** Level 0 videos + play with `./build/lob_engine --synthetic --dump-csv`; open the CSV, map columns to concepts.
2. **Week 2:** Stoikov micro-price paper + `micro_price.hpp` + its tests.
3. **Week 3:** CKS 2014 + `ofi.hpp` + hand-verify each OFI unit test on paper.
4. **Week 4:** VPIN paper + why it's mis-specified here; Ntakaris FI-2010 paper before P3 starts.
