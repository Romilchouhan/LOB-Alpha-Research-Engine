#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// Order Book Imbalance (OBI)
// ─────────────────────────────────────────────────────────────────────────────
//
// OBI = (Σ V_bid − Σ V_ask) / (Σ V_bid + Σ V_ask)
//
// Range: [−1, +1]
//   +1 → all volume on bid side (buying pressure)
//   −1 → all volume on ask side (selling pressure)
// ─────────────────────────────────────────────────────────────────────────────

#include "lob/order_book.hpp"

namespace features {

class OrderBookImbalance {
public:
    /// Compute OBI across `depth` levels (default = full 10 levels).
    [[nodiscard]]
    static double compute(const lob::LimitOrderBook& book,
                          int depth = lob::kMaxDepth) noexcept {
        const double vb = book.total_volume_bid(depth);
        const double va = book.total_volume_ask(depth);
        const double denom = vb + va;
        if (denom <= 0.0) [[unlikely]]
            return 0.0;
        return (vb - va) / denom;
    }

    /// Per-level OBI (returns imbalance at a single price level index).
    [[nodiscard]]
    static double compute_at_level(const lob::LimitOrderBook& book,
                                   int level = 0) noexcept {
        const auto& snap = book.snapshot();
        const double vb = snap.bids[level].volume;
        const double va = snap.asks[level].volume;
        const double denom = vb + va;
        if (denom <= 0.0) [[unlikely]]
            return 0.0;
        return (vb - va) / denom;
    }
};

} // namespace features
