#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// Queue Imbalance — per-level order book imbalance, levels 1..10
// ─────────────────────────────────────────────────────────────────────────────
//
//   QI_i = (V_b,i − V_a,i) / (V_b,i + V_a,i)      i = 1..kMaxDepth
//
// Range [−1, +1] per level; a zero denominator (both queues empty) maps to 0.
// ─────────────────────────────────────────────────────────────────────────────

#include "lob/order_book.hpp"

#include <array>

namespace features {

class QueueImbalance {
public:
    [[nodiscard]]
    static std::array<double, lob::kMaxDepth>
    compute(const lob::LimitOrderBook& book) noexcept {
        const auto& snap = book.snapshot();
        std::array<double, lob::kMaxDepth> out{};
        for (int i = 0; i < lob::kMaxDepth; ++i) {
            const double vb    = snap.bids[i].volume;
            const double va    = snap.asks[i].volume;
            const double denom = vb + va;
            out[static_cast<std::size_t>(i)] =
                (denom <= 0.0) ? 0.0 : (vb - va) / denom;
        }
        return out;
    }
};

} // namespace features
