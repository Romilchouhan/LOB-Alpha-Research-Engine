#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// Micro-Price — Stoikov (2018) weighted mid-price estimator
// ─────────────────────────────────────────────────────────────────────────────
//
// μ = P_ask × (V_bid / (V_bid + V_ask)) + P_bid × (V_ask / (V_bid + V_ask))
//
// Intuition: when bid volume dominates the BBO, the "fair" price leans toward
// the ask because buying pressure exceeds selling pressure.
// ─────────────────────────────────────────────────────────────────────────────

#include "lob/order_book.hpp"

namespace features {

class MicroPrice {
public:
    /// Compute Stoikov Micro-Price from the current book state.
    [[nodiscard]]
    static double compute(const lob::LimitOrderBook& book) noexcept {
        const double pa = book.best_ask().price;
        const double pb = book.best_bid().price;
        const double va = book.best_ask().volume;
        const double vb = book.best_bid().volume;

        const double denom = va + vb;
        if (denom <= 0.0) [[unlikely]]
            return book.mid_price();

        return pa * (vb / denom) + pb * (va / denom);
    }

    /// Multi-level Micro-Price using top-N levels of depth.
    [[nodiscard]]
    static double compute_multilevel(const lob::LimitOrderBook& book,
                                     int depth = lob::kMaxDepth) noexcept {
        const auto& snap = book.snapshot();
        double num = 0.0, total_v = 0.0;

        const int d = std::min(depth, lob::kMaxDepth);
        for (int i = 0; i < d; ++i) {
            const double va = snap.asks[i].volume;
            const double vb = snap.bids[i].volume;
            num += snap.asks[i].price * vb + snap.bids[i].price * va;
            total_v += va + vb;
        }
        if (total_v <= 0.0) [[unlikely]]
            return book.mid_price();

        return num / total_v;
    }
};

} // namespace features
