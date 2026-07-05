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
};

} // namespace features
