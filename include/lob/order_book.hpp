#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// LimitOrderBook — 10-level depth view built from LOBSnapshot
// ─────────────────────────────────────────────────────────────────────────────

#include "lob/price_level.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numeric>

namespace lob {

class LimitOrderBook {
public:
    // ── Snapshot ingestion ───────────────────────────────────────────────
    void update(const LOBSnapshot& snap) noexcept {
        snap_ = snap;
        ++tick_;
    }

    // ── Accessors ────────────────────────────────────────────────────────
    [[nodiscard]] const LOBSnapshot& snapshot()   const noexcept { return snap_; }
    [[nodiscard]] std::uint64_t      tick_count()  const noexcept { return tick_; }

    [[nodiscard]] const PriceLevel& best_ask() const noexcept { return snap_.asks[0]; }
    [[nodiscard]] const PriceLevel& best_bid() const noexcept { return snap_.bids[0]; }

    // ── Derived quantities ───────────────────────────────────────────────
    [[nodiscard]] double mid_price() const noexcept {
        return 0.5 * (best_ask().price + best_bid().price);
    }

    [[nodiscard]] double spread() const noexcept {
        return best_ask().price - best_bid().price;
    }

    /// Total volume on one side up to `depth` levels.
    [[nodiscard]] double total_volume_ask(int depth = kMaxDepth) const noexcept {
        double sum = 0.0;
        for (int i = 0; i < std::min(depth, kMaxDepth); ++i)
            sum += snap_.asks[i].volume;
        return sum;
    }

    [[nodiscard]] double total_volume_bid(int depth = kMaxDepth) const noexcept {
        double sum = 0.0;
        for (int i = 0; i < std::min(depth, kMaxDepth); ++i)
            sum += snap_.bids[i].volume;
        return sum;
    }

private:
    LOBSnapshot    snap_{};
    std::uint64_t  tick_ = 0;
};

} // namespace lob
