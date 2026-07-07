#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// Book Slope — OLS steepness of the depth profile, per side
// ─────────────────────────────────────────────────────────────────────────────
//
// For each side, regress cumulative resting size y_i against the price
// distance x_i of level i from the mid:
//
//   bids:  x_i = mid − P_b,i        asks:  x_i = P_a,i − mid
//   y_i   = Σ_{k ≤ i} V_k           (cumulative volume out to level i)
//   slope = Σ(x−x̄)(y−ȳ) / Σ(x−x̄)²
//
// A steep slope ⇒ volume accumulates quickly away from the touch ⇒ thick,
// liquid book. Unpopulated levels (price ≤ 0) are skipped. Degenerate cases
// (fewer than 2 valid levels, or zero x-variance) return 0.
// ─────────────────────────────────────────────────────────────────────────────

#include "lob/order_book.hpp"

namespace features {

struct BookSlopeResult {
    double bid_slope = 0.0;
    double ask_slope = 0.0;
};

class BookSlope {
public:
    /// Compute per-side OLS slope across levels 1..kMaxDepth.
    [[nodiscard]]
    static BookSlopeResult compute(const lob::LimitOrderBook& book) noexcept {
        const auto&  snap = book.snapshot();
        const double mid  = book.mid_price();

        BookSlopeResult r;
        r.bid_slope = side_slope(snap.bids, mid, /*is_bid=*/true);
        r.ask_slope = side_slope(snap.asks, mid, /*is_bid=*/false);
        return r;
    }

private:
    static double side_slope(const lob::PriceLevel (&levels)[lob::kMaxDepth],
                             double mid, bool is_bid) noexcept {
        // Single pass: collect (x, cumulative volume) for populated levels.
        double xs[lob::kMaxDepth];
        double ys[lob::kMaxDepth];
        int    n     = 0;
        double cum_v = 0.0;

        for (int i = 0; i < lob::kMaxDepth; ++i) {
            if (levels[i].price <= 0.0) continue;      // unpopulated level
            cum_v += levels[i].volume;
            xs[n] = is_bid ? (mid - levels[i].price) : (levels[i].price - mid);
            ys[n] = cum_v;
            ++n;
        }
        if (n < 2) return 0.0;

        double mean_x = 0.0, mean_y = 0.0;
        for (int i = 0; i < n; ++i) { mean_x += xs[i]; mean_y += ys[i]; }
        mean_x /= n;
        mean_y /= n;

        double sxx = 0.0, sxy = 0.0;
        for (int i = 0; i < n; ++i) {
            const double dx = xs[i] - mean_x;
            sxx += dx * dx;
            sxy += dx * (ys[i] - mean_y);
        }
        if (sxx <= 0.0) return 0.0;                    // zero x-variance
        return sxy / sxx;
    }
};

} // namespace features
