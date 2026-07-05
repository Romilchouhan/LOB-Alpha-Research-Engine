#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// VPIN — Volume-Synchronized Probability of Informed Trading
// ─────────────────────────────────────────────────────────────────────────────
//
// Algorithm (Easley, López de Prado & O'Hara, 2012):
//   1. Classify each tick's volume as buy or sell (tick rule on mid-price).
//   2. Accumulate buy/sell volume into fixed-size buckets.
//   3. VPIN = mean(|V_buy − V_sell|) / bucket_size   over a rolling window.
//
// High VPIN ⇒ toxic flow (informed traders dominating).
// ─────────────────────────────────────────────────────────────────────────────

#include <cmath>
#include <cstddef>
#include <deque>
#include <limits>
#include <vector>

namespace features {

/// VPIN — Volume-Synchronized Probability of Informed Trading.
///
/// DATA CAVEAT (FI-2010): FI-2010 is L2 snapshot data with no trade prints.
/// The tick-rule classification here operates on a depth-change proxy
/// (BBO volume attributed by mid-price direction), not true executed trades.
/// Consequently the VPIN values produced on this dataset are NOT a valid
/// flow-toxicity metric — they measure snapshot-volume asymmetry, not order
/// flow. The implementation is kept for pedagogical comparison against OFI
/// (Order Flow Imbalance, coming in P2), which is well-defined on L2 data.
class VPIN {
public:
    /// @param bucket_volume  Total volume per bucket.
    /// @param window_buckets Number of buckets in the rolling window.
    explicit VPIN(double bucket_volume = 1000.0,
                  std::size_t window_buckets = 50);

    /// Feed one tick.  `volume` is the tick's traded volume,
    /// `mid_price` is the current mid-price (for tick-rule classification).
    /// Returns the latest VPIN value (NaN if insufficient data).
    double update(double volume, double mid_price);

    /// Current VPIN value.
    [[nodiscard]] double value()       const noexcept { return current_vpin_; }

    /// Full history of VPIN values (one per completed bucket after warm-up).
    [[nodiscard]] const std::vector<double>& history() const noexcept { return history_; }

    /// Compute the p-th percentile of the historical VPIN distribution.
    [[nodiscard]] double percentile(double p) const;

    /// True if the latest VPIN exceeds the p-th percentile of history.
    [[nodiscard]] bool is_toxic(double p = 90.0) const;

    void reset();

private:
    void flush_bucket_helper(double buy, double sell);

    double        bucket_volume_;
    std::size_t   window_buckets_;

    double        prev_mid_ = 0.0;
    bool          has_prev_ = false;

    // Current bucket accumulators
    double        cur_buy_  = 0.0;
    double        cur_sell_ = 0.0;
    double        cur_total_= 0.0;

    // Completed buckets in the rolling window
    struct Bucket { double buy; double sell; };
    std::deque<Bucket> buckets_;

    double              current_vpin_ = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> history_;
};

} // namespace features
