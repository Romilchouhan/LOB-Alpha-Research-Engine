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
#include <vector>

namespace features {

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
