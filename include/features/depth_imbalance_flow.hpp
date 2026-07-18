#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// DepthImbalanceFlow — bucketed depth-imbalance flow indicator
// ─────────────────────────────────────────────────────────────────────────────
//
// The math is the VPIN bucket formula (Easley, López de Prado & O'Hara, 2012)
// REPURPOSED as a plain feature — NOT a toxicity signal (see caveat below):
//   1. Attribute each snapshot's BBO volume to buy/sell by mid-price direction
//      (tick rule).
//   2. Accumulate signed volume into fixed-size buckets.
//   3. value = mean(|V_buy − V_sell|) / bucket_size   over a rolling window.
// ─────────────────────────────────────────────────────────────────────────────

#include <cmath>
#include <cstddef>
#include <deque>
#include <limits>
#include <vector>

namespace features {

/// DepthImbalanceFlow — bucketed depth-imbalance flow indicator.
///
/// DATA CAVEAT (FI-2010): FI-2010 is L2 snapshot data with no trade prints.
/// The tick-rule classification here operates on a depth-change proxy
/// (BBO volume attributed by mid-price direction), not true executed trades.
/// This is why the class is NOT called VPIN: without a trade feed the VPIN
/// toxicity interpretation is mis-specified. The values measure snapshot-volume
/// asymmetry, not informed order flow. Kept as a plain feature for comparison
/// against OFI (Order Flow Imbalance), which IS well-defined on L2 data.
class DepthImbalanceFlow {
public:
    /// @param bucket_volume  Total volume per bucket.
    /// @param window_buckets Number of buckets in the rolling window.
    explicit DepthImbalanceFlow(double bucket_volume = 1000.0,
                  std::size_t window_buckets = 50);

    /// Feed one tick.  `volume` is the tick's traded volume,
    /// `mid_price` is the current mid-price (for tick-rule classification).
    /// Returns the latest DepthImbalanceFlow value (NaN if insufficient data).
    double update(double volume, double mid_price);

    /// Current DepthImbalanceFlow value.
    [[nodiscard]] double value()       const noexcept { return current_dif_; }

    /// Full history of DepthImbalanceFlow values (one per completed bucket after warm-up).
    [[nodiscard]] const std::vector<double>& history() const noexcept { return history_; }

    /// Compute the p-th percentile of the historical DepthImbalanceFlow distribution.
    [[nodiscard]] double percentile(double p) const;

    /// True if the latest DepthImbalanceFlow exceeds the p-th percentile of history.
    [[nodiscard]] bool is_elevated(double p = 90.0) const;

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

    double              current_dif_ = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> history_;
};

} // namespace features
