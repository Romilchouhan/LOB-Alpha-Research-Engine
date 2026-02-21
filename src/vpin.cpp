// ─────────────────────────────────────────────────────────────────────────────
// VPIN — Implementation
// ─────────────────────────────────────────────────────────────────────────────

#include "features/vpin.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace features {

VPIN::VPIN(double bucket_volume, std::size_t window_buckets)
    : bucket_volume_(bucket_volume)
    , window_buckets_(window_buckets)
{}

double VPIN::update(double volume, double mid_price) {
    // ── Tick-rule classification ────────────────────────────────────────
    // If mid_price went up since last tick → this volume is "buy-initiated".
    // If mid_price went down → "sell-initiated".
    // If unchanged, split 50/50.
    double buy_vol = 0.0, sell_vol = 0.0;

    if (!has_prev_) {
        // First tick: split evenly
        buy_vol  = volume * 0.5;
        sell_vol = volume * 0.5;
        has_prev_ = true;
    } else {
        const double delta = mid_price - prev_mid_;
        if (delta > 0.0) {
            buy_vol = volume;
        } else if (delta < 0.0) {
            sell_vol = volume;
        } else {
            buy_vol  = volume * 0.5;
            sell_vol = volume * 0.5;
        }
    }
    prev_mid_ = mid_price;

    // ── Accumulate into current bucket ──────────────────────────────────
    cur_buy_   += buy_vol;
    cur_sell_  += sell_vol;
    cur_total_ += volume;

    // Check if the bucket is full (may overflow into the next bucket)
    while (cur_total_ >= bucket_volume_) {
        // Proportion of this bucket that fits
        const double overflow = cur_total_ - bucket_volume_;
        const double ratio    = (overflow > 0.0)
                                    ? (bucket_volume_ / cur_total_)
                                    : 1.0;

        const double bucket_buy  = cur_buy_  * ratio;
        const double bucket_sell = cur_sell_ * ratio;

        flush_bucket_helper(bucket_buy, bucket_sell);

        // Carry-over remainder
        cur_buy_   -= bucket_buy;
        cur_sell_  -= bucket_sell;
        cur_total_ -= bucket_volume_;
    }

    return current_vpin_;
}

void VPIN::flush_bucket_helper(double buy, double sell) {
    buckets_.push_back({buy, sell});
    if (buckets_.size() > window_buckets_)
        buckets_.pop_front();

    if (buckets_.size() >= window_buckets_) {
        double sum_abs = 0.0;
        for (const auto& b : buckets_)
            sum_abs += std::abs(b.buy - b.sell);
        current_vpin_ = sum_abs / (static_cast<double>(window_buckets_) * bucket_volume_);
        history_.push_back(current_vpin_);
    }
}

double VPIN::percentile(double p) const {
    if (history_.empty()) return 0.0;
    std::vector<double> sorted = history_;
    std::sort(sorted.begin(), sorted.end());
    const double idx = (p / 100.0) * static_cast<double>(sorted.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(idx);
    const std::size_t hi = std::min(lo + 1, sorted.size() - 1);
    const double frac = idx - static_cast<double>(lo);
    return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

bool VPIN::is_toxic(double p) const {
    if (std::isnan(current_vpin_) || history_.size() < 10) return false;
    return current_vpin_ >= percentile(p);
}

void VPIN::reset() {
    prev_mid_  = 0.0;
    has_prev_  = false;
    cur_buy_   = 0.0;
    cur_sell_  = 0.0;
    cur_total_ = 0.0;
    current_vpin_ = std::numeric_limits<double>::quiet_NaN();
    buckets_.clear();
    history_.clear();
}

} // namespace features
