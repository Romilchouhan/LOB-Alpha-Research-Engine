#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// Realized Volatility — rolling stdev of log-mid returns
// ─────────────────────────────────────────────────────────────────────────────
//
//   r_t  = ln(mid_t / mid_{t−1})
//   rv_t = population stdev of { r_{t−W+1}, …, r_t }     (W = window)
//
// Strictly BACKWARD-looking: the value at tick t uses only returns up to and
// including t. Warm-up (fewer than W returns accumulated) returns NaN so that
// downstream Python can simply drop warm-up rows.
//
// O(1) per update via running Σr and Σr² (Welford-free two-sum form; windows
// of 50–200 ticks keep the numeric error negligible for LOB-scale returns).
// ─────────────────────────────────────────────────────────────────────────────

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <limits>

namespace features {

class RealizedVolatility {
public:
    /// @param window  Number of returns in the rolling window (clamped to ≥ 1).
    explicit RealizedVolatility(std::size_t window = 50) noexcept
        : window_(std::max<std::size_t>(window, 1)) {}

    /// Feed the next mid-price (must be > 0). Returns the rolling stdev of
    /// log returns, or NaN during warm-up.
    double update(double mid) {
        if (!has_prev_ || prev_mid_ <= 0.0 || mid <= 0.0) {
            has_prev_ = true;
            prev_mid_ = mid;
            value_    = kNaN;
            return value_;
        }
        const double r = std::log(mid / prev_mid_);
        prev_mid_ = mid;

        returns_.push_back(r);
        sum_    += r;
        sum_sq_ += r * r;
        if (returns_.size() > window_) {
            const double old = returns_.front();
            returns_.pop_front();
            sum_    -= old;
            sum_sq_ -= old * old;
        }

        if (returns_.size() < window_) {
            value_ = kNaN;                              // warm-up
        } else {
            const double n    = static_cast<double>(returns_.size());
            const double mean = sum_ / n;
            const double var  = std::max(0.0, sum_sq_ / n - mean * mean);
            value_ = std::sqrt(var);
        }
        return value_;
    }

    /// Latest realized volatility (NaN during warm-up).
    [[nodiscard]] double value() const noexcept { return value_; }

    [[nodiscard]] std::size_t window() const noexcept { return window_; }

    void reset() noexcept {
        has_prev_ = false;
        prev_mid_ = 0.0;
        returns_.clear();
        sum_ = sum_sq_ = 0.0;
        value_ = kNaN;
    }

private:
    static constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

    std::size_t window_;

    bool   has_prev_ = false;
    double prev_mid_ = 0.0;

    std::deque<double> returns_;
    double sum_    = 0.0;
    double sum_sq_ = 0.0;
    double value_  = kNaN;
};

} // namespace features
