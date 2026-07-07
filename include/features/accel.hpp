#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// Mid-Price Acceleration — second difference of the mid-price
// ─────────────────────────────────────────────────────────────────────────────
//
//   a_t = mid_t − 2·mid_{t−1} + mid_{t−2}
//
// Strictly backward-looking. The first two ticks (insufficient history)
// return NaN, consistent with RealizedVolatility's warm-up convention, so
// downstream Python can drop warm-up rows uniformly.
// ─────────────────────────────────────────────────────────────────────────────

#include <cstddef>
#include <limits>

namespace features {

class MidAcceleration {
public:
    /// Feed the next mid-price; returns a_t (NaN for the first two ticks).
    double update(double mid) noexcept {
        double a = std::numeric_limits<double>::quiet_NaN();
        if (count_ >= 2)
            a = mid - 2.0 * m1_ + m2_;
        m2_ = m1_;
        m1_ = mid;
        if (count_ < 2) ++count_;
        value_ = a;
        return a;
    }

    /// Latest acceleration (NaN until three mids have been observed).
    [[nodiscard]] double value() const noexcept { return value_; }

    void reset() noexcept {
        count_ = 0;
        m1_ = m2_ = 0.0;
        value_ = std::numeric_limits<double>::quiet_NaN();
    }

private:
    std::size_t count_ = 0;
    double      m1_    = 0.0;   // mid_{t−1}
    double      m2_    = 0.0;   // mid_{t−2}
    double      value_ = std::numeric_limits<double>::quiet_NaN();
};

} // namespace features
